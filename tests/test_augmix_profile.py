"""CPU-only diagnostic semantics and containment contracts; no official arrays."""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone, timedelta
import importlib
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research import profile_augmix as profile


POLICY = {"concurrent_local_gpu_jobs": 1, "local_gpu_seconds_per_utc_day": 21600,
          "local_job_timeout_seconds": 10800, "stops_at": "2026-11-05T00:00:00-05:00",
          "baseline_benchmark": "toy_baseline.json"}
NOW = datetime(2026, 10, 31, 23, 10, tzinfo=timezone.utc)


class AugMixProfileTests(unittest.TestCase):
    def toy(self):
        sentinel, calls, ranges = object(), [], []
        class Pipeline:
            def __next__(self):
                calls.append("queue")
                return sentinel
        class Model:
            def forward(self, value):
                calls.append("forward")
                return value
        engine = SimpleNamespace(EpochPipeline=Pipeline)
        def loss(value):
            calls.append("loss")
            return value
        def optimizer(value):
            calls.append("optimizer")
            return value
        def step(value):
            return engine.optimizer_step(engine.augmix_loss(Model().forward(value)))
        def overflow(value):
            return engine.train_step(value)
        engine.augmix_loss, engine.optimizer_step = loss, optimizer
        engine.train_step, engine.overflow_smoke = step, overflow
        @contextmanager
        def record(name):
            ranges.append(("enter", name))
            try:
                yield
            finally:
                ranges.append(("exit", name))
        return engine, Model, sentinel, calls, ranges, record

    def test_wrappers_preserve_result_call_order_and_restore_all_boundaries(self):
        engine, model, token, calls, ranges, record = self.toy()
        originals = (engine.EpochPipeline.__next__, model.forward, engine.augmix_loss,
                     engine.optimizer_step, engine.train_step, engine.overflow_smoke)
        with profile.instrument_boundaries(engine, model, record) as samples:
            self.assertIs(next(engine.EpochPipeline()), token)
            self.assertIs(engine.train_step(token), token)
            self.assertIs(engine.overflow_smoke(token), token)
        self.assertEqual(calls, ["queue", "forward", "loss", "optimizer", "forward", "loss", "optimizer"])
        self.assertEqual(originals, (engine.EpochPipeline.__next__, model.forward, engine.augmix_loss,
                     engine.optimizer_step, engine.train_step, engine.overflow_smoke))
        self.assertTrue(any(v[1] == "augmix_profile.overflow.loss" for v in ranges))
        self.assertEqual(len([v for v in ranges if v[0] == "enter"]), len(samples))
        self.assertTrue(all(v["wall_seconds"] >= 0 for v in samples))

    def test_exception_restores_callables_and_emits_failed_range_without_swallowing(self):
        engine, model, token, _, ranges, record = self.toy()
        def failed(*_): raise RuntimeError("toy")
        engine.augmix_loss = failed
        originals = (model.forward, engine.augmix_loss, engine.train_step)
        with self.assertRaisesRegex(RuntimeError, "toy"):
            with profile.instrument_boundaries(engine, model, record):
                engine.train_step(token)
        self.assertEqual(originals, (model.forward, engine.augmix_loss, engine.train_step))
        self.assertEqual(len([x for x in ranges if x[0] == "enter"]),
                         len([x for x in ranges if x[0] == "exit"]))

    def test_partial_instrumentation_setup_failure_restores_earlier_patches(self):
        engine, model, _, _, _, record = self.toy()
        original = model.forward
        del engine.optimizer_step
        with self.assertRaises(AttributeError):
            with profile.instrument_boundaries(engine, model, record): pass
        self.assertIs(model.forward, original)

    def test_normal_wall_aggregates_omit_exactly_three_warmups_and_separate_overflow(self):
        rows = [{"phase": "normal", "step": i, "stage": "queue_wait", "wall_seconds": i/100}
                for i in range(1, 24)]
        rows.append({"phase": "overflow", "step": 23, "stage": "loss", "wall_seconds": .5})
        result = profile.wall_summary(rows)
        self.assertEqual(result["normal.queue_wait"]["calls"], 20)
        self.assertEqual(result["normal.queue_wait"]["min_seconds"], .04)
        self.assertEqual(result["overflow.loss"]["calls"], 1)
        self.assertAlmostEqual(result["normal.queue_wait"]["total_seconds"], sum(i/100 for i in range(4,24)))

    def test_wall_rejects_nonfinite_metrics(self):
        with self.assertRaises(ValueError):
            profile.wall_summary([{"phase":"normal", "step":4, "stage":"queue", "wall_seconds":float("nan")}])

    def test_operator_summary_exports_only_bounded_aggregate_fields(self):
        def event(name, count, cpu, device):
            return SimpleNamespace(key=name, count=count, cpu_time_total=cpu+100,
                self_cpu_time_total=cpu, device_time_total=device+100, self_device_time_total=device,
                input_shapes=["never exported"], stack=["private path"], tensor_values=[123])
        result = profile.operator_summary([event("aten::_local_scalar_dense",2,2000,0),
            event("augmix_profile.normal.loss",1,1000,3000),event("aten::conv",10,400,5000)], limit=1)
        self.assertEqual(result["top_self_cpu"][0]["name"],"aten::_local_scalar_dense")
        self.assertEqual(result["top_self_device"][0]["device_self_ms"],5.)
        self.assertEqual(result["instrumented_ranges"][0]["calls"],1)
        self.assertNotIn("private",json.dumps(result))
        self.assertNotIn("tensor_values",json.dumps(result))
        self.assertEqual(set(result["top_self_cpu"][0]),
            {"name","calls","cpu_total_ms","cpu_self_ms","device_total_ms","device_self_ms"})

    def test_operator_summary_rejects_invalid_times(self):
        with self.assertRaises(ValueError):
            profile.operator_summary([SimpleNamespace(key="bad",count=1,cpu_time_total=float("inf"))])

    def test_actual_CPU_profiler_aggregate_interface_without_raw_trace_or_GPU_activity(self):
        import torch
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=False,profile_memory=False,with_stack=False) as recorded:
            with torch.profiler.record_function("augmix_profile.normal.toy"):
                result=torch.tensor([2.,3.])*2
        self.assertTrue(torch.equal(result,torch.tensor([4.,6.])))
        summary=profile.operator_summary(recorded.key_averages())
        self.assertTrue(any(row["name"]=="aten::mul" for row in summary["top_self_cpu"]))
        self.assertEqual(summary["instrumented_ranges"][0]["calls"],1)
        self.assertEqual(summary["instrumented_ranges"][0]["device_total_ms"],0.)
        observed=profile.cuda_activity_summary(recorded.events())
        self.assertFalse(observed["cuda_activity_observed"])
        self.assertEqual(observed["positive_device_event_count"],0)
        self.assertFalse(observed["cuda_decomposition_available"])

    def test_CUDA_capture_requires_observed_positive_device_activities_not_capability(self):
        events=[SimpleNamespace(device_type="DeviceType.CPU",device_time_total=200),
            SimpleNamespace(device_type="DeviceType.CUDA",device_time_total=0),
            SimpleNamespace(device_type="DeviceType.CUDA",device_time_total=10),
            SimpleNamespace(device_type="DeviceType.CUDA",device_time_total=5)]
        observed=profile.cuda_activity_summary(events)
        self.assertTrue(observed["cuda_activity_observed"])
        self.assertEqual(observed["positive_device_event_count"],2)
        self.assertFalse(profile.cuda_activity_summary(events[:2])["cuda_decomposition_available"])

    def test_budget_caps_180_day_remaining_and_hard_stop(self):
        self.assertEqual(profile.budget_seconds(POLICY,NOW,0),180)
        self.assertEqual(profile.budget_seconds(POLICY,NOW,21590),10)
        stop=datetime.fromisoformat(POLICY["stops_at"])
        self.assertEqual(profile.budget_seconds(POLICY,stop-timedelta(seconds=2),0),2)

    def test_budget_rejects_exhausted_nonfinite_changed_policy_and_passed_stop(self):
        for used in (21600,float("nan"),-1,True):
            with self.subTest(used=used),self.assertRaises(ValueError):
                profile.budget_seconds(POLICY,NOW,used)
        with self.assertRaises(ValueError):
            profile.budget_seconds({**POLICY,"concurrent_local_gpu_jobs":2},NOW,0)
        with self.assertRaises(ValueError):
            profile.budget_seconds(POLICY,datetime.fromisoformat(POLICY["stops_at"]),0)

    def test_source_snapshot_binds_fixed_method_config_and_diagnostic_itself(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            names=["method.py",profile.CONFIG,"research/profile_augmix.py","research/run_local.py","research/campaign.json"]
            for name in names:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b"fixed\n")
            digest=profile.sha256(root/"method.py")
            with patch.object(profile,"PINNED_SOURCES",{"method.py":digest}),patch.object(profile,"CONFIG_SHA",digest):
                result=profile.source_snapshot(root)
                self.assertIn("research/profile_augmix.py",result)
                (root/"method.py").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError,"pipeline changed"):profile.source_snapshot(root)
                (root/"method.py").write_bytes(b"fixed\n")
                (root/profile.CONFIG).write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError,"configuration changed"):profile.source_snapshot(root)

    def test_no_arguments_is_help_only_and_worker_without_manifest_is_refused(self):
        with patch.object(profile,"parent") as parent,patch.object(profile,"worker") as worker,patch("sys.stdout",new_callable=io.StringIO):
            profile.main([])
            parent.assert_not_called();worker.assert_not_called()
        with self.assertRaises(ValueError):profile.main(["--_worker","."])

    def test_direct_worker_requires_exact_manifest_before_importing_training(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);(folder/"profile_inputs.json").write_text("{}")
            with self.assertRaisesRegex(ValueError,"manifest changed"):
                profile.worker(folder,"0"*64)

    @contextmanager
    def active_worker_contract(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root=Path(temp);folder=root/"results/research/toy_profile";folder.mkdir(parents=True)
            guard=root/"research/process_tree_guard.py";guard.parent.mkdir()
            guard.write_bytes((profile.ROOT/"research/process_tree_guard.py").read_bytes())
            manifest={"source_sha256":{"toy":"bound"},"input_sha256":{}}
            (folder/"profile_inputs.json").write_text(json.dumps(manifest))
            digest=profile.sha256(folder/"profile_inputs.json")
            created=time.time()-10;deadline=time.time()+100
            parent=SimpleNamespace(pid=202,create_time=lambda:created-1)
            worker=SimpleNamespace(pid=101,create_time=lambda:created)
            receipt={"schema":profile.SCHEMA,"status":"profiling","pid":202,"process_created_at":created-1,
                "child_pid":101,"child_created_at":created,"deadline_unix":deadline,
                "started_at":datetime.fromtimestamp(deadline-120,timezone.utc).isoformat(),
                "reserved_seconds":120,"input_manifest_sha256":digest}
            lease={"pid":202,"process_created_at":created-1}
            lease_path=root/"results/research/local_lease.json";lease_path.write_text(json.dumps(lease))
            (folder/"supervisor.json").write_text(json.dumps(receipt))
            nonce="a"*32;directory=folder/("process_tree_"+nonce);directory.mkdir()
            spec={"schema":"owned_python_process_tree_v1","nonce":nonce,"host_pid":202,
                "host_created_at":created-1,"deadline_unix":deadline-.1,"script":str(Path(profile.__file__).resolve()),
                "script_sha256":profile.sha256(profile.__file__),"arguments":["--_worker",str(folder),"--_manifest-sha",digest],
                "cwd":str(root),"guard_source_sha256":profile.PINNED_SOURCES["research/process_tree_guard.py"],
                "job_name":"Local\\SaberAugMix_"+nonce if profile.os.name=="nt" else None}
            ready={"schema":spec["schema"],"nonce":nonce,"pid":101,"created_at":created,
                "script_sha256":spec["script_sha256"],
                "method":"windows_job_v1" if profile.os.name=="nt" else "posix_session_v1",
                "watchdog_pid":303,"watchdog_created_at":created+.1}
            spec_path=directory/"spec.json";(spec_path).write_text(json.dumps(spec))
            ready_path=directory/"ready.json";ready_path.write_text(json.dumps(ready))
            worker.cmdline=lambda:[profile.sys.executable,str(guard),"--bootstrap",str(spec_path)]
            stack.enter_context(patch.object(profile,"ROOT",root))
            stack.enter_context(patch.object(profile,"source_snapshot",return_value={"toy":"bound"}))
            stack.enter_context(patch.object(profile.os,"getppid",return_value=202))
            stack.enter_context(patch.object(profile.psutil,"Process",side_effect=lambda pid=None:worker if pid is None else parent))
            native=stack.enter_context(patch.object(profile,"verify_os_boundary"))
            yield SimpleNamespace(folder=folder,manifest=manifest,digest=digest,receipt=receipt,lease=lease,
                lease_path=lease_path,spec=spec,spec_path=spec_path,ready=ready,ready_path=ready_path,
                worker=worker,parent=parent,native=native)

    def test_worker_positive_canonical_lease_active_guard_contract_precedes_training_import(self):
        with self.active_worker_contract() as case:
            self.assertEqual(profile.validate_worker_owner(case.folder,case.digest),case.manifest)
            case.native.assert_called_once()

    def test_self_consistent_receipt_without_canonical_lease_is_rejected(self):
        with self.active_worker_contract() as case:
            case.lease_path.unlink()
            with self.assertRaisesRegex(ValueError,"lease is missing"):
                profile.validate_worker_owner(case.folder,case.digest)
            case.native.assert_not_called()

    def test_foreign_or_cleanup_unknown_lease_is_rejected_despite_self_consistent_receipt(self):
        for change in ({"pid":999},{"process_created_at":0},{"cleanup_requires_reconciliation":True},
                       {"process_tree_cleanup_verified":False},{"status":"unknown"}):
            with self.subTest(change=change),self.active_worker_contract() as case:
                case.lease_path.write_text(json.dumps({**case.lease,**change}))
                with self.assertRaisesRegex(ValueError,"foreign or requires cleanup"):
                    profile.validate_worker_owner(case.folder,case.digest)
                case.native.assert_not_called()

    def test_non_guard_launch_missing_ready_or_wrong_nonce_is_rejected(self):
        for corruption in ("command","ready","nonce","script","source","deadline"):
            with self.subTest(corruption=corruption),self.active_worker_contract() as case:
                if corruption=="command":
                    case.worker.cmdline=lambda:[profile.sys.executable,str(profile.__file__),"--_worker",str(case.folder)]
                elif corruption=="ready":case.ready_path.unlink()
                elif corruption=="nonce":
                    case.ready["nonce"]="b"*32;case.ready_path.write_text(json.dumps(case.ready))
                else:
                    case.spec[{"script":"script_sha256","source":"guard_source_sha256","deadline":"deadline_unix"}[corruption]]= "0"*64 if corruption!="deadline" else case.receipt["deadline_unix"]+10
                    case.spec_path.write_text(json.dumps(case.spec))
                with self.assertRaises(ValueError):profile.validate_worker_owner(case.folder,case.digest)
                case.native.assert_not_called()

    def test_guard_os_verification_failure_prevents_worker_admission(self):
        with self.active_worker_contract() as case:
            case.native.side_effect=ValueError("native boundary missing")
            with self.assertRaisesRegex(ValueError,"native boundary missing"):
                profile.validate_worker_owner(case.folder,case.digest)

    def test_worker_budget_rejects_handcrafted_long_guard_and_mismatched_manifest(self):
        for change in ({"reserved_seconds":10800},{"deadline_unix":time.time()+1000},
                       {"input_manifest_sha256":"0"*64}):
            with self.subTest(change=change),self.active_worker_contract() as case:
                (case.folder/"supervisor.json").write_text(json.dumps({**case.receipt,**change}))
                with self.assertRaises(ValueError):profile.validate_worker_owner(case.folder,case.digest)
                case.native.assert_not_called()

    def test_read_only_Windows_membership_query_closes_handle_and_rejects_nonmember(self):
        from research import process_tree_guard as guard
        for member in (True,False):
            closed=[]
            def membership(current,handle,result):result._obj.value=int(member);return True
            kernel=SimpleNamespace(OpenJobObjectW=lambda *args:42,GetCurrentProcess=lambda:99,
                IsProcessInJob=membership,CloseHandle=lambda h:closed.append(h))
            with patch.object(guard,"_kernel",return_value=kernel):
                if member:profile.verify_os_boundary({"job_name":"toy"},{},SimpleNamespace(pid=101),platform="nt")
                else:
                    with self.assertRaisesRegex(ValueError,"not inside"):
                        profile.verify_os_boundary({"job_name":"toy"},{},SimpleNamespace(pid=101),platform="nt")
            self.assertEqual(closed,[42])

    def test_POSIX_boundary_requires_live_exact_watcher_in_own_session(self):
        watcher=SimpleNamespace(pid=303,create_time=lambda:50.,status=lambda:"running")
        with patch.object(profile.os,"getpgrp",return_value=101,create=True),patch.object(profile.os,"getsid",return_value=101,create=True),patch.object(profile.os,"getpgid",return_value=101,create=True),patch.object(profile.psutil,"Process",return_value=watcher):
            profile.verify_os_boundary({}, {"watchdog_pid":303,"watchdog_created_at":50.},SimpleNamespace(pid=101),platform="posix")
            with self.assertRaisesRegex(ValueError,"watchdog identity"):
                profile.verify_os_boundary({}, {"watchdog_pid":303,"watchdog_created_at":49.},SimpleNamespace(pid=101),platform="posix")

    @contextmanager
    def fake_parent_environment(self, guarded, *, cleanup=True):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root=Path(temp);anchor=root/profile.ANCHOR;anchor.mkdir(parents=True)
            (anchor/"engineering_benchmark.json").write_text("{}")
            digest=profile.sha256(anchor/"engineering_benchmark.json")
            (anchor/"supervisor.json").write_text(json.dumps({"status":"complete","artifact_sha256":digest,
                "process_tree_cleanup_verified":True,"source_sha256":{"toy":"source"}}))
            (root/"research").mkdir();(root/"research/campaign.json").write_text(json.dumps(POLICY))
            input_file=root/"toy_input";input_file.write_text("fixed")
            stack.enter_context(patch.object(profile,"ROOT",root))
            stack.enter_context(patch.object(profile,"ANCHOR_SHA",digest))
            stack.enter_context(patch.object(profile,"source_snapshot",return_value={"toy":"diagnostic"}))
            local=importlib.import_module("research.run_local")
            supervisor=importlib.import_module("research.run_augmix_local")
            guard=importlib.import_module("research.process_tree_guard")
            engine=importlib.import_module("research.train_augmix")
            stack.enter_context(patch.object(local,"now",return_value=NOW))
            stack.enter_context(patch.object(local,"used_seconds",return_value=100))
            stack.enter_context(patch.object(local,"assert_stopped"))
            stack.enter_context(patch.object(supervisor,"bound_sources",return_value={"toy":"source"}))
            stack.enter_context(patch.object(supervisor,"inputs",return_value={str(input_file):profile.sha256(input_file)}))
            stack.enter_context(patch.object(supervisor,"cleanup_owned_pools",return_value=cleanup))
            stack.enter_context(patch.object(engine,"validate_preflight"))
            mock_guard=stack.enter_context(patch.object(guard,"run_guarded",side_effect=guarded))
            yield root,mock_guard

    def test_parent_calls_exact_guarded_diagnostic_and_releases_only_verified_lease(self):
        def guarded(command,log,timeout,receipt_path,receipt,cwd):
            self.assertIn("--_worker",command);self.assertLessEqual(timeout,180)
            receipt["process_tree_cleanup_verified"]=True
            (receipt_path.parent/"profile_summary.json").write_text(json.dumps({
                "schema":profile.SUMMARY_SCHEMA,"status":"complete","formal_admission_evidence":False,
                "accuracy_measured":False}))
        with self.fake_parent_environment(guarded) as (root,mock_guard):
            result=profile.parent("toy_profile")
            self.assertEqual(result["status"],"complete")
            self.assertFalse((root/"results/research/local_lease.json").exists())
            mock_guard.assert_called_once()
            self.assertFalse(result["submitted"])

    def test_unverified_cleanup_retains_and_locks_the_lease(self):
        def guarded(command,log,timeout,receipt_path,receipt,cwd):
            receipt["process_tree_cleanup_verified"]=False
            raise RuntimeError("toy guarded failure")
        with self.fake_parent_environment(guarded,cleanup=False) as (root,_):
            with self.assertRaisesRegex(RuntimeError,"cleanup is unverified"):
                profile.parent("toy_profile")
            lease=json.loads((root/"results/research/local_lease.json").read_text())
            self.assertTrue(lease["cleanup_requires_reconciliation"])
            self.assertFalse(lease["process_tree_cleanup_verified"])
            receipt=json.loads((root/"results/research/toy_profile/supervisor.json").read_text())
            self.assertEqual(receipt["status"],"cleanup_unverified")

    def test_existing_uncertain_lease_refuses_before_guard(self):
        with self.fake_parent_environment(lambda *_:None) as (root,guarded):
            lease=root/"results/research/local_lease.json"
            lease.write_text(json.dumps({"cleanup_requires_reconciliation":True}))
            with self.assertRaisesRegex(ValueError,"reconciliation"):
                profile.parent("toy_profile")
            guarded.assert_not_called()
            self.assertTrue(lease.exists())


if __name__=="__main__":unittest.main()
