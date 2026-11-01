"""Pure contracts: no Torch, official arrays, GPU, account API or kernel."""
import ast
from contextlib import ExitStack, redirect_stdout
import copy
from datetime import datetime, timedelta, timezone
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research import profile_core_control as profile


class ProfileContracts(unittest.TestCase):
    def test_no_arguments_do_nothing_and_worker_is_rejected_before_torch(self):
        before = set(sys.modules)
        with patch.object(profile,"parent") as parent, patch.object(profile,"worker") as worker, redirect_stdout(io.StringIO()):
            profile.main([])
            parent.assert_not_called(); worker.assert_not_called()
        with tempfile.TemporaryDirectory() as temp, self.assertRaises(ValueError):
            profile.worker(Path(temp),"a"*64)
        self.assertNotIn("torch",set(sys.modules)-before)
        with self.assertRaises(ValueError):
            profile.main(["--_manifest-sha","a"*64])

    def test_frozen_source_closure_and_accounting_are_torch_free(self):
        before = set(sys.modules)
        sources = profile.source_snapshot()
        self.assertEqual(sources["research/preflight_core_control.py"],profile.CORE_SHA)
        self.assertEqual(sources[profile.PLAN],profile.PLAN_SHA)
        self.assertEqual(profile.c.configuration(2026,recipe=profile.RECIPE)["epochs"],300)
        self.assertNotIn("acquire_lease",profile.c.accounting())
        self.assertFalse({"torch","runtime"}&(set(sys.modules)-before))

    def samples(self):
        return [(i,name,.01+i/1000) for i in range(23) for name in profile.STAGES]

    def test_wall_aggregates_require_all23_steps_and_include_every_proof(self):
        rows = self.samples()
        grouped = profile.wall_aggregates(rows)
        self.assertEqual(set(grouped),set(profile.STAGES))
        self.assertTrue(all(r["calls"]==20 for r in grouped.values()))
        self.assertEqual(grouped["original_core_step"]["total_seconds"],sum(.01+i/1000 for i in range(3,23)))
        for changed in (rows[:-1],rows+rows[:1],[(i,n,float("nan")) for i,n,_ in rows]):
            with self.assertRaises(ValueError): profile.wall_aggregates(changed)

    def test_cuda_capability_or_cpu_device_duration_is_not_captured_cuda_activity(self):
        cpu = SimpleNamespace(device_type="DeviceType.CPU",device_time_total=10.)
        zero = SimpleNamespace(device_type="DeviceType.CUDA",device_time_total=0.)
        actual = SimpleNamespace(device_type="DeviceType.CUDA",device_time_total=10.)
        self.assertFalse(profile.p.cuda_activity_summary([cpu,zero])["cuda_decomposition_available"])
        self.assertEqual(profile.p.cuda_activity_summary([cpu,actual])["positive_device_event_count"],1)

    def result_fixture(self):
        manifest = {"source_sha256":{"own.py":"a"*64},"config_sha256":profile.c.WRN300_CONFIGS[2026][1],
                    "manifest_sha256":"b"*64,"source_git_commit":"c"*40,"data_identity":profile.c.DATA_IDENTITY}
        result = dict(schema=profile.RESULT_SCHEMA,status="complete",diagnostic_only=True,formal_admission_evidence=False,
            accuracy_measured=False,confirmation_evaluated=False,hidden_images_loaded=False,external_data_loaded=False,
            recipe=profile.RECIPE,seed=2026,precision="bf16",grad_scaler_enabled=False,grad_scaler_scale=1.,
            parameter_count=4166298,bn_modules=31,batch_size=128,source_examples_seen=2944,unique_source_examples=2944,
            warmup_updates=3,timed_updates=20,successful_updates=23,skipped_updates=0,parameter_changed_steps=23,
            captured_updates=20,excluded_warmup_updates=3,
            ema_changed_steps=23,finite_proof_steps=23,bn_forward_counts=[23]*31,ema_buffer_copies=0,
            source_sha256=manifest["source_sha256"],config_sha256=manifest["config_sha256"],
            input_manifest_sha256=manifest["manifest_sha256"],source_git_commit=manifest["source_git_commit"],data_identity=manifest["data_identity"],
            wall_stages=profile.wall_aggregates(self.samples()),instrumented_whole_step={"calls":20,"median_seconds":.1,"total_seconds":2.},
            operators={k:[] for k in ("top_self_cpu","top_self_device","synchronization_and_copy_operators","instrumented_ranges")},
            cuda_activity=profile.p.cuda_activity_summary([]),interpretation=profile.INTERPRETATION)
        result["operators"]["instrumented_ranges"]=[dict(name="core_profile."+name,calls=20,cpu_total_ms=1.,
            cpu_self_ms=.5,device_total_ms=0.,device_self_ms=0.) for name in profile.STAGES]
        env=dict(device="cuda",gpu="Synthetic fixture",cuda_runtime="test",cudnn=1,packages={},amp_dtype="torch.bfloat16",cudnn_benchmark=True,deterministic_algorithms=False)
        result.update(environment=env,environment_sha256=profile.c.json_sha(env))
        return result,manifest

    def test_result_is_diagnostic_only_exact23_updates_no_tensor_trace_or_admission_fields(self):
        result,manifest=self.result_fixture()
        profile.validate_result(result,manifest)
        for key,value in (("formal_admission_evidence",True),("successful_updates",22),("finite_proof_steps",22),
                          ("captured_updates",23),("excluded_warmup_updates",0),
                          ("batch_size",384),("grad_scaler_scale",256.),("precision","fp16"),("predictions",[]),("trace",{})):
            with self.assertRaises(ValueError): profile.validate_result({**result,key:value},manifest)
        altered=copy.deepcopy(result)
        altered["cuda_activity"]["cuda_activity_observed"]=True
        with self.assertRaises(ValueError): profile.validate_result(altered,manifest)
        altered=copy.deepcopy(result)
        altered["operators"]["top_self_cpu"]=[{"tensor":[1,2]}]
        with self.assertRaises(ValueError): profile.validate_result(altered,manifest)

    def test_capture_ranges_require_twenty_real_calls_even_when_cuda_capture_is_empty(self):
        result,manifest=self.result_fixture()
        self.assertFalse(result["cuda_activity"]["cuda_decomposition_available"])
        profile.validate_result(result,manifest)
        for count in (0,19,23):
            altered=copy.deepcopy(result)
            next(r for r in altered["operators"]["instrumented_ranges"] if r["name"]=="core_profile.original_core_step")["calls"]=count
            with self.assertRaises(ValueError): profile.validate_result(altered,manifest)
        altered=copy.deepcopy(result)
        altered["operators"]["instrumented_ranges"].pop()
        with self.assertRaises(ValueError): profile.validate_result(altered,manifest)

    def test_actual_worker_control_flow_runs_three_complete_warmups_before_twenty_captured_steps(self):
        tree=ast.parse(Path(profile.__file__).read_text())
        worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="worker")
        step=next(n for n in worker.body if isinstance(n,ast.FunctionDef) and n.name=="run_step")
        # Execute the real window-control AST with fake callbacks, not GPU math.
        offset=worker.body.index(step)
        windows=worker.body[offset+1:offset+3]
        self.assertIsInstance(windows[0],ast.For)
        self.assertIsInstance(windows[1],ast.With)
        active=False; calls=[]
        class Capture:
            def __enter__(self):
                nonlocal active
                active=True
            def __exit__(self,*args):
                nonlocal active
                active=False
        fake=SimpleNamespace(profiler=SimpleNamespace(profile=lambda **kwargs:Capture(),ProfilerActivity=SimpleNamespace(CPU="cpu",CUDA="cuda")))
        def run_step(index):
            calls.append((index,active))
            return {"updates":index+1}
        namespace={"torch":fake,"run_step":run_step}
        exec(compile(ast.Module(body=windows,type_ignores=[]),"actual_profile_windows","exec"),namespace)
        self.assertEqual(calls,[(i,i>=3) for i in range(23)])
        self.assertEqual(namespace["row"]["updates"],23)
        self.assertFalse(active)
        # Both windows call one body retaining all five proofs and the original timer.
        stages=[n.args[2].value for n in ast.walk(step) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="stage"]
        self.assertEqual(stages,list(profile.STAGES))
        self.assertEqual(sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="synchronize" for n in ast.walk(step)),2)
    def test_worker_invokes_original_core_step_without_replacing_learning_primitives(self):
        tree=ast.parse(Path(profile.__file__).read_text())
        worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="worker")
        self.assertIsInstance(worker.body[0],ast.Assign)
        self.assertEqual(worker.body[0].value.func.id,"validate_worker")
        calls=[n for n in ast.walk(worker) if isinstance(n,ast.Call)]
        steps=[n for n in calls if isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=="c" and n.func.attr=="core_step"]
        self.assertEqual(len(steps),1)
        self.assertEqual(ast.literal_eval(steps[0].args[-2]),"shadow")
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and
            (n.func.attr in ("export_chrome_trace","add_metadata_json","evaluate") or
             n.func.attr == "train" and isinstance(n.func.value,ast.Name) and n.func.value.id in ("core","c")) for n in ast.walk(worker)))

    def parent_fixture(self, root, used=0):
        policy={"concurrent_local_gpu_jobs":1,"local_gpu_seconds_per_utc_day":21600,"local_job_timeout_seconds":10800,
                "stops_at":(datetime.now(timezone.utc)+timedelta(days=1)).isoformat()}
        (root/"research").mkdir()
        profile.c.write(root/"research/campaign.json",policy)
        return {"now":lambda:datetime.now(timezone.utc),"used_seconds":lambda *a:used}

    def test_active_lease_and_insufficient_budget_refuse_before_new_artifacts_or_child(self):
        from research import process_tree_guard as guard
        for existing,used in ((True,0),(False,21500)):
            with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
                root=Path(temp); account=self.parent_fixture(root,used)
                lease=root/"results/research/local_lease.json"; lease.parent.mkdir(parents=True)
                if existing: lease.write_bytes(b'{"status":"unknown","cleanup_requires_reconciliation":true}')
                raw=lease.read_bytes() if existing else None
                stack.enter_context(patch.object(profile,"ROOT",root))
                stack.enter_context(patch.object(profile,"source_snapshot",return_value={}))
                stack.enter_context(patch.object(profile,"binding_for",return_value=({},profile.c.DATA_IDENTITY)))
                stack.enter_context(patch.object(profile.c,"accounting",return_value=account))
                stack.enter_context(patch.object(profile.subprocess,"check_output",return_value="a"*40))
                child=stack.enter_context(patch.object(guard,"run_guarded"))
                with self.assertRaises(ValueError): profile.parent("fresh",root/"data")
                child.assert_not_called(); self.assertFalse((lease.parent/"fresh").exists())
                if existing: self.assertEqual(lease.read_bytes(),raw)
                else: self.assertFalse(lease.exists())

    def test_unknown_child_cleanup_keeps_matching_lease_and_receipt(self):
        from research import process_tree_guard as guard
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root=Path(temp); account=self.parent_fixture(root)
            stack.enter_context(patch.object(profile,"ROOT",root))
            stack.enter_context(patch.object(profile,"source_snapshot",return_value={}))
            stack.enter_context(patch.object(profile,"binding_for",return_value=({},profile.c.DATA_IDENTITY)))
            stack.enter_context(patch.object(profile.c,"accounting",return_value=account))
            stack.enter_context(patch.object(profile.subprocess,"check_output",return_value="a"*40))
            def fail(command,log,timeout,path,receipt,**kwargs):
                self.assertGreater(timeout,0); self.assertLess(timeout,180)
                receipt.update(child_pid=999,process_tree_cleanup_verified=False)
                raise RuntimeError("Synthetic unknown cleanup")
            stack.enter_context(patch.object(guard,"run_guarded",side_effect=fail))
            with self.assertRaises(RuntimeError): profile.parent("fresh",root/"data")
            lease=profile.c.read(root/"results/research/local_lease.json")
            receipt=profile.c.read(root/"results/research/fresh/supervisor.json")
            self.assertTrue(lease["cleanup_requires_reconciliation"])
            self.assertEqual(receipt["status"],"cleanup_unverified")
            self.assertIn("finished_at",receipt)

    def test_exact_os_nonce_script_argv_source_lease_and_membership_before_torch(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); folder=root/"results/research/fresh"; directory=folder/("process_tree_"+"a"*32)
            directory.mkdir(parents=True); guard=root/"research/process_tree_guard.py"; guard.parent.mkdir()
            shutil.copyfile(profile.ROOT/"research/process_tree_guard.py",guard)
            manifest={"schema":profile.INPUT_SCHEMA,"recipe":profile.RECIPE,"seed":2026,"run":"fresh","data_dir":str(root/"data"),
                "source_git_commit":"c"*40,"effective_config":profile.c.configuration(2026,recipe=profile.RECIPE),
                "source_sha256":{"own":"a"*64},"config_sha256":profile.c.WRN300_CONFIGS[2026][1],
                "data_identity":profile.c.DATA_IDENTITY,"input_sha256":{"fixture":"b"*64}}
            manifest["effective_config_sha256"]=profile.c.json_sha(manifest["effective_config"])
            profile.c.write(folder/"profile_inputs.json",manifest); digest=profile.c.sha256(folder/"profile_inputs.json")
            now=datetime.now(timezone.utc)
            receipt=dict(schema=profile.SCHEMA,status="profiling",backend="local",mode="profile",run="fresh",reserved_seconds=180,
                started_at=now.isoformat(),deadline_unix=now.timestamp()+180,pid=11,process_created_at=1.5,child_pid=12,child_created_at=2.5,
                input_manifest_sha256=digest,source_sha256=manifest["source_sha256"],input_sha256=manifest["input_sha256"],confirmation_evaluated=False,formal_admission_evidence=False)
            lease=dict(pid=11,process_created_at=1.5)
            spec=dict(schema="owned_python_process_tree_v1",nonce="a"*32,host_pid=11,host_created_at=1.5,cwd=str(root),
                script=str(Path(profile.__file__).resolve()),script_sha256=profile.c.sha256(profile.__file__),
                guard_source_sha256=profile.c.PINS["research/process_tree_guard.py"],arguments=["--_worker",str(folder),"--_manifest-sha",digest],
                deadline_unix=now.timestamp()+170,job_name="Local\\SaberAugMix_"+"a"*32 if os.name=="nt" else None)
            ready=dict(schema=spec["schema"],nonce=spec["nonce"],pid=12,created_at=2.5,script_sha256=spec["script_sha256"],method="windows_job_v1" if os.name=="nt" else "posix_session_v1")
            argv=[sys.executable,str(guard),"--bootstrap",str(directory/"spec.json")]
            process=SimpleNamespace(pid=12,create_time=lambda:2.5,cmdline=lambda:argv)
            parent=SimpleNamespace(pid=11,create_time=lambda:1.5)
            def save():
                for path,value in ((folder/"supervisor.json",receipt),(root/"results/research/local_lease.json",lease),(directory/"spec.json",spec),(directory/"ready.json",ready)): profile.c.write(path,value)
            with ExitStack() as stack:
                stack.enter_context(patch.object(profile,"ROOT",root)); stack.enter_context(patch.object(profile.os,"getppid",return_value=11))
                stack.enter_context(patch.object(profile.psutil,"Process",side_effect=lambda pid=None:process if pid is None else parent))
                stack.enter_context(patch.object(profile,"source_snapshot",return_value=manifest["source_sha256"]))
                stack.enter_context(patch.object(profile,"binding_for",return_value=(manifest["input_sha256"],manifest["data_identity"])))
                boundary=stack.enter_context(patch.object(profile.p,"verify_os_boundary"))
                save(); self.assertEqual(profile.validate_worker(folder,digest),manifest); boundary.assert_called_once()
                for target,key,value in ((lease,"cleanup_requires_reconciliation",True),(lease,"pid",99),(spec,"nonce","b"*32),
                    (spec,"script_sha256","b"*64),(spec,"deadline_unix",now.timestamp()+200),(ready,"pid",99),(receipt,"child_created_at",99)):
                    old=target.get(key); target[key]=value; save()
                    with self.assertRaises(ValueError): profile.validate_worker(folder,digest)
                    if old is None: target.pop(key)
                    else: target[key]=old
                save(); argv[2]="--not-bootstrap"
                with self.assertRaises(ValueError): profile.validate_worker(folder,digest)
                argv[2]="--bootstrap"; boundary.side_effect=ValueError("outside own Job/session")
                with self.assertRaises(ValueError): profile.validate_worker(folder,digest)


if __name__ == "__main__":
    unittest.main()
