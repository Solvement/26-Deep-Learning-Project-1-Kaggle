"""Synthetic private bundles and mocked CLI transport; no WSL/API/GPU/data access."""
import ast
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
import zipfile

import numpy as np
from research import run_colab_augmix as runner
from research.train_augmix import fixed_config
ACTUAL_VALIDATE_DATA = runner.validate_data


class FakeCLI:
    """Never runs a process: models one known/unknown transport conversation."""
    def __init__(self, folder, manifest, pool, complete=False, lag=False, wrong_guard=False):
        self.folder,self.manifest,self.pool=folder,manifest,pool
        self.complete,self.lag,self.wrong_guard=complete,lag,wrong_guard
        self.calls=[];self.assigned=False;self.stopped=False;self.parts={}

    def usage(self):
        active=int(self.assigned and (not self.stopped or self.lag))
        return f"Current balance: 0.0 compute units\nUsage rate: {1.15 if active else 0.0}/hr\nActive assignments: {active}"

    def require(self,args,timeout=60):
        self.calls.append(args)
        if args[0]=="usage":return self.usage()
        if args[0]=="new":
            assert json.loads((self.folder/"supervisor.json").read_text())["assignment_attempts"]==1
            self.assigned=True;return "[colab] Session READY."
        receipt=json.loads((self.folder/"supervisor.json").read_text())
        if args[0]=="exec" and Path(args[args.index("--file")+1]).name=="remote_guard.py":
            identity=receipt["guard_identity"].copy()
            if self.wrong_guard:identity["input_bundle_sha256"]="f"*64
            hardware={"status":"armed","deadline_unix":receipt["deadline_unix"],"device":"Tesla T4","amp_dtype":"float16","torch_version":"2.11.0+cu130"}
            return "AUGMIX_GUARD_JSON="+json.dumps(identity)+"\nSABER_GUARD_JSON="+json.dumps(hardware)
        if args[0]=="upload":
            assert receipt["remote_guard_verified"] is True
            return "uploaded"
        if args[0]=="exec":
            remote={"schema":runner.REMOTE_SCHEMA,"trainer":"augmix_jsd","status":"complete" if self.complete else "failed",
                "run":receipt["run"],"input_bundle_sha256":self.manifest["bundle_sha256"],"input_manifest_sha256":receipt["input_manifest_sha256"],
                "host_source_sha256":self.manifest["host_source_sha256"],"source_files_sha256":self.manifest["files_sha256"],
                "guard_identity":receipt["guard_identity"],"benchmark_passed":self.complete,"confirmation_evaluated":False,"hidden_images_loaded":False}
            members=list(runner.COMPLETE_OUTPUTS) if self.complete else ["remote_receipt.json","preflight_failure.json"]
            data=io.BytesIO()
            with zipfile.ZipFile(data,"w") as archive:
                for name in members:
                    value=json.dumps(remote).encode() if name=="remote_receipt.json" else json.dumps(self.pool).encode() if name in runner.POOL_OUTPUTS else b"synthetic trainer output"
                    archive.writestr(name,value)
            block=data.getvalue();name="augmix_artifacts_000.part";self.parts[name]=block
            import hashlib
            remote.update(artifact_members=members,artifact_sha256=hashlib.sha256(block).hexdigest(),
                artifact_chunks=[{"name":name,"bytes":len(block),"sha256":hashlib.sha256(block).hexdigest()}])
            return "AUGMIX_RESULT_JSON="+json.dumps(remote)
        if args[0]=="download":
            Path(args[-1]).write_bytes(self.parts[Path(args[-2]).name]);return "downloaded"
        raise AssertionError("Unexpected mock transport action")

    def run(self,args,timeout=60):
        self.calls.append(args)
        if args[0]=="stop":self.stopped=True;return 0,"[colab] Session terminated."
        if args[0]=="usage":return 0,self.usage()
        raise AssertionError("Unexpected mock cleanup action")


class AugMixColabTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name in set(runner.SOURCES+runner.HOST_SOURCES):
            path = self.root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("synthetic own source "+name).encode())
        for name in runner.CACHE:
            path = self.root/"data"/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("synthetic cache bytes "+name).encode())
        for name in ("best.pt", "metrics.json", "history.json"):
            path = self.root/"results/regularized"/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("synthetic trusted baseline "+name).encode())
        self.config = fixed_config()
        self.config_path = self.root/"config.json"
        self.config_path.write_text(json.dumps(self.config))
        self.folder = self.root/"results/research/colab_augmix_test"
        self.folder.mkdir(parents=True)
        policy = {"stops_at": "2099-01-01T00:00:00+00:00", "colab_additional_purchases_authorized": False}
        (self.root/"research/campaign.json").write_text(json.dumps(policy))
        predecessor=self.root/"results/research"/runner.PREDECESSOR_RUN/"supervisor.json"
        predecessor.parent.mkdir(parents=True,exist_ok=True)
        predecessor.write_text(json.dumps({"run":"wrn34_3_300_fp16_s2027_v3","finished_at":"2026-01-01T00:00:00+00:00",
            "runtime_stop_verified":True,"allocation_returned_success":True,"allocation_identity_verified":True,
            "stop_reported_termination":True,"stop_exit_code":0,"usage_after":{"active_assignments":0,"rate_compute_units_per_hour":0.},
            "pid":9999999,"process_created_at":1.}))
        for item in (patch.object(runner, "ROOT", self.root),
            patch.object(runner, "CORE", {name: runner.sha256(self.root/name) for name in runner.CORE}),
            patch.object(runner, "BASELINE_SHA", runner.sha256(self.root/"results/regularized/best.pt")),
            patch.object(runner, "FROZEN_COMMON_SHA", runner.sha256(self.root/"research/run_colab.py")),
            patch.object(runner, "validate_data", return_value={})):
            item.start(); self.addCleanup(item.stop)
        self.manifest = runner.prepare(self.root, self.config_path, self.folder, "augmix_test")

    def report(self, median=.05):
        result = {"schema": "research_augmix_preflight_v1", "status": "complete", "synthetic_only": False,
            "official_training_only": True, "precision": "fp16", "grad_scaler_enabled": True, "parameters": 4298970,
            "initial_grad_scaler_scale":256.,
            "model_config": {"depth":22, "widen_factor":4, "dropout":0., "num_classes":10}, "config": self.config,
            "input_config_sha256": self.manifest["raw_config_sha256"], "batch_size":128, "model_batch_size":384,
            "warmup_steps":3, "timed_steps":20, "attempted_updates":23, "successful_updates":23, "skipped_updates":0,
            "parameter_changed_steps":23, "forward_calls":23, "single_forward_per_step":True,
            "num_workers":2, "prefetch_factor":2, "multiprocessing_context":"spawn", "worker_cuda_initialized":False,
            "persistent_workers":False, "all_workers_stopped":True, "source_examples_seen":2944, "model_views_seen":8832,
            "training_examples":40000, "validation_examples":5000, "validation_evaluated":False, "confirmation_evaluated":False,
            "hidden_images_loaded":False, "external_data_loaded":False, "source_ids_are_training_only":True,
            "source_ids_unique_count":2944, "preflight_source_ids_sha256":"a"*64, "data_signature":runner.SIGNATURE,
            "source_sha256":{name:self.manifest["files_sha256"][name] for name in (*runner.CORE,*runner.METHOD_SOURCES)},
            "overflow_smoke":{"status":"passed", "finite_loss":True, "infinite_gradient_injected":True, "updated":False,
                "all_parameters_unchanged":True, "optimizer_state_unchanged":True, "ema_parameters_unchanged":True,
                "scale_before":256., "scale_after":128., "source_examples":128, "model_views":384, "forward_calls":1},
            "environment":{"gpu":"Tesla T4", "device":"cuda", "amp_dtype":"torch.float16"},
            "median_step_seconds":median, "p90_step_seconds":median*1.1, "images_per_second":128/median,
            "peak_gpu_memory_bytes":1024, "required_training_and_benchmark_seconds":1.5*median*313*160+300}
        return result

    def pool(self):
        return {"schema":"research_augmix_worker_pool_v1","status":"closed","owner_pid":100,"owner_created_at":1.,
            "workers":[{"pid":101,"created_at":2.},{"pid":102,"created_at":3.}],"num_workers":2,"prefetch_factor":2,
            "context":"spawn","worker_cuda_initialized":False,"all_workers_stopped":True,
            "source_ids_sha256":"a"*64,"sample_seeds_sha256":"b"*64}

    def test_offline_package_has_exact_explicit_members_and_binds_controller_raw_data_and_chunks(self):
        expected=set(runner.entries(self.root,self.config_path))|{runner.PACKAGE_MANIFEST}
        with zipfile.ZipFile(self.folder/"input.private.zip") as archive:
            self.assertEqual(set(archive.namelist()),expected)
        self.assertEqual(self.manifest["schema"],runner.SCHEMA)
        self.assertEqual(self.manifest["trainer"],"augmix_jsd")
        self.assertEqual(self.manifest["raw_config_sha256"],runner.sha256(self.config_path))
        self.assertIn("research/run_colab_augmix.py",self.manifest["host_source_sha256"])
        self.assertNotIn("research/benchmark_shadow.py",expected)
        self.assertEqual(runner.validate_package(self.root,self.folder),self.manifest)

    def test_unknown_method_config_and_source_or_chunk_mutation_fail_closed(self):
        for change in ({"seed":2027},{"batch_size":64},{"augmix_jsd_weight":0.},{"train_all":True},{"cutmix_alpha":1.}):
            with self.assertRaises(ValueError):runner.validate_config({**self.config,**change})
        part=self.folder/self.manifest["chunks"][0]["name"]
        original=part.read_bytes();part.write_bytes(original+b"changed")
        with self.assertRaisesRegex(ValueError,"chunk changed"):runner.validate_package(self.root,self.folder)
        part.write_bytes(original)
        (self.root/"research/augmix.py").write_text("changed source")
        with self.assertRaisesRegex(ValueError,"fingerprints differ"):runner.validate_package(self.root,self.folder)

    def test_npy_headers_enforce_shape_dtype_numeric_layout_and_actual_file_length(self):
        data=io.BytesIO();np.save(data,np.zeros((2,3),dtype=np.uint8),allow_pickle=False);raw=data.getvalue()
        runner.npy_header(io.BytesIO(raw),(2,3),"uint8",len(raw))
        for shape,dtype,length in (((3,2),"uint8",len(raw)),((2,3),"int64",len(raw)),((2,3),"uint8",len(raw)+1)):
            with self.assertRaises(ValueError):runner.npy_header(io.BytesIO(raw),shape,dtype,length)

    def test_forged_official_manifest_is_rejected_before_any_array_header_or_payload(self):
        with patch.object(runner,"npy_header") as header:
            with self.assertRaisesRegex(ValueError,"official data manifest"):
                ACTUAL_VALIDATE_DATA(self.root)
        header.assert_not_called()

    def test_fixed_width_official_string_ids_are_accepted_without_allowing_pickle_or_other_dtypes(self):
        # The SHA-bound cache stores official IDs as fixed-width Unicode U5;
        # these are synthetic representatives of its header contract.
        stream = io.BytesIO()
        np.save(stream, np.asarray(["1", "50000"], dtype="<U5"), allow_pickle=False)
        raw = stream.getvalue()
        runner.npy_header(io.BytesIO(raw), (2,), "<U5", len(raw))
        for dtype in ("int64", "<U4", "<U6", "object"):
            with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                runner.npy_header(io.BytesIO(raw), (2,), dtype, len(raw))
        objects = io.BytesIO()
        np.save(objects, np.asarray(["1", "50000"], dtype=object), allow_pickle=True)
        with self.assertRaises(ValueError):
            runner.npy_header(io.BytesIO(objects.getvalue()), (2,), "object", len(objects.getvalue()))

    def test_extra_hidden_input_is_rejected_even_if_bundle_and_chunk_hashes_are_recomputed(self):
        bundle=self.folder/"input.private.zip"
        with zipfile.ZipFile(bundle,"a") as archive:archive.writestr("data/official_test.npz",b"not permitted")
        import hashlib
        raw=bundle.read_bytes();part=self.folder/self.manifest["chunks"][0]["name"];part.write_bytes(raw)
        document={**self.manifest,"bundle_sha256":runner.sha256(bundle),"chunks":[{"name":part.name,"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}]}
        (self.folder/"input_manifest.json").write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError,"exact member allowlist"):
            runner.validate_package(self.root,self.folder)

    def test_zip_rejects_extra_traversal_windows_paths_duplicates_and_symlinks(self):
        for name in ("../x","/x","C:x","a\\x","x/"):
            data=io.BytesIO()
            with zipfile.ZipFile(data,"w") as archive:archive.writestr(name,b"x")
            with zipfile.ZipFile(data) as archive:
                with self.assertRaises(ValueError):runner.safe_zip(archive,[name],100)
        data=io.BytesIO()
        with zipfile.ZipFile(data,"w") as archive:
            link=zipfile.ZipInfo("x");link.external_attr=(0o120777<<16);archive.writestr(link,b"elsewhere")
        with zipfile.ZipFile(data) as archive:
            with self.assertRaises(ValueError):runner.safe_zip(archive,["x"],100)

    def test_real_pipeline_admission_uses_fixed_formula_and_separate_overflow_counts(self):
        proof=self.report()
        result=runner.validate_preflight_report(proof,self.manifest,10800)
        self.assertEqual(result["runtime_safety_multiplier"],1.5)
        for field,value in (("successful_updates",22),("model_batch_size",128),("num_workers",0),("synthetic_only",True)):
            with self.assertRaises(ValueError):runner.validate_preflight_report({**proof,field:value},self.manifest,10800)
        bad=self.report();bad["overflow_smoke"]["updated"]=True
        with self.assertRaises(ValueError):runner.validate_preflight_report(bad,self.manifest,10800)
        with self.assertRaises(RuntimeError):runner.validate_preflight_report(self.report(.3),self.manifest,10800)

    def test_generated_program_compiles_orders_guard_real_preflight_baseline_scratch_training_and_benchmark(self):
        program=runner.remote_program(self.manifest,"safe_session",time.time()+10800,"a"*64)
        compile(program,"remote_augmix.py","exec")
        self.assertLess(program.index("globals().get('_augmix_guard_identity')"),program.index("bundle=R/"))
        self.assertLess(program.index("stage('preflight'"),program.index("stage('baseline'"))
        self.assertLess(program.index("validate_preflight_report(report,M,D-time.time())"),program.index("stage('training'"))
        self.assertLess(program.index("stage('training'"),program.index("stage('benchmark'"))
        self.assertNotIn("pip install",program)
        self.assertNotIn("--resume",program)
        self.assertIn("metrics.get('attempted_updates')==50080",program)
        self.assertIn("metrics.get('model_views_seen')==19200000",program)
        guard=runner.remote_guard(self.manifest,"safe_session",12345.)
        compile(guard,"guard.py","exec")
        self.assertIn("colab_runtime.unassign()",guard)

    def test_actual_generated_budget_failure_never_reaches_baseline_or_formal_stage(self):
        program=runner.remote_program(self.manifest,"safe_session",time.time()+10800,"a"*64)
        tree=ast.parse(program);node=next(n for n in tree.body if isinstance(n,ast.Try))
        start=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="report" for t in n.targets))
        preflight=self.folder/"preflight";preflight.mkdir()
        pool=preflight/"worker_pool_preflight.json";pool.write_text(json.dumps(self.pool()))
        proof=self.report(.3);proof["worker_pool_receipt_sha256"]=runner.sha256(pool)
        (preflight/"engineering_benchmark.json").write_text(json.dumps(proof))
        stage=Mock()
        namespace={"R":self.folder,"M":self.manifest,"D":time.time()+10800,"json":json,"time":time,
            "hashlib":__import__("hashlib"),"CORE":runner.CORE,"METHOD_SOURCES":runner.METHOD_SOURCES,
            "require":runner.require,"validate_preflight":Mock(),"validate_preflight_report":runner.validate_preflight_report,
            "validate_pool_record":runner.validate_pool_record,"receipt":{},"stage":stage}
        module=ast.Module(body=node.body[start:],type_ignores=[]);ast.fix_missing_locations(module)
        with self.assertRaises(RuntimeError):exec(compile(module,"actual_generated_budget.py","exec"),namespace)
        stage.assert_not_called()

    def test_active_shared_cloud_lease_refuses_before_any_cli_or_allocation_intent(self):
        lease=self.folder.parent/"cloud_lease.json";lease.write_text('{"session":"other"}')
        with patch.object(runner,"ColabCLI") as cli:
            with self.assertRaises(FileExistsError):runner.launch(self.folder)
        cli.assert_not_called()
        self.assertFalse((self.folder/"supervisor.json").exists())
        self.assertEqual(json.loads(lease.read_text())["session"],"other")

    def test_predecessor_uncertain_cleanup_refuses_even_with_empty_shared_lease_before_api(self):
        path=self.root/"results/research"/runner.PREDECESSOR_RUN/"supervisor.json"
        value=json.loads(path.read_text());value["runtime_stop_verified"]=False;path.write_text(json.dumps(value))
        with patch.object(runner,"ColabCLI") as cli:
            with self.assertRaisesRegex(ValueError,"known completed cleanup"):
                runner.launch(self.folder)
        cli.assert_not_called()
        self.assertFalse((self.folder/"supervisor.json").exists())

    def test_unknown_allocation_keeps_lease_and_never_sends_stop_or_retries(self):
        fake=Mock()
        def require(args,timeout):
            if args[0]=="usage":return "Current balance: 0.0 compute units\nUsage rate: 0.0/hr\nActive assignments: 0"
            if args[0]=="new":raise RuntimeError("unknown allocation outcome")
            raise AssertionError("No upload or exec may follow unknown allocation")
        fake.require.side_effect=require
        process=Mock(pid=os.getpid());process.poll.return_value=0
        with patch.object(runner,"ColabCLI",return_value=fake),patch.object(runner,"private_directory",side_effect=lambda p:p.mkdir(exist_ok=True)), \
            patch.object(runner.subprocess,"Popen",return_value=process):
            with self.assertRaisesRegex(RuntimeError,"unknown allocation"):runner.launch(self.folder)
            with self.assertRaises(FileExistsError):runner.launch(self.folder)
        fake.run.assert_not_called()
        self.assertEqual(sum(c.args[0][0]=="new" for c in fake.require.call_args_list),1)
        receipt=json.loads((self.folder/"supervisor.json").read_text())
        self.assertFalse(receipt["runtime_stop_verified"])
        self.assertTrue((self.folder.parent/"cloud_lease.json").exists())

    def test_known_cleanup_is_claimed_once_and_lagging_usage_retains_uncertainty(self):
        fake=Mock()
        receipt={"allocation_returned_success":True,"session":"own"}
        response={"runtime_stop_verified":True,"usage_after":{"active_assignments":1,"rate_compute_units_per_hour":1.15}}
        with patch.object(runner,"stop_and_verify",return_value=response) as stop:
            first=runner.cleanup_once(fake,self.folder,receipt)
            second=runner.cleanup_once(fake,self.folder,receipt)
        self.assertFalse(first["runtime_stop_verified"])
        self.assertFalse(second["runtime_stop_verified"])
        stop.assert_called_once_with(fake,"own",allocation_known=True)

    def test_transport_deadline_refuses_work_after_reserve_is_exhausted(self):
        with patch.object(runner.time,"time",return_value=1000.):
            self.assertEqual(runner.bounded_timeout(1500.,90,180),90)
            with self.assertRaises(TimeoutError):runner.bounded_timeout(1179.,90,180)

    def run_transport(self,fake):
        process=Mock(pid=os.getpid());process.poll.return_value=0
        with patch.object(runner,"ColabCLI",return_value=fake),patch.object(runner,"private_directory",side_effect=lambda p:p.mkdir(exist_ok=True)), \
            patch.object(runner,"wsl_path",side_effect=str),patch.object(runner.subprocess,"Popen",return_value=process):
            return runner.launch(self.folder)

    def test_known_negative_preflight_download_is_preserved_one_assign_one_stop_then_own_lease_release(self):
        fake=FakeCLI(self.folder,self.manifest,self.pool())
        result=self.run_transport(fake)
        self.assertEqual(result["status"],"remote_failed")
        self.assertTrue(result["runtime_stop_verified"])
        self.assertFalse(result["benchmark_passed"])
        self.assertEqual(sum(c[0]=="new" for c in fake.calls),1)
        self.assertEqual(sum(c[0]=="stop" for c in fake.calls),1)
        self.assertFalse((self.folder.parent/"cloud_lease.json").exists())
        self.assertTrue((self.folder/"preflight_failure.json").exists())
        self.assertEqual(json.loads((self.folder/"remote_receipt.json").read_text())["trainer"],"augmix_jsd")

    def test_known_complete_has_exact_worker_outputs_and_retains_lease_when_usage_lags(self):
        fake=FakeCLI(self.folder,self.manifest,self.pool(),complete=True,lag=True)
        with patch.object(runner,"validate_downloaded",return_value=True) as deep_audit:
            result=self.run_transport(fake)
        deep_audit.assert_called_once()
        self.assertEqual(result["status"],"cleanup_unverified")
        self.assertTrue(result["benchmark_passed"])
        self.assertFalse(result["runtime_stop_verified"])
        self.assertTrue((self.folder.parent/"cloud_lease.json").exists())
        self.assertEqual(sum(c[0]=="stop" for c in fake.calls),1)
        self.assertEqual(len(runner.POOL_OUTPUTS),161)
        self.assertTrue(all((self.folder/name).is_file() for name in runner.POOL_OUTPUTS))

    def test_guard_identity_mismatch_stops_known_runtime_before_upload_or_work(self):
        fake=FakeCLI(self.folder,self.manifest,self.pool(),wrong_guard=True)
        with self.assertRaisesRegex(ValueError,"guard identity"):
            self.run_transport(fake)
        self.assertFalse(any(c[0] in ("upload","download") for c in fake.calls))
        self.assertEqual(sum(c[0]=="exec" for c in fake.calls),1)
        self.assertEqual(sum(c[0]=="stop" for c in fake.calls),1)

    def test_unknown_upload_outcome_never_repeats_upload_or_allocation(self):
        fake=FakeCLI(self.folder,self.manifest,self.pool())
        original=fake.require
        def unknown(args,timeout):
            if args[0]=="upload":
                fake.calls.append(args)
                raise RuntimeError("unknown upload outcome")
            return original(args,timeout)
        fake.require=unknown
        with self.assertRaisesRegex(RuntimeError,"unknown upload"):
            self.run_transport(fake)
        self.assertEqual(sum(c[0]=="new" for c in fake.calls),1)
        self.assertEqual(sum(c[0]=="upload" for c in fake.calls),1)
        self.assertEqual(sum(c[0]=="stop" for c in fake.calls),1)
        self.assertFalse(any(c[0]=="download" for c in fake.calls))
        self.assertEqual(json.loads((self.folder/"supervisor.json").read_text())["status"],"failed")

    def test_worker_pool_pid_create_or_unclosed_workers_cannot_be_complete_evidence(self):
        runner.validate_pool_record(self.pool())
        for change in ({"all_workers_stopped":False},{"worker_cuda_initialized":True},{"owner_created_at":float("nan")},
                       {"workers":[{"pid":101,"created_at":2.},{"pid":101,"created_at":3.}]}):
            with self.assertRaises(ValueError):runner.validate_pool_record({**self.pool(),**change})

    def test_downloaded_core_comparison_preserves_typed_fixed_policy_field(self):
        import torch
        from research import benchmark,benchmark_augmix
        candidate=self.folder/"candidate";candidate.mkdir()
        preflight=self.folder/"preflight";preflight.mkdir()
        baseline_folder=self.folder/"baseline";baseline_folder.mkdir()
        pool=preflight/"worker_pool_preflight.json";pool.write_text(json.dumps(self.pool()))
        proof=self.report();proof["worker_pool_receipt_sha256"]=runner.sha256(pool)
        proof["environment"]["packages"]={"torch":"2.11.0+cu130"}
        (preflight/"engineering_benchmark.json").write_text(json.dumps(proof))
        for name in ("metrics.json","history.json","provenance.json","environment.json"):
            (candidate/name).write_text("{}")
        (candidate/"config.json").write_text(json.dumps(self.config))
        (candidate/"input_config.json").write_bytes(self.config_path.read_bytes())
        (candidate/"best.pt").write_bytes(b"CPU synthetic checkpoint")
        (candidate/"last.pt").write_bytes(b"CPU synthetic last")
        def report(clean,stress):
            return {"clean_accuracy":clean,"clean_loss":.2,"stress_accuracy":stress,
                "cases":{case:{"examples":5000,"accuracy":clean if case=="clean" else stress,"loss":.2,
                    "input_pixel_sha256":"c"*64} for case in benchmark.CASES}}
        context={"device_type":"cuda","amp_dtype":"torch.float16","torch_version":"2.11.0+cu130","training_core_sha256":runner.CORE}
        baseline={"schema":benchmark.SCHEMA,"status":"complete","examples":5000,"training_examples":40000,
            "data_signature":runner.SIGNATURE,"evaluation_context":context,"batch_size":128,"bn_recalibrated":False,
            "checkpoint_sha256":benchmark.BASELINE_CHECKPOINT_SHA,"stress_cases":list(benchmark.CASES),
            "benchmark_source_sha256":self.manifest["files_sha256"]["research/benchmark.py"],"policies":{"flip":report(.96,.78)}}
        baseline_path=baseline_folder/"benchmark.json";baseline_path.write_text(json.dumps(baseline))
        policies={p:report(.97,.8) for p in benchmark.POLICIES}
        core={"schema":benchmark.SCHEMA,"status":"complete","examples":5000,"training_examples":40000,
            "parameter_count":4298970,"checkpoint_sha256":runner.sha256(candidate/"best.pt"),
            "data_signature":runner.SIGNATURE,"config":self.config,"evaluation_context":context,
            "training_metrics_sha256":runner.sha256(candidate/"metrics.json"),"training_history_sha256":runner.sha256(candidate/"history.json"),
            "policies":policies,"confirmation_evaluated":False,"hidden_images_loaded":False,"external_data_loaded":False,
            "baseline_sha256":runner.sha256(baseline_path),"promotion":{p:benchmark.promotion_gate(r,baseline["policies"]["flip"]) for p,r in policies.items()}}
        core_path=candidate/"benchmark.core.json";core_path.write_text(json.dumps(core))
        integrity={"schema":"research_augmix_integrity_v1"}
        document={**core,"fixed_primary_policy":"center80_shift20","research_training_integrity":{**integrity,
            "validator_source_sha256":self.manifest["files_sha256"]["research/benchmark_augmix.py"],
            "core_benchmark_sha256":runner.sha256(core_path),"provenance_document_sha256":runner.sha256(candidate/"provenance.json"),
            "environment_document_sha256":runner.sha256(candidate/"environment.json")}}
        evidence_path=candidate/"benchmark.json";evidence_path.write_text(json.dumps(document))
        with patch.object(torch,"load",side_effect=[{"config":self.config},{}]), \
             patch.object(benchmark_augmix,"validate_augmix",return_value=integrity):
            self.assertTrue(runner.validate_downloaded(self.folder,self.manifest,{"torch_version":"2.11.0+cu130"}))
        document["fixed_primary_policy"]="plain";evidence_path.write_text(json.dumps(document))
        with patch.object(torch,"load",side_effect=[{"config":self.config},{}]), \
             patch.object(benchmark_augmix,"validate_augmix",return_value=integrity):
            with self.assertRaisesRegex(ValueError,"typed AugMix benchmark integrity"):
                runner.validate_downloaded(self.folder,self.manifest,{"torch_version":"2.11.0+cu130"})


if __name__=="__main__":unittest.main()
