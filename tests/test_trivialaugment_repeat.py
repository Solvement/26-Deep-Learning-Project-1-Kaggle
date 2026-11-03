"""Seed27 lineage/resource contracts; no Torch or official-image execution."""
import ast
from contextlib import ExitStack, redirect_stdout
import copy
from datetime import datetime, timedelta, timezone
import io
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research import run_trivialaugment_repeat as repeat


class RepeatContracts(unittest.TestCase):
    def fixture(self):
        from tests import test_trivialaugment_primary as original
        report, manifest = original.PureContracts().result_fixture()
        config = repeat.configuration()
        manifest.update(config_sha256=repeat.CONFIGS[2027][1], effective_config_sha256=repeat.c.json_sha(config),
                        frozen_manifest_sha256="f"*64)
        report.update(schema=repeat.RESULT_SCHEMA,recipe=repeat.RECIPE,seed=2027,ta_rng_seed=2027,
                      config_sha256=manifest["config_sha256"],effective_config_sha256=manifest["effective_config_sha256"],
                      frozen_manifest_sha256=manifest["frozen_manifest_sha256"])
        return report,manifest

    def test_fixed_seed27_configuration_and_complete_source_closure_without_torch(self):
        before=set(sys.modules)
        config=repeat.configuration()
        self.assertEqual(config,{**repeat.primary.configuration(2026),"seed":2027})
        self.assertEqual(config["split_seed"],2026)
        self.assertEqual(repeat.c.sha256(repeat.ROOT/repeat.CONFIGS[2027][0]),repeat.CONFIGS[2027][1])
        with self.assertRaises(ValueError):repeat.configuration(2026)
        source=repeat.source_snapshot()
        self.assertEqual(source["research/run_trivialaugment_repeat.py"],repeat.c.sha256(repeat.__file__))
        for key,value in repeat.primary.source_snapshot().items():self.assertEqual(source[key],value)
        self.assertIn(repeat.PAIR_ENTRY,source);self.assertIn(repeat.PAIR_PLAN,source)
        self.assertFalse({"torch","runtime","numpy"}&(set(sys.modules)-before))

    def test_no_action_or_unguarded_worker_and_no_implicit_formal(self):
        before=set(sys.modules)
        with patch.object(repeat,"parent") as parent,patch.object(repeat,"worker") as worker,redirect_stdout(io.StringIO()):
            repeat.main([]);parent.assert_not_called();worker.assert_not_called()
            for args in (["--run","fresh","--mode","preflight"],["--frozen","missing"],["--_manifest-sha","a"*64]):
                with self.assertRaises(ValueError):repeat.main(args)
        with tempfile.TemporaryDirectory() as temp,self.assertRaises(ValueError):repeat.worker(Path(temp),"a"*64)
        self.assertFalse({"torch","runtime"}&(set(sys.modules)-before))

    def test_guard_keeps_full180_and_refuses_today_before_new_day(self):
        now=datetime(2026,11,2,23,tzinfo=timezone.utc)
        policy=dict(concurrent_local_gpu_jobs=1,local_gpu_seconds_per_utc_day=21600,local_job_timeout_seconds=10800,
                    stops_at="2026-11-05T00:00:00-05:00")
        self.assertEqual(repeat.guard_seconds(.1185),1.5*.1185*313*160+300)
        self.assertEqual(repeat.admission_budget(policy,now,14133.,"preflight"),180)
        with self.assertRaises(ValueError):repeat.admission_budget(policy,now,14133.,"formal",{"median_step_seconds":.1185})
        self.assertEqual(repeat.admission_budget(policy,now+timedelta(hours=1),0,"formal",{"median_step_seconds":.1185}),10800)
        for mode,used,report in (("preflight",21500,None),("formal",0,None),("formal",0,{"median_step_seconds":.2})):
            with self.assertRaises(ValueError):repeat.admission_budget(policy,now,used,mode,report)
        for value in (True,0,-1,float("nan"),float("inf")):
            with self.assertRaises(ValueError):repeat.guard_seconds(value)

    def frozen_fixture(self,root,sources):
        target=root/"results/research/ta_seed27_formal"
        document=dict(schema=repeat.FREEZE_SCHEMA,status="frozen",policy=repeat.POLICY,
            frozen_at=(datetime.now(timezone.utc)-timedelta(minutes=1)).isoformat(),
            checkpoint_path=str(root/"results/research"/repeat.PRIMARY_RUN/"best.pt"),checkpoint_sha256=repeat.PRIMARY_CHECKPOINT_SHA,
            config=repeat.primary.configuration(2026),planned_repeat_seed=2027,
            planned_repeat_config_path=str((root/repeat.CONFIGS[2027][0]).resolve()),planned_repeat_config_sha256=repeat.CONFIGS[2027][1],
            planned_repeat_effective_config=repeat.configuration(),planned_repeat_effective_config_sha256=repeat.c.json_sha(repeat.configuration()),
            planned_repeat_source_sha256=sources,repeat_run_path=str(target),evidence_sha256={})
        path=root/"results/research/frozen_primary.json";path.parent.mkdir(parents=True)
        repeat.c.write(path,document)
        return path,document

    def test_freeze_exact_source_config_primary_path_seed_and_prior_time(self):
        from research import trivialaugment_paired_v2 as paired
        sources={"own.py":"a"*64};config=repeat.configuration()
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            root=Path(temp);path,value=self.frozen_fixture(root,sources)
            stack.enter_context(patch.object(repeat,"ROOT",root))
            stack.enter_context(patch.object(repeat,"configuration",return_value=config))
            validator=stack.enter_context(patch.object(paired,"validate_frozen",side_effect=lambda p:repeat.c.read(p)))
            self.assertEqual(repeat.validate_freeze(path,sources),value);self.assertEqual(validator.call_count,1)
            changes=(("planned_repeat_source_sha256",{}),("planned_repeat_seed",2026),("planned_repeat_config_sha256","b"*64),
                     ("planned_repeat_effective_config",{**config,"epochs":300}),("checkpoint_sha256","c"*64),
                     ("policy","flip"),("repeat_run_path",str(root/"outside")),
                     ("frozen_at",(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()))
            for key,bad in changes:
                repeat.c.write(path,{**value,key:bad})
                with self.subTest(key=key),self.assertRaises(ValueError):repeat.validate_freeze(path,sources)
            repeat.c.write(path,value)
            Path(value["repeat_run_path"]).mkdir()
            # The worker must still revalidate its frozen primary after its own directory exists.
            self.assertEqual(repeat.validate_freeze(path,sources),value)

    def test_seed27_report_rejects_old_primary_partial_state_and_wrong_freeze(self):
        report,manifest=self.fixture();repeat.validate_result(report,manifest)
        for key,value in (("schema",repeat.primary.RESULT_SCHEMA),("seed",2026),("ta_rng_seed",2026),
                          ("frozen_manifest_sha256","0"*64),("successful_updates",22),("state_tensor_count",479),
                          ("formal_training_admitted",True),("raw_rng_state",{})):
            with self.subTest(key=key),self.assertRaises(ValueError):repeat.validate_result({**report,key:value},manifest)
        changed=copy.deepcopy(report);changed["steps"]=changed["steps"][:-1]
        with self.assertRaises(ValueError):repeat.validate_result(changed,manifest)

    def test_learning_proof_wrapper_is_the_actual_frozen_function_without_monkeypatch(self):
        tree=ast.parse(Path(repeat.__file__).read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="execute_step")
        call=function.body[-1].value
        self.assertEqual(ast.unparse(call.func),"primary.execute_step")
        self.assertEqual(len([n for n in ast.walk(function) if isinstance(n,ast.Call)]),1)
        for name in ("preflight_worker","formal_worker"):
            worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.unparse(worker.body[0].value.func),"validate_worker") if name=="formal_worker" else None
        formal=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="formal_worker")
        text=ast.unparse(formal)
        self.assertIn("CONFIGS[2027][0]",text);self.assertNotIn("incumbent_compare.run",text)
        self.assertIn("benchmark.main()",text);self.assertIn("resume=False",text)
        self.assertLess(text.index("runtime.seed_all(2027)"),text.index("runtime.environment()"))

    def owner_fixture(self,root):
        folder=root/"results/research/pref27";directory=folder/("process_tree_"+"a"*32);directory.mkdir(parents=True)
        guard=root/"research/process_tree_guard.py";guard.parent.mkdir();shutil.copyfile(repeat.ROOT/"research/process_tree_guard.py",guard)
        frozen=root/"results/research/frozen.json";repeat.c.write(frozen,{})
        now=datetime.now(timezone.utc);digest_freeze=repeat.c.sha256(frozen)
        config=repeat.configuration();sources={"own":"a"*64};binding={"fixture":"b"*64}
        manifest=dict(schema=repeat.INPUT_SCHEMA,mode="preflight",role="engineering_preflight",preflight_path=None,recipe=repeat.RECIPE,
            seed=2027,run=folder.name,data_dir=str(root/"data"),source_git_commit="c"*40,effective_config=config,
            effective_config_sha256=repeat.c.json_sha(config),source_sha256=sources,config_sha256=repeat.CONFIGS[2027][1],
            data_identity=repeat.c.DATA_IDENTITY,input_sha256=binding,frozen_manifest_path=str(frozen),frozen_manifest_sha256=digest_freeze)
        repeat.c.write(folder/"inputs.json",manifest);digest=repeat.c.sha256(folder/"inputs.json")
        receipt=dict(schema=repeat.SCHEMA,status="benchmarking",backend="local",mode="preflight",role="engineering_preflight",run=folder.name,
            reserved_seconds=180,started_at=now.isoformat(),deadline_unix=now.timestamp()+180,pid=11,process_created_at=1.5,child_pid=12,child_created_at=2.5,
            input_manifest_sha256=digest,source_sha256=sources,input_sha256=binding,confirmation_evaluated=False,formal_training_admitted=False,
            frozen_manifest_path=str(frozen),frozen_manifest_sha256=digest_freeze)
        lease=dict(pid=11,process_created_at=1.5)
        spec=dict(schema="owned_python_process_tree_v1",nonce="a"*32,host_pid=11,host_created_at=1.5,cwd=str(root),
            script=str(Path(repeat.__file__).resolve()),script_sha256=repeat.c.sha256(repeat.__file__),guard_source_sha256=repeat.c.PINS["research/process_tree_guard.py"],
            arguments=["--_worker",str(folder),"--_manifest-sha",digest],deadline_unix=now.timestamp()+170,
            job_name="Local\\SaberAugMix_"+"a"*32 if os_name()=="nt" else None)
        ready=dict(schema=spec["schema"],nonce=spec["nonce"],pid=12,created_at=2.5,script_sha256=spec["script_sha256"],method="windows_job_v1" if os_name()=="nt" else "posix_session_v1")
        process=SimpleNamespace(pid=12,create_time=lambda:2.5,cmdline=lambda:[sys.executable,str(guard),"--bootstrap",str(directory/"spec.json")])
        parent=SimpleNamespace(pid=11,create_time=lambda:1.5)
        frozen_value=dict(frozen_at=(now-timedelta(minutes=1)).isoformat(),repeat_run_path=str(root/"results/research/formal27"))
        def save():
            for path,value in ((folder/"supervisor.json",receipt),(root/"results/research/local_lease.json",lease),(directory/"spec.json",spec),(directory/"ready.json",ready)):repeat.c.write(path,value)
        save()
        return folder,digest,manifest,receipt,lease,spec,ready,process,parent,frozen_value,save

    def test_own_os_nonce_argv_lease_and_frozen_chronology_before_torch(self):
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            root=Path(temp);folder,digest,manifest,receipt,lease,spec,ready,process,parent,frozen,save=self.owner_fixture(root)
            stack.enter_context(patch.object(repeat,"ROOT",root));stack.enter_context(patch.object(repeat.os,"getppid",return_value=11))
            stack.enter_context(patch.object(repeat.psutil,"Process",side_effect=lambda pid=None:process if pid is None else parent))
            stack.enter_context(patch.object(repeat,"configuration",return_value=manifest["effective_config"]))
            stack.enter_context(patch.object(repeat,"source_snapshot",return_value=manifest["source_sha256"]))
            stack.enter_context(patch.object(repeat,"binding_for",return_value=(manifest["input_sha256"],manifest["data_identity"])))
            stack.enter_context(patch.object(repeat,"validate_freeze",return_value=frozen))
            boundary=stack.enter_context(patch.object(repeat.p,"verify_os_boundary"))
            self.assertEqual(repeat.validate_worker(folder,digest),manifest);boundary.assert_called_once()
            for target,key,value in ((lease,"cleanup_requires_reconciliation",True),(spec,"nonce","b"*32),
                                     (spec,"arguments",["--_worker","old"]),(receipt,"frozen_manifest_sha256","d"*64),
                                     (frozen,"frozen_at",(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat())):
                old=target.get(key);target[key]=value;save()
                with self.subTest(key=key),self.assertRaises(ValueError):repeat.validate_worker(folder,digest)
                if old is None:target.pop(key)
                else:target[key]=old
            save();boundary.side_effect=ValueError("not in actual own Job")
            with self.assertRaises(ValueError):repeat.validate_worker(folder,digest)

    def parent_patches(self,stack,root,used=0):
        policy=dict(concurrent_local_gpu_jobs=1,local_gpu_seconds_per_utc_day=21600,local_job_timeout_seconds=10800,stops_at="2026-11-05T00:00:00-05:00")
        (root/"research").mkdir();repeat.c.write(root/"research/campaign.json",policy)
        frozen=root/"frozen.json";repeat.c.write(frozen,{})
        current=datetime(2026,11,3,tzinfo=timezone.utc)
        document=dict(repeat_run_path=str(root/"results/research/formal27"),frozen_at=(current-timedelta(hours=1)).isoformat())
        for name,value in (("ROOT",root),("configuration",lambda: {}),("source_snapshot",lambda: {}),
                           ("validate_freeze",lambda *a:document),("binding_for",lambda *a:({},repeat.c.DATA_IDENTITY))):
            stack.enter_context(patch.object(repeat,name,value))
        stack.enter_context(patch.object(repeat.subprocess,"check_output",return_value="a"*40))
        stack.enter_context(patch.object(repeat.time,"time",return_value=current.timestamp()))
        stack.enter_context(patch.object(repeat.c,"accounting",return_value={"now":lambda:current,"used_seconds":lambda *a,**k:used}))
        return frozen

    def test_active_lease_or_insufficient_budget_never_launch_or_reclaim(self):
        from research import process_tree_guard as guard
        for active,used in ((True,0),(False,21500)):
            with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
                root=Path(temp);frozen=self.parent_patches(stack,root,used)
                lease=root/"results/research/local_lease.json";lease.parent.mkdir(parents=True)
                if active:lease.write_bytes(b'{"status":"unknown"}')
                launch=stack.enter_context(patch.object(guard,"run_guarded"))
                with self.assertRaises(ValueError):repeat.parent("pref27",root/"data","preflight",frozen)
                launch.assert_not_called();self.assertFalse((lease.parent/"pref27").exists())
                if active:self.assertEqual(lease.read_bytes(),b'{"status":"unknown"}')

    def test_unknown_cleanup_preserves_original_repeat_lease_and_failed_receipt(self):
        from research import process_tree_guard as guard
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            root=Path(temp);frozen=self.parent_patches(stack,root)
            def fail(command,log,seconds,path,receipt,**kwargs):
                receipt.update(child_pid=999,process_tree_cleanup_verified=False)
                raise guard.TreeCleanupError("synthetic unknown cleanup")
            stack.enter_context(patch.object(guard,"run_guarded",side_effect=fail))
            with self.assertRaises(guard.TreeCleanupError):repeat.parent("pref27",root/"data","preflight",frozen)
            owner=repeat.c.read(root/"results/research/local_lease.json")
            self.assertTrue(owner["cleanup_requires_reconciliation"])
            self.assertEqual(repeat.c.read(root/"results/research/pref27/supervisor.json")["status"],"cleanup_unverified")

    def guard_fixture(self,root,folder,digest,receipt):
        directory=folder/("process_tree_"+"a"*32);directory.mkdir()
        spec=dict(schema="owned_python_process_tree_v1",nonce="a"*32,host_pid=receipt["pid"],host_created_at=receipt["process_created_at"],
            cwd=str(root),script=str(Path(repeat.__file__).resolve()),script_sha256=repeat.c.sha256(repeat.__file__),
            guard_source_sha256=repeat.c.PINS["research/process_tree_guard.py"],arguments=["--_worker",str(folder),"--_manifest-sha",digest],
            deadline_unix=receipt["deadline_unix"]-2,job_name="Local\\SaberAugMix_"+"a"*32)
        ready=dict(schema=spec["schema"],nonce=spec["nonce"],pid=receipt["child_pid"],created_at=receipt["child_created_at"],
                   script_sha256=spec["script_sha256"],method="windows_job_v1")
        completion={**ready,"exit_code":0}
        proof=dict(schema=spec["schema"],nonce=spec["nonce"],method="windows_job_v1",guard_directory=str(directory),
            guard_source_sha256=spec["guard_source_sha256"],cleanup_verified=True,timed_out=False,
            child_pid=receipt["child_pid"],child_created_at=receipt["child_created_at"],child_returncode=0,
            completion_authenticated=True,completion_exit_code=0)
        receipt["process_tree_guard_results"]=[proof]
        def save():
            for name,value in (("spec.json",spec),("ready.json",ready),("completion.json",completion),("host_result.json",proof)):
                repeat.c.write(directory/name,value)
        save()
        return proof,spec,ready,completion,save

    def test_closed_native_records_reject_forged_cleanup_or_wrong_identity(self):
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            root=Path(temp);folder=root/"results/research/pref27";folder.mkdir(parents=True)
            now=datetime.now(timezone.utc)-timedelta(seconds=20)
            receipt=dict(pid=11,process_created_at=1.5,child_pid=12,child_created_at=2.5,
                started_at=now.isoformat(),deadline_unix=now.timestamp()+180)
            proof,spec,ready,completion,save=self.guard_fixture(root,folder,"b"*64,receipt)
            stack.enter_context(patch.object(repeat,"ROOT",root))
            self.assertEqual(len(repeat.completed_guard_paths(folder,receipt,"b"*64)),4)
            for target,key,value in ((proof,"cleanup_verified",False),(proof,"completion_authenticated",False),
                (proof,"timed_out",True),(proof,"child_returncode",1),(proof,"completion_exit_code",1),
                (spec,"arguments",["--_worker","other"]),(spec,"script_sha256","d"*64),
                (ready,"nonce","b"*32),(completion,"exit_code",1),(completion,"created_at",99.)):
                old=target[key];target[key]=value;save()
                with self.subTest(key=key),self.assertRaises(ValueError):repeat.completed_guard_paths(folder,receipt,"b"*64)
                target[key]=old
            save();receipt["process_tree_guard_results"]=[]
            with self.assertRaises(ValueError):repeat.completed_guard_paths(folder,receipt,"b"*64)

    def test_full_seed27_preflight_bundle_binds_freeze_guard_files_timing_and_inactive_owners(self):
        report,base_manifest=self.fixture();config=repeat.configuration()
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            root=Path(temp);folder=root/"results/research/pref27";folder.mkdir(parents=True)
            frozen_path=root/"results/research/frozen.json";repeat.c.write(frozen_path,{})
            freeze_sha=repeat.c.sha256(frozen_path);report["frozen_manifest_sha256"]=freeze_sha
            data=root/"fixture.bin";data.write_bytes(b"synthetic proof fixture only")
            binding={str(data):repeat.c.sha256(data)};sources=base_manifest["source_sha256"]
            manifest={**base_manifest,"schema":repeat.INPUT_SCHEMA,"mode":"preflight","role":"engineering_preflight", "recipe":repeat.RECIPE,
                "seed":2027,"run":"pref27","preflight_path":None,"data_dir":str(root/"data"),"input_sha256":binding,
                "effective_config":config,"frozen_manifest_path":str(frozen_path),"frozen_manifest_sha256":freeze_sha}
            manifest.pop("manifest_sha256");inputs=folder/"inputs.json";repeat.c.write(inputs,manifest)
            digest=repeat.c.sha256(inputs);report["input_manifest_sha256"]=digest
            output=folder/"trivialaugment_repeat_preflight.json";repeat.c.write(output,report)
            now=datetime.now(timezone.utc)-timedelta(seconds=20)
            receipt=dict(schema=repeat.SCHEMA,mode="preflight",role="engineering_preflight",status="complete",run="pref27",seed=2027,
                precision="bf16",config_sha256=repeat.CONFIGS[2027][1],source_sha256=sources,input_manifest_sha256=digest,input_sha256=binding,
                reserved_seconds=180,artifact=output.name,artifact_sha256=repeat.c.sha256(output),formal_training_admitted=False,
                process_tree_cleanup_verified=True,started_at=now.isoformat(),finished_at=(now+timedelta(seconds=10)).isoformat(),elapsed_seconds=10.,
                deadline_unix=now.timestamp()+180,pid=11,process_created_at=1.5,child_pid=12,child_created_at=2.5,
                frozen_manifest_path=str(frozen_path),frozen_manifest_sha256=freeze_sha)
            proof,spec,ready,completion,save_guard=self.guard_fixture(root,folder,digest,receipt)
            frozen=dict(frozen_at=(now-timedelta(minutes=1)).isoformat(),repeat_run_path=str(root/"results/research/formal27"))
            stack.enter_context(patch.object(repeat,"ROOT",root));stack.enter_context(patch.object(repeat,"configuration",return_value=config))
            stack.enter_context(patch.object(repeat,"validate_freeze",return_value=frozen))
            stack.enter_context(patch.object(repeat,"binding_for",return_value=(binding,repeat.c.DATA_IDENTITY)))
            process=stack.enter_context(patch.object(repeat.psutil,"Process",side_effect=repeat.psutil.NoSuchProcess(11)))
            def save():repeat.c.write(folder/"supervisor.json",receipt)
            save();value,paths=repeat.validate_preflight_bundle(output,sources,frozen_path)
            self.assertEqual(value,report);self.assertEqual(len(paths),7)
            self.assertIn(folder/("process_tree_"+"a"*32)/"completion.json",paths)
            for key,bad in (("seed",2026),("elapsed_seconds",.1),("frozen_manifest_sha256","0"*64),
                            ("process_tree_cleanup_verified",False),("process_tree_guard_results",[])):
                old=receipt[key];receipt[key]=bad;save()
                with self.subTest(key=key),self.assertRaises(ValueError):repeat.validate_preflight_bundle(output,sources,frozen_path)
                receipt[key]=old
            save();process.side_effect=None;process.return_value=SimpleNamespace(create_time=lambda:1.5)
            with self.assertRaises(ValueError):repeat.validate_preflight_bundle(output,sources,frozen_path)
            process.return_value=SimpleNamespace(create_time=lambda:999.)
            self.assertEqual(repeat.validate_preflight_bundle(output,sources,frozen_path)[0],report)

    def test_seed27_typed_comparison_without_primary_only_run_and_fixed_weighted_gates(self):
        """Synthetic scalar fixture, not an executed model or training result."""
        from tests import test_incumbent_compare as original
        reference=original.IncumbentComparisonTests().fixture(incumbent=True)
        config=repeat.configuration();document=copy.deepcopy(reference)
        document.update(config=config,parameter_count=4166298)
        learning=("research/train_trivialaugment.py","research/trivialaugment.py","research/train_shadow.py","research/augmentations.py")
        source={name:repeat.PINS.get(name,repeat.c.PINS.get(name)) for name in learning}
        frozen=dict(data_signature=document["data_signature"],evaluation_context=document["evaluation_context"],planned_repeat_source_sha256=source)
        marker=dict(schema="research_trivialaugment_training_v1",precision="bf16",grad_scaler_enabled=False,training_seed=2027,split_seed=2026,
            ta_rng_algorithm="PCG64",ta_rng_seed=2027,input_config_sha256=repeat.CONFIGS[2027][1],effective_config_sha256=repeat.c.json_sha(config),source_sha256=source)
        document["research_training_integrity"]=dict(schema="research_trivialaugment_integrity_v1",training=marker,
            formal_last_state_tensor_optimizer_parent_and_TA_RNG_audited=True,successful_updates=50080,skipped_updates=0,
            forward_passes=50080,source_examples_seen=6400000,model_views_seen=6400000,
            ta_operation_counts={name:6400000 if name=="identity" else 0 for name in repeat.OPERATIONS})
        with patch.object(repeat.c,"read",return_value=reference),patch.object(repeat.c,"sha256",return_value=repeat.INCUMBENT_SHA),patch.object(repeat,"configuration",return_value=config):
            result=repeat.compare_repeat_benchmark(document,frozen)
            self.assertEqual(result["seed"],2027);self.assertEqual(result["role"],"development_repeat")
            for key,bad in (("training_seed",2026),("precision","fp16"),("input_config_sha256","a"*64)):
                changed=copy.deepcopy(document);changed["research_training_integrity"]["training"][key]=bad
                with self.subTest(key=key),self.assertRaises(ValueError):repeat.compare_repeat_benchmark(changed,frozen)
            changed=copy.deepcopy(document);changed["policies"][repeat.POLICY]["cases"]["noise_004"]["examples"]=4999
            with self.assertRaises(ValueError):repeat.compare_repeat_benchmark(changed,frozen)


def os_name():
    import os
    return os.name


if __name__=="__main__":unittest.main()
