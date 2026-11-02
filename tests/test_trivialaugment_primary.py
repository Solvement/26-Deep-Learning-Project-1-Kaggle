"""Pure controller contracts and Linux-CI-only Torch numerical equivalence."""
import ast
from contextlib import ExitStack,redirect_stdout
import copy
from datetime import datetime,timedelta,timezone
import io
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research import run_trivialaugment_primary as preflight


class PureContracts(unittest.TestCase):
    def test_fixed_config_registry_source_and_planned_repeat_are_torch_free(self):
        before=set(sys.modules);primary=preflight.configuration();repeat=preflight.configuration(2027)
        self.assertEqual(repeat,{**primary,"seed":2027})
        base=preflight.c.read(preflight.ROOT/"research/configs/wrn34_3.json")
        raw=preflight.c.read(preflight.ROOT/preflight.CONFIGS[2026][0])
        self.assertEqual(raw,{**base,"ta_space":"fixed_standard","ta_strengths":31})
        self.assertEqual((primary["depth"],primary["widen_factor"],primary["epochs"],primary["batch_size"]),(34,3,160,128))
        tree=ast.parse((preflight.ROOT/"research/trivialaugment.py").read_text())
        names=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="OPERATIONS" for t in n.targets))
        self.assertEqual(preflight.OPERATIONS,names);self.assertEqual(len(names),14)
        source=preflight.source_snapshot();self.assertEqual(source[preflight.PLAN],preflight.PLAN_SHA)
        self.assertFalse({"torch","runtime"}&(set(sys.modules)-before))

    def test_original_fixed_160_guard_full180_day_cap_and_hard_stop(self):
        now=datetime(2026,11,2,tzinfo=timezone.utc)
        policy=dict(concurrent_local_gpu_jobs=1,local_gpu_seconds_per_utc_day=21600,local_job_timeout_seconds=10800,
                    stops_at="2026-11-05T00:00:00-05:00")
        self.assertEqual(preflight.guard_seconds(.08),1.5*.08*313*160+300)
        self.assertEqual(preflight.admission_budget(policy,now,0,"preflight"),180)
        self.assertEqual(preflight.admission_budget(policy,now,0,"formal",{"median_step_seconds":.08}),10800)
        for mode,used,report in (("preflight",21500,None),("formal",0,None),("formal",0,{"median_step_seconds":.2}),
                                 ("formal",18000,{"median_step_seconds":.08})):
            with self.assertRaises(ValueError):preflight.admission_budget(policy,now,used,mode,report)
        with self.assertRaises(ValueError):preflight.admission_budget({**policy,"stops_at":"2026-11-06T00:00:00-05:00"},now,0,"preflight")

    def test_own_early_identity_full_gather_pil_timing_and_original_method_calls(self):
        tree=ast.parse(Path(preflight.__file__).read_text())
        for name in ("preflight_worker","formal_worker"):
            function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
            authenticate=next(n.lineno for n in ast.walk(function) if isinstance(n,ast.Call) and ast.unparse(n.func)=="validate_worker")
            self.assertLess(authenticate,min(n.lineno for n in ast.walk(function) if isinstance(n,(ast.Import,ast.ImportFrom))))
        step=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="execute_step")
        calls=[n for n in ast.walk(step) if isinstance(n,ast.Call)]
        self.assertEqual(sum(ast.unparse(n.func)=="r.train_step" for n in calls),1)
        self.assertIn("train_pixels[ids.cpu().numpy()]",ast.unparse(step))
        worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="preflight_worker")
        loop=next(n for n in worker.body if isinstance(n,ast.For));body=[ast.unparse(n) for n in loop.body]
        self.assertEqual(ast.unparse(loop.iter),"range(23)")
        self.assertEqual(body[0],"torch.cuda.synchronize()");self.assertIn("execute_step(",body[2])
        self.assertEqual(body[3],"torch.cuda.synchronize()")
        self.assertFalse(any(isinstance(n,ast.Attribute) and n.attr=="profiler" for n in ast.walk(tree)))
        formal=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="formal_worker")
        calls=[n for n in ast.walk(formal) if isinstance(n,ast.Call)]
        train=next(n for n in calls if ast.unparse(n.func)=="trainer.train")
        keywords={k.arg:ast.unparse(k.value) for k in train.keywords}
        self.assertEqual(keywords["precision"],"'bf16'");self.assertEqual(keywords["resume"],"False")
        self.assertIn("read_bytes()",keywords["input_config_bytes"])
        self.assertLess(next(n.lineno for n in calls if ast.unparse(n.func)=="runtime.seed_all"),
                        next(n.lineno for n in calls if ast.unparse(n.func)=="runtime.environment"))

    def test_registry_complete_counts_reject_bool_extra_missing_negative_and_wrong_total(self):
        good=dict.fromkeys(preflight.OPERATIONS,0);good[preflight.OPERATIONS[0]]=128
        preflight.validate_operation_counts(good,128)
        for value in ({**good,"unknown":0},{k:v for k,v in good.items() if k!=preflight.OPERATIONS[-1]},
                      {**good,preflight.OPERATIONS[0]:True},{**good,preflight.OPERATIONS[0]:-1},
                      {**good,preflight.OPERATIONS[0]:127}):
            with self.assertRaises(ValueError):preflight.validate_operation_counts(value,128)

    def test_complete_typed_ta_report_rejects_partial_counts_foreign_rng_and_persisted_state(self):
        report,manifest=self.result_fixture();preflight.validate_result(report,manifest)
        for key,value in (("state_tensor_count",479),("model_views_seen",2943),("views_per_source",3),
                          ("ta_rng_algorithm","MT19937"),("ta_rng_seed",2027),("preflight_state_discarded",False),
                          ("formal_training_admitted",True),("successful_updates",22),("ta_rng_end_sha256","0"*64)):
            with self.subTest(key=key),self.assertRaises(ValueError):preflight.validate_result({**report,key:value},manifest)
        for key,value in (("ta_rng_after_sha256","0"*64),("state_tensor_count",479),
                          ("ta_operation_counts",dict.fromkeys(preflight.OPERATIONS,0))):
            changed=copy.deepcopy(report);changed["steps"][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):preflight.validate_result(changed,manifest)
        changed=copy.deepcopy(report);changed["ta_operation_counts"][preflight.OPERATIONS[0]]+=1
        with self.assertRaises(ValueError):preflight.validate_result(changed,manifest)
        changed=copy.deepcopy(report);changed["steps"].pop()
        with self.assertRaises(ValueError):preflight.validate_result(changed,manifest)


    def result_fixture(self):
        config = preflight.configuration()
        manifest = {"source_sha256":{"own.py":"a"*64},"config_sha256":preflight.CONFIGS[2026][1],
            "effective_config_sha256":preflight.c.json_sha(config),"manifest_sha256":"b"*64,
            "source_git_commit":"c"*40,"data_identity":preflight.c.DATA_IDENTITY}
        rows = [dict(loss=1.,gradient_norm=.2,lr=.1*(i+1)/(5*313),mixed=False,updates=i+1,
            logits_dtype="torch.bfloat16",parameter_probe_max_delta=.01,ema_probe_max_delta=.001,
            all_parameters_finite=True,bn_single_forward=True,bn_counters_before=[i]*31,
            bn_counters_after=[i+1]*31,state_tensor_count=480) for i in range(23)]
        totals=dict.fromkeys(preflight.OPERATIONS,0)
        for i,row in enumerate(rows):
            counts=dict.fromkeys(preflight.OPERATIONS,0);counts[preflight.OPERATIONS[i%14]]=128
            row.update(ta_operation_counts=counts,ta_rng_before_sha256=f"{i:064x}",ta_rng_after_sha256=f"{i+1:064x}")
            for name,value in counts.items():totals[name]+=value
        durations = [.08+i*.00001 for i in range(23)]
        import statistics
        median = statistics.median(durations[3:])
        env = dict(python="fixture-python",python_version="3.14.3",platform="Synthetic fixture",device="cuda",
            gpu="Synthetic fixture",amp_dtype="torch.bfloat16",cuda_runtime="13.0",cudnn=92000,gpu_memory_GiB=7.96,
            cudnn_benchmark=True,deterministic_algorithms=False,process_local_cudnn_path_entries_removed=0,
            packages={key: "2.12.0+cu130" if key=="torch" else "1.0" for key in
                      ("torch","torchvision","numpy","Pillow","matplotlib","nbformat","reportlab")})
        report = dict(schema=preflight.RESULT_SCHEMA,status="complete",engineering_only=True,formal_training_admitted=False,
            accuracy_evaluated=False,confirmation_evaluated=False,hidden_images_loaded=False,external_data_loaded=False,
            recipe=preflight.RECIPE,seed=2026,training_precision="bf16",native_bf16_without_emulation=True,
            grad_scaler_enabled=False,disabled_scaler_scale=1.,master_parameter_dtype="torch.float32",ema_parameter_dtype="torch.float32",
            parameter_count=4166298,trainable_parameter_count=4166298,bn_modules=31,epochs=160,planned_formal_attempted_updates=50080,
            training_pool_examples=40000,evaluation_examples=0,batch_size=128,warmup_updates=3,timed_updates=20,
            successful_updates=23,skipped_updates=0,source_examples_seen=2944,unique_source_examples=2944,
            bn_forward_counts=[23]*31,ema_buffer_copies=0,state_tensor_count=480,state_float32_count=418,state_int64_count=62,
            preflight_state_discarded=True,ta_rng_algorithm="PCG64",ta_rng_seed=2026,model_views_seen=2944,views_per_source=1,
            ta_operation_counts=totals,ta_rng_start_sha256=f"{0:064x}",ta_rng_end_sha256=f"{23:064x}",source_sha256=manifest["source_sha256"],
            config_sha256=manifest["config_sha256"],effective_config_sha256=manifest["effective_config_sha256"],
            input_manifest_sha256=manifest["manifest_sha256"],source_git_commit=manifest["source_git_commit"],data_identity=manifest["data_identity"],
            timing_scope=preflight.TIMING_SCOPE,steps=rows,step_seconds=durations,median_step_seconds=median,
            full_step_seconds=sum(durations),setup_seconds=1.,conservative_pipeline_seconds=preflight.guard_seconds(median),
            peak_gpu_memory_bytes=100,peak_gpu_reserved_bytes=120,environment=env,environment_sha256=preflight.c.json_sha(env),
            inventory_sha256="d"*64,ordered_training_indices_sha256="e"*64,ordered_official_ids_sha256="f"*64)
        return report,manifest

    def test_no_arguments_no_action_and_unguarded_worker_before_torch(self):
        before=set(sys.modules)
        with patch.object(preflight,"parent") as parent,patch.object(preflight,"worker") as worker,redirect_stdout(io.StringIO()):
            preflight.main([])
        parent.assert_not_called();worker.assert_not_called()
        with tempfile.TemporaryDirectory() as folder,self.assertRaises(ValueError):preflight.worker(Path(folder),"a"*64)
        self.assertFalse({"torch","runtime"}&(set(sys.modules)-before))
        for args in (["--run","fresh"],["--mode","formal"],["--_manifest-sha","a"*64]):
            with self.assertRaises(ValueError):preflight.main(args)

    def comparison(self,clean=-10,noise=-50,blur=175):
        counts={"clean":clean,"noise_004":noise,"boxblur3":blur,"contrast065":0,"brightness080":0,"resize24":0}
        base={name:.8 for name in counts};base.update(clean=.9654,noise_004=.7768)
        cases={name:dict(candidate_accuracy=(round(base[name]*5000)+delta)/5000,
                         incumbent_accuracy=base[name],accuracy_delta=delta/5000) for name,delta in counts.items()}
        return dict(cases=cases,same_policy_deltas={"clean_accuracy":clean/5000,"mean_stress_accuracy":sum(v for k,v in counts.items() if k!="clean")/25000})

    def test_incumbent_inclusive_threshold_counts_and_each_case_floor_are_exact(self):
        benchmark={"promotion":{"center80_shift20":{"passed":True}}}
        # Actual incumbent numbers cause binary delta to fall slightly below inclusive bounds.
        comparison=self.comparison();self.assertLess(.9634-.9654,-.002)
        self.assertLess(.7668-.7768,-.01)
        self.assertTrue(preflight.scalar_screen(benchmark,comparison)["scalar_screen_passed"])
        for changed in (self.comparison(clean=-11),self.comparison(noise=-51),self.comparison(blur=174)):
            result=preflight.scalar_screen(benchmark,changed)
            self.assertFalse(result["scalar_screen_passed"])
            self.assertEqual(result["next_action"],"stop_no_pair_repeat_or_submit")
        comparison["cases"]["clean"]["candidate_accuracy"]+=.00000001
        with self.assertRaises(ValueError):preflight.scalar_screen(benchmark,comparison)
        bad=self.comparison();bad["cases"]["clean"]["candidate_accuracy"]=True
        with self.assertRaises(ValueError):preflight.scalar_screen(benchmark,bad)
        self.assertFalse(preflight.scalar_screen({"promotion":{"center80_shift20":{"passed":False}}},self.comparison())["scalar_screen_passed"])

    def test_timing_booleans_nonfinite_and_missing_native_metadata_are_rejected(self):
        report,manifest=self.result_fixture()
        for key in ("median_step_seconds","full_step_seconds","conservative_pipeline_seconds","setup_seconds"):
            for value in (True,False,float("nan"),float("inf"),0.,-1.):
                with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                    preflight.validate_result({**report,key:value},manifest)
        changed=copy.deepcopy(report);changed["step_seconds"]=[1.]*23
        changed.update(median_step_seconds=True,full_step_seconds=23.,conservative_pipeline_seconds=preflight.guard_seconds(1.))
        with self.assertRaises(ValueError):preflight.validate_result(changed,manifest)
        for key in report["environment"]:
            changed=copy.deepcopy(report);changed["environment"].pop(key)
            changed["environment_sha256"]=preflight.c.json_sha(changed["environment"])
            with self.subTest(missing=key),self.assertRaises(ValueError):preflight.validate_result(changed,manifest)
        for key,value in (("packages",{}),("cuda_runtime",""),("cudnn",True),("gpu",[]),("gpu_memory_GiB",True),
                          ("amp_dtype","torch.float16"),("cudnn_benchmark",1),("deterministic_algorithms",True),
                          ("packages",{**report["environment"]["packages"],"torch":""})):
            changed=copy.deepcopy(report);changed["environment"][key]=value
            changed["environment_sha256"]=preflight.c.json_sha(changed["environment"])
            with self.subTest(key=key),self.assertRaises(ValueError):preflight.validate_result(changed,manifest)

    def parent_fixture(self, root, used=0):
        policy={"concurrent_local_gpu_jobs":1,"local_gpu_seconds_per_utc_day":21600,"local_job_timeout_seconds":10800,
                "stops_at":"2026-11-05T00:00:00-05:00"}
        (root/"research").mkdir()
        preflight.c.write(root/"research/campaign.json",policy)
        return {"now":lambda:datetime(2026,11,2,tzinfo=timezone.utc),"used_seconds":lambda *a:used}

    def test_active_lease_and_insufficient_budget_refuse_before_new_artifacts_or_child(self):
        from research import process_tree_guard as guard
        for existing,used in ((True,0),(False,21500)):
            with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
                root=Path(temp); account=self.parent_fixture(root,used)
                lease=root/"results/research/local_lease.json"; lease.parent.mkdir(parents=True)
                if existing: lease.write_bytes(b'{"status":"unknown","cleanup_requires_reconciliation":true}')
                raw=lease.read_bytes() if existing else None
                stack.enter_context(patch.object(preflight,"configuration",return_value={}))
                stack.enter_context(patch.object(preflight,"ROOT",root))
                stack.enter_context(patch.object(preflight,"source_snapshot",return_value={}))
                stack.enter_context(patch.object(preflight,"binding_for",return_value=({},preflight.c.DATA_IDENTITY)))
                stack.enter_context(patch.object(preflight.c,"accounting",return_value=account))
                stack.enter_context(patch.object(preflight.subprocess,"check_output",return_value="a"*40))
                child=stack.enter_context(patch.object(guard,"run_guarded"))
                with self.assertRaises(ValueError): preflight.parent("fresh",root/"data","preflight")
                child.assert_not_called(); self.assertFalse((lease.parent/"fresh").exists())
                if existing: self.assertEqual(lease.read_bytes(),raw)
                else: self.assertFalse(lease.exists())

    def test_unknown_child_cleanup_keeps_matching_lease_and_receipt(self):
        from research import process_tree_guard as guard
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root=Path(temp); account=self.parent_fixture(root)
            stack.enter_context(patch.object(preflight.time,"time",return_value=account["now"]().timestamp()))
            stack.enter_context(patch.object(preflight,"configuration",return_value={}))
            stack.enter_context(patch.object(preflight,"ROOT",root))
            stack.enter_context(patch.object(preflight,"source_snapshot",return_value={}))
            stack.enter_context(patch.object(preflight,"binding_for",return_value=({},preflight.c.DATA_IDENTITY)))
            stack.enter_context(patch.object(preflight.c,"accounting",return_value=account))
            stack.enter_context(patch.object(preflight.subprocess,"check_output",return_value="a"*40))
            def fail(command,log,timeout,path,receipt,**kwargs):
                self.assertGreater(timeout,0); self.assertLess(timeout,180)
                receipt.update(child_pid=999,process_tree_cleanup_verified=False)
                raise RuntimeError("Synthetic unknown cleanup")
            stack.enter_context(patch.object(guard,"run_guarded",side_effect=fail))
            with self.assertRaises(RuntimeError): preflight.parent("fresh",root/"data","preflight")
            lease=preflight.c.read(root/"results/research/local_lease.json")
            receipt=preflight.c.read(root/"results/research/fresh/supervisor.json")
            self.assertTrue(lease["cleanup_requires_reconciliation"])
            self.assertEqual(receipt["status"],"cleanup_unverified")
            self.assertIn("finished_at",receipt)

    def test_exact_os_nonce_script_argv_source_lease_and_membership_before_torch(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); folder=root/"results/research/fresh"; directory=folder/("process_tree_"+"a"*32)
            directory.mkdir(parents=True); guard=root/"research/process_tree_guard.py"; guard.parent.mkdir()
            shutil.copyfile(preflight.ROOT/"research/process_tree_guard.py",guard)
            manifest={"schema":preflight.INPUT_SCHEMA,"mode":"preflight","role":"engineering_preflight","preflight_path":None,"recipe":preflight.RECIPE,"seed":2026,"run":"fresh","data_dir":str(root/"data"),
                "source_git_commit":"c"*40,"effective_config":preflight.configuration(),
                "source_sha256":{"own":"a"*64},"config_sha256":preflight.CONFIGS[2026][1],
                "data_identity":preflight.c.DATA_IDENTITY,"input_sha256":{"fixture":"b"*64}}
            manifest["effective_config_sha256"]=preflight.c.json_sha(manifest["effective_config"])
            preflight.c.write(folder/"inputs.json",manifest); digest=preflight.c.sha256(folder/"inputs.json")
            now=datetime.now(timezone.utc)
            receipt=dict(schema=preflight.SCHEMA,status="benchmarking",backend="local",mode="preflight",role="engineering_preflight",run="fresh",reserved_seconds=180,
                started_at=now.isoformat(),deadline_unix=now.timestamp()+180,pid=11,process_created_at=1.5,child_pid=12,child_created_at=2.5,
                input_manifest_sha256=digest,source_sha256=manifest["source_sha256"],input_sha256=manifest["input_sha256"],confirmation_evaluated=False,formal_training_admitted=False)
            lease=dict(pid=11,process_created_at=1.5)
            spec=dict(schema="owned_python_process_tree_v1",nonce="a"*32,host_pid=11,host_created_at=1.5,cwd=str(root),
                script=str(Path(preflight.__file__).resolve()),script_sha256=preflight.c.sha256(preflight.__file__),
                guard_source_sha256=preflight.c.PINS["research/process_tree_guard.py"],arguments=["--_worker",str(folder),"--_manifest-sha",digest],
                deadline_unix=now.timestamp()+170,job_name="Local\\SaberAugMix_"+"a"*32 if os.name=="nt" else None)
            ready=dict(schema=spec["schema"],nonce=spec["nonce"],pid=12,created_at=2.5,script_sha256=spec["script_sha256"],method="windows_job_v1" if os.name=="nt" else "posix_session_v1")
            argv=[sys.executable,str(guard),"--bootstrap",str(directory/"spec.json")]
            process=SimpleNamespace(pid=12,create_time=lambda:2.5,cmdline=lambda:argv)
            parent=SimpleNamespace(pid=11,create_time=lambda:1.5)
            def save():
                for path,value in ((folder/"supervisor.json",receipt),(root/"results/research/local_lease.json",lease),(directory/"spec.json",spec),(directory/"ready.json",ready)): preflight.c.write(path,value)
            with ExitStack() as stack:
                stack.enter_context(patch.object(preflight,"ROOT",root)); stack.enter_context(patch.object(preflight.os,"getppid",return_value=11))
                stack.enter_context(patch.object(preflight.psutil,"Process",side_effect=lambda pid=None:process if pid is None else parent))
                stack.enter_context(patch.object(preflight,"configuration",return_value=manifest["effective_config"]))
                stack.enter_context(patch.object(preflight,"source_snapshot",return_value=manifest["source_sha256"]))
                stack.enter_context(patch.object(preflight,"binding_for",return_value=(manifest["input_sha256"],manifest["data_identity"])))
                boundary=stack.enter_context(patch.object(preflight.p,"verify_os_boundary"))
                save(); self.assertEqual(preflight.validate_worker(folder,digest),manifest); boundary.assert_called_once()
                for target,key,value in ((lease,"cleanup_requires_reconciliation",True),(lease,"pid",99),(spec,"nonce","b"*32),
                    (spec,"script_sha256","b"*64),(spec,"deadline_unix",now.timestamp()+200),(ready,"pid",99),(receipt,"child_created_at",99)):
                    old=target.get(key); target[key]=value; save()
                    with self.assertRaises(ValueError): preflight.validate_worker(folder,digest)
                    if old is None: target.pop(key)
                    else: target[key]=old
                save(); argv[2]="--not-bootstrap"
                with self.assertRaises(ValueError): preflight.validate_worker(folder,digest)
                argv[2]="--bootstrap"; boundary.side_effect=ValueError("outside own Job/session")
                with self.assertRaises(ValueError): preflight.validate_worker(folder,digest)

    def test_closed_preflight_requires_full_bound_inputs_cleanup_elapsed_and_stopped_owner(self):
        report,base_manifest=self.result_fixture();config=preflight.configuration()
        with tempfile.TemporaryDirectory() as temporary,ExitStack() as stack:
            root=Path(temporary);folder=root/"results/research/fresh";folder.mkdir(parents=True)
            inputs=folder/"inputs.json";output=folder/"trivialaugment_preflight.json";receipt_path=folder/"supervisor.json"
            data=root/"input.bin";data.write_bytes(b"only a synthetic fixture")
            binding={str(data):preflight.c.sha256(data)};sources=base_manifest["source_sha256"]
            manifest={**base_manifest,"schema":preflight.INPUT_SCHEMA,"recipe":preflight.RECIPE,"seed":2026,
                "role":"engineering_preflight","mode":"preflight","run":"fresh","preflight_path":None,
                "data_dir":str(root/"data"),"input_sha256":binding,"effective_config":config}
            manifest.pop("manifest_sha256");preflight.c.write(inputs,manifest)
            digest=preflight.c.sha256(inputs);report["input_manifest_sha256"]=digest;preflight.c.write(output,report)
            now=datetime.now(timezone.utc)
            receipt=dict(schema=preflight.SCHEMA,mode="preflight",role="engineering_preflight",status="complete",run="fresh",
                source_sha256=sources,input_manifest_sha256=digest,input_sha256=binding,reserved_seconds=180,
                artifact="trivialaugment_preflight.json",artifact_sha256=preflight.c.sha256(output),formal_training_admitted=False,
                process_tree_cleanup_verified=True,seed=2026,precision="bf16",config_sha256=preflight.CONFIGS[2026][1],
                started_at=now.isoformat(),finished_at=(now+timedelta(seconds=10)).isoformat(),elapsed_seconds=10.,
                pid=111,process_created_at=1.,child_pid=222,child_created_at=2.)
            stack.enter_context(patch.object(preflight,"ROOT",root));stack.enter_context(patch.object(preflight,"configuration",return_value=config))
            stack.enter_context(patch.object(preflight,"binding_for",return_value=(binding,preflight.c.DATA_IDENTITY)))
            process=stack.enter_context(patch.object(preflight.psutil,"Process",side_effect=preflight.psutil.NoSuchProcess(111)))
            def save():preflight.c.write(receipt_path,receipt)
            save();self.assertEqual(preflight.validate_preflight_bundle(output,sources)[0],report)
            for key,value in (("process_tree_cleanup_verified",False),("role","development_repeat"),("status","running"),
                              ("elapsed_seconds",.1),("input_sha256",{}),("formal_training_admitted",True),("seed",2027)):
                old=receipt[key];receipt[key]=value;save()
                with self.subTest(key=key),self.assertRaises(ValueError):preflight.validate_preflight_bundle(output,sources)
                receipt[key]=old
            save();process.side_effect=None;process.return_value=SimpleNamespace(create_time=lambda:1.)
            with self.assertRaises(ValueError):preflight.validate_preflight_bundle(output,sources)
            process.return_value=SimpleNamespace(create_time=lambda:999.)
            self.assertEqual(preflight.validate_preflight_bundle(output,sources)[0],report)


class TorchEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import runtime
        import torch
        from research import train_trivialaugment,batched_finite_proof
        cls.torch,cls.r,cls.proof=torch,train_trivialaugment,batched_finite_proof
        torch.set_num_threads(1)

    def tiny(self):
        t=self.torch
        return t.nn.Sequential(t.nn.Conv2d(3,4,3,padding=1,bias=False),t.nn.BatchNorm2d(4),t.nn.ReLU(),
            t.nn.AdaptiveAvgPool2d(1),t.nn.Flatten(),t.nn.Linear(4,10))

    def setup_step(self):
        import numpy as np
        t,r=self.torch,self.r;r.seed_all(2026);config=preflight.configuration()
        model=self.tiny();ema=copy.deepcopy(model).eval().requires_grad_(False);parameters,averages=list(model.parameters()),list(ema.parameters())
        optimizer=t.optim.SGD(parameters,lr=.1,momentum=.9,weight_decay=.0005,nesterov=True);scaler=t.amp.GradScaler("cpu",enabled=False)
        pixels=np.random.default_rng(72).integers(0,256,(512,32,32,3),dtype=np.uint8);labels=t.arange(512)%10;order=t.randperm(512)
        rng=r.ta_rng(2026);bns=[m for m in model.modules() if isinstance(m,t.nn.BatchNorm2d)]
        template=preflight.b.inventory_template(model,ema,t)
        state=dict(updates=0,correct=t.zeros(()),loss_sum=t.zeros(()),seen=0,ta_operation_counts=dict.fromkeys(preflight.OPERATIONS,0))
        return model,ema,optimizer,scaler,pixels,labels,order,config,rng,bns,template,parameters,averages,state

    def test_actual_trainer_batch_ast_vs_wrapper_preserves_state_metrics_TA_and_all_parent_rng(self):
        import random
        import numpy as np
        t,r=self.torch,self.r
        tree=ast.parse((preflight.ROOT/"research/train_trivialaugment.py").read_text())
        train=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="train")
        epoch=next(n for n in train.body if isinstance(n,ast.For) and ast.unparse(n.target)=="epoch")
        batch=next(n for n in epoch.body if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple))
        code=compile(ast.Module(body=batch.body,type_ignores=[]),"actual_TA_trainer_batch","exec");results=[]
        for original in (True,False):
            model,ema,opt,scaler,pixels,y,order,cfg,rng,bns,template,parameters,averages,state=self.setup_step()
            ns=dict(torch=t,schedule=r.schedule,config=cfg,epoch=1,optimizer=opt,model=model,ema=ema,scaler=scaler,
                    train_step=r.train_step,train_pixels=pixels,train_y=y,order=order,rng=rng,updates=0,device=t.device("cpu"),dtype=t.float32,
                    counts=dict.fromkeys(preflight.OPERATIONS,0),counts_total=dict.fromkeys(preflight.OPERATIONS,0),correct=t.zeros(()),loss_sum=t.zeros(()),seen=0)
            for i in range(4):
                if original:ns.update(batch=i,start=i*128);exec(code,ns)
                else:preflight.execute_step(r,t,model,ema,opt,pixels,y,order,i,cfg,scaler,parameters,averages,bns,template,self.proof,state,rng)
            metrics={k:(ns if original else state)[k] for k in ("updates","correct","loss_sum","seen")}
            counts=ns["counts_total"] if original else state["ta_operation_counts"]
            results.append((copy.deepcopy(model.state_dict()),copy.deepcopy(ema.state_dict()),copy.deepcopy(opt.state_dict()),copy.deepcopy(metrics),
                            copy.deepcopy(counts),copy.deepcopy(rng.bit_generator.state),random.getstate(),np.random.get_state(),t.get_rng_state().clone()))
        a,b=results
        for index in (0,1):
            for key in a[index]:self.assertTrue(t.equal(a[index][key],b[index][key]),key)
        self.assertEqual(a[2]["param_groups"],b[2]["param_groups"])
        for key in a[2]["state"]:self.assertTrue(t.equal(a[2]["state"][key]["momentum_buffer"],b[2]["state"][key]["momentum_buffer"]))
        for key in a[3]:
            if t.is_tensor(a[3][key]):self.assertTrue(t.equal(a[3][key],b[3][key]),key)
            else:self.assertEqual(a[3][key],b[3][key])
        self.assertEqual(a[4:7],b[4:7]);self.assertEqual(a[7][0],b[7][0]);self.assertTrue(np.array_equal(a[7][1],b[7][1]));self.assertEqual(a[7][2:],b[7][2:])
        self.assertTrue(t.equal(a[8],b[8]))

    def test_nonfinite_loss_or_bad_registry_or_nonadvancing_TA_rng_is_rejected(self):
        t,r=self.torch,self.r;original=r.train_step
        for kind in ("nan","inf","registry","rng"):
            model,ema,opt,scaler,pixels,y,order,cfg,rng,bns,template,parameters,averages,state=self.setup_step()
            def faulty(*args):
                before=copy.deepcopy(rng.bit_generator.state);result=original(*args)
                if kind in ("nan","inf"):result["loss"]=t.tensor(float(kind))
                elif kind=="registry":result["ta_operation_counts"][preflight.OPERATIONS[0]]+=1
                else:rng.bit_generator.state=before
                return result
            with patch.object(r,"train_step",side_effect=faulty),self.subTest(kind=kind),self.assertRaises(ValueError):
                preflight.execute_step(r,t,model,ema,opt,pixels,y,order,0,cfg,scaler,parameters,averages,bns,template,self.proof,state,rng)


if __name__=="__main__":unittest.main()
