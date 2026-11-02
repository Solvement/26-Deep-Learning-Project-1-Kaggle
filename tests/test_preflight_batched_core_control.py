"""New-entry contracts and separate CPU math/RNG equivalence; no real inputs."""
import ast
from contextlib import ExitStack, redirect_stdout
import copy
from datetime import datetime, timedelta, timezone
import io
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research import preflight_batched_core_control as preflight


class PureContracts(unittest.TestCase):
    def test_bare_help_no_action_and_unguarded_worker_fails_before_torch(self):
        before = set(sys.modules)
        with patch.object(preflight,"parent") as parent, patch.object(preflight,"worker") as worker, redirect_stdout(io.StringIO()):
            preflight.main([])
        parent.assert_not_called(); worker.assert_not_called()
        with tempfile.TemporaryDirectory() as temporary, self.assertRaises(ValueError):
            preflight.worker(Path(temporary),"a"*64)
        self.assertFalse({"torch","runtime"}&(set(sys.modules)-before))
        for args in (["--_manifest-sha","a"*64], ["--run","fresh","--_manifest-sha","a"*64]):
            with self.assertRaises(ValueError): preflight.main(args)

    def test_source_closure_is_fixed_and_torch_free(self):
        before = set(sys.modules)
        source = preflight.source_snapshot()
        self.assertEqual(source[preflight.HELPER],preflight.HELPER_SHA)
        self.assertEqual(source[preflight.PLAN],preflight.PLAN_SHA)
        self.assertEqual(source["research/preflight_core_control.py"],preflight.CORE_SHA)
        self.assertIn("research/preflight_batched_core_control.py",source)
        self.assertFalse({"torch","runtime"}&(set(sys.modules)-before))

    def test_original_core_call_auth_order_and_all_timing_are_structurally_pinned(self):
        tree = ast.parse(Path(preflight.__file__).read_text())
        execute = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="execute_step")
        calls = [n for n in ast.walk(execute) if isinstance(n,ast.Call) and ast.unparse(n.func)=="c.core_step"]
        self.assertEqual(len(calls),1)
        self.assertEqual([ast.unparse(a) for a in calls[0].args],
            ["core","torch","model","ema","optimizer","train_x","train_y","order","index","config","index","'shadow'","scaler"])
        self.assertFalse(any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) for t in n.targets) for n in ast.walk(execute)))
        worker = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="worker")
        auth = next(n.lineno for n in ast.walk(worker) if isinstance(n,ast.Call) and ast.unparse(n.func)=="validate_worker")
        imports = [n.lineno for n in ast.walk(worker) if isinstance(n,(ast.Import,ast.ImportFrom))]
        self.assertLess(auth,min(imports))
        loop = next(n for n in worker.body if isinstance(n,ast.For))
        self.assertEqual(ast.unparse(loop.iter),"range(23)")
        body = [ast.unparse(n) for n in loop.body]
        self.assertEqual(body[0],"torch.cuda.synchronize()")
        self.assertEqual(body[1],"began = time.perf_counter()")
        self.assertIn("execute_step(",body[2])
        self.assertEqual(body[3],"torch.cuda.synchronize()")
        self.assertEqual(body[4],"durations.append(time.perf_counter() - began)")
        self.assertFalse(any(isinstance(n,ast.Attribute) and n.attr=="profiler" for n in ast.walk(tree)))
        core = ast.parse((preflight.ROOT/"research/preflight_core_control.py").read_text())
        function = next(n for n in core.body if isinstance(n,ast.FunctionDef) and n.name=="core_step")
        text = ast.unparse(function)
        self.assertIn("optimizer_step(loss, optimizer, scaler)",text)
        self.assertIn("torch.isfinite(g).all()",text)
        self.assertIn("norm.item() > 0",text)
        self.assertEqual(preflight.c.sha256(preflight.ROOT/"research/preflight_core_control.py"),preflight.CORE_SHA)

    def test_guard_formula_and_original_time_budget_cannot_be_relaxed(self):
        self.assertEqual(preflight.guard_seconds(.08),1.5*.08*313*300+300)
        self.assertGreater(preflight.guard_seconds(.08),10800)
        for value in (0,-1,True,float("nan"),float("inf")):
            with self.assertRaises(ValueError): preflight.guard_seconds(value)
        now = datetime(2026,11,2,tzinfo=timezone.utc)
        policy = {"concurrent_local_gpu_jobs":1,"local_gpu_seconds_per_utc_day":21600,
                  "local_job_timeout_seconds":10800,"stops_at":(now+timedelta(days=1)).isoformat()}
        self.assertEqual(preflight.c.bounded_seconds(policy,now,0),180)
        for changed,used in ((policy,21500),({**policy,"local_job_timeout_seconds":20000},0),
                             ({**policy,"stops_at":(now+timedelta(seconds=179)).isoformat()},0)):
            with self.assertRaises(ValueError): preflight.c.bounded_seconds(changed,now,used)

    def result_fixture(self):
        config = preflight.c.configuration(2026,recipe=preflight.RECIPE)
        manifest = {"source_sha256":{"own.py":"a"*64},"config_sha256":preflight.c.WRN300_CONFIGS[2026][1],
            "effective_config_sha256":preflight.c.json_sha(config),"manifest_sha256":"b"*64,
            "source_git_commit":"c"*40,"data_identity":preflight.c.DATA_IDENTITY}
        rows = [dict(loss=1.,gradient_norm=.2,lr=.1*(i+1)/(5*313),mixed=False,updates=i+1,
            logits_dtype="torch.bfloat16",parameter_probe_max_delta=.01,ema_probe_max_delta=.001,
            all_parameters_finite=True,bn_single_forward=True,bn_counters_before=[i]*31,
            bn_counters_after=[i+1]*31,state_tensor_count=480) for i in range(23)]
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
            parameter_count=4166298,trainable_parameter_count=4166298,bn_modules=31,epochs=300,planned_formal_attempted_updates=93900,
            training_pool_examples=40000,evaluation_examples=0,batch_size=128,warmup_updates=3,timed_updates=20,
            successful_updates=23,skipped_updates=0,source_examples_seen=2944,unique_source_examples=2944,
            bn_forward_counts=[23]*31,ema_buffer_copies=0,state_tensor_count=480,state_float32_count=418,state_int64_count=62,
            preflight_state_discarded=True,profile_evidence_sha256=preflight.PROFILE_SHA,source_sha256=manifest["source_sha256"],
            config_sha256=manifest["config_sha256"],effective_config_sha256=manifest["effective_config_sha256"],
            input_manifest_sha256=manifest["manifest_sha256"],source_git_commit=manifest["source_git_commit"],data_identity=manifest["data_identity"],
            timing_scope=preflight.TIMING_SCOPE,steps=rows,step_seconds=durations,median_step_seconds=median,
            full_step_seconds=sum(durations),setup_seconds=1.,conservative_pipeline_seconds=preflight.guard_seconds(median),
            peak_gpu_memory_bytes=100,peak_gpu_reserved_bytes=120,environment=env,environment_sha256=preflight.c.json_sha(env),
            inventory_sha256="d"*64,ordered_training_indices_sha256="e"*64,ordered_official_ids_sha256="f"*64)
        return report,manifest

    def test_complete_typed_report_rejects_partial_inventory_bn_overflow_and_smoke(self):
        report,manifest = self.result_fixture()
        preflight.validate_result(report,manifest)
        for key,value in (("schema",preflight.c.REPORT_SCHEMA),("formal_training_admitted",True),("state_tensor_count",479),
                          ("state_float32_count",417),("bn_forward_counts",[22]*31),("successful_updates",22),
                          ("batch_size",384),("peak_gpu_memory_bytes",0),("conservative_pipeline_seconds",100.),
                          ("predictions",[]),("profiler",{})):
            with self.assertRaises(ValueError): preflight.validate_result({**report,key:value},manifest)
        for key,value in (("state_tensor_count",479),("gradient_norm",float("nan")),("bn_counters_after",[1]*30),
                          ("bn_counters_before",[2**63-1]*31),("bn_counters_after",[-2**63]*31)):
            changed=copy.deepcopy(report);changed["steps"][0][key]=value
            with self.assertRaises(ValueError): preflight.validate_result(changed,manifest)
        changed=copy.deepcopy(report);changed["steps"]=changed["steps"][:-1]
        with self.assertRaises(ValueError): preflight.validate_result(changed,manifest)

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

    def test_profile_evidence_is_fixed_complete_diagnostic_and_never_timing_admission(self):
        fixture=dict(schema="research_core_control_profile_v2",status="complete",diagnostic_only=True,
                     formal_admission_evidence=False,successful_updates=23,captured_updates=20,recipe=preflight.RECIPE)
        with patch.object(preflight.c,"sha256",return_value=preflight.PROFILE_SHA),patch.object(preflight.c,"read",return_value=fixture):
            preflight.profile_evidence()
            for key,value in (("status","failed"),("formal_admission_evidence",True),("captured_updates",19)):
                with patch.object(preflight.c,"read",return_value={**fixture,key:value}), self.assertRaises(ValueError):
                    preflight.profile_evidence()


    def parent_fixture(self, root, used=0):
        policy={"concurrent_local_gpu_jobs":1,"local_gpu_seconds_per_utc_day":21600,"local_job_timeout_seconds":10800,
                "stops_at":(datetime.now(timezone.utc)+timedelta(days=1)).isoformat()}
        (root/"research").mkdir()
        preflight.c.write(root/"research/campaign.json",policy)
        return {"now":lambda:datetime.now(timezone.utc),"used_seconds":lambda *a:used}

    def test_active_lease_and_insufficient_budget_refuse_before_new_artifacts_or_child(self):
        from research import process_tree_guard as guard
        for existing,used in ((True,0),(False,21500)):
            with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
                root=Path(temp); account=self.parent_fixture(root,used)
                lease=root/"results/research/local_lease.json"; lease.parent.mkdir(parents=True)
                if existing: lease.write_bytes(b'{"status":"unknown","cleanup_requires_reconciliation":true}')
                raw=lease.read_bytes() if existing else None
                stack.enter_context(patch.object(preflight,"ROOT",root))
                stack.enter_context(patch.object(preflight,"source_snapshot",return_value={}))
                stack.enter_context(patch.object(preflight,"binding_for",return_value=({},preflight.c.DATA_IDENTITY)))
                stack.enter_context(patch.object(preflight.c,"accounting",return_value=account))
                stack.enter_context(patch.object(preflight.subprocess,"check_output",return_value="a"*40))
                child=stack.enter_context(patch.object(guard,"run_guarded"))
                with self.assertRaises(ValueError): preflight.parent("fresh",root/"data")
                child.assert_not_called(); self.assertFalse((lease.parent/"fresh").exists())
                if existing: self.assertEqual(lease.read_bytes(),raw)
                else: self.assertFalse(lease.exists())

    def test_unknown_child_cleanup_keeps_matching_lease_and_receipt(self):
        from research import process_tree_guard as guard
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root=Path(temp); account=self.parent_fixture(root)
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
            with self.assertRaises(RuntimeError): preflight.parent("fresh",root/"data")
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
            manifest={"schema":preflight.INPUT_SCHEMA,"recipe":preflight.RECIPE,"seed":2026,"run":"fresh","data_dir":str(root/"data"),
                "source_git_commit":"c"*40,"effective_config":preflight.c.configuration(2026,recipe=preflight.RECIPE),
                "source_sha256":{"own":"a"*64},"config_sha256":preflight.c.WRN300_CONFIGS[2026][1],
                "data_identity":preflight.c.DATA_IDENTITY,"input_sha256":{"fixture":"b"*64}}
            manifest["effective_config_sha256"]=preflight.c.json_sha(manifest["effective_config"])
            preflight.c.write(folder/"inputs.json",manifest); digest=preflight.c.sha256(folder/"inputs.json")
            now=datetime.now(timezone.utc)
            receipt=dict(schema=preflight.SCHEMA,status="benchmarking",backend="local",mode="preflight",run="fresh",reserved_seconds=180,
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


class TorchEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import runtime
        import torch
        import train
        from research import batched_finite_proof
        cls.torch,cls.core,cls.proof = torch,train,batched_finite_proof
        torch.set_num_threads(1)

    def tiny(self):
        t=self.torch
        return t.nn.Sequential(t.nn.Conv2d(3,4,3,padding=1,bias=False),t.nn.BatchNorm2d(4),t.nn.ReLU(),
            t.nn.AdaptiveAvgPool2d(1),t.nn.Flatten(),t.nn.Linear(4,10))

    def test_new_outer_proofs_match_executed_original_loop_ast_math_sgd_ema_bn_rng(self):
        import random
        import numpy as np
        t,core=self.torch,self.core
        config={**preflight.c.configuration(2026,recipe=preflight.RECIPE),"batch_size":4,"mixup_probability":1.}
        tree=ast.parse((preflight.ROOT/"research/preflight_core_control.py").read_text())
        worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="worker")
        loop=next(n for n in worker.body if isinstance(n,ast.For) and ast.unparse(n.iter)=="range(23)")
        original=compile(ast.Module(body=loop.body,type_ignores=[]),"frozen_outer_preflight","exec")
        results=[]
        for legacy in (True,False):
            core.seed_all(2026)
            model=self.tiny();ema=copy.deepcopy(model).eval().requires_grad_(False)
            optimizer=t.optim.SGD(model.parameters(),lr=.1,momentum=.9,weight_decay=.0005,nesterov=True)
            scaler=t.amp.GradScaler("cpu",enabled=False)
            images=t.randint(0,256,(16,32,32,3),dtype=t.uint8);labels=t.arange(16)%10;order=t.randperm(16)
            parameters,averages=list(model.parameters()),list(ema.parameters())
            bns=[m for m in model.modules() if isinstance(m,t.nn.BatchNorm2d)]
            template=preflight.inventory_template(model,ema,t)
            namespace=dict(torch=t,time=__import__('time'),parameters=parameters,averages=averages,bns=bns,
                core=core,model=model,ema=ema,optimizer=optimizer,train_x=images,train_y=labels,order=order,
                config=config,recipe={"future_trainer":"shadow"},scaler=scaler,core_step=preflight.c.core_step,
                require=preflight.c.require,durations=[],rows=[])
            with patch.object(t.cuda,"synchronize"):
                for index in range(4):
                    if legacy:
                        namespace["batch"]=index;exec(original,namespace)
                    else:
                        preflight.execute_step(core,t,model,ema,optimizer,images,labels,order,index,config,scaler,
                                               parameters,averages,bns,template,self.proof)
            results.append((copy.deepcopy(model.state_dict()),copy.deepcopy(ema.state_dict()),copy.deepcopy(optimizer.state_dict()),
                            random.getstate(),np.random.get_state(),t.get_rng_state().clone()))
        left,right=results
        for index in (0,1):
            for key in left[index]:self.assertTrue(t.equal(left[index][key],right[index][key]),key)
        self.assertEqual(left[2]["param_groups"],right[2]["param_groups"])
        for key in left[2]["state"]:
            self.assertTrue(t.equal(left[2]["state"][key]["momentum_buffer"],right[2]["state"][key]["momentum_buffer"]))
        self.assertEqual(left[3],right[3]);self.assertEqual(left[4][0],right[4][0])
        self.assertTrue(np.array_equal(left[4][1],right[4][1]));self.assertEqual(left[4][2:],right[4][2:])
        self.assertTrue(t.equal(left[5],right[5]))

    def populated(self):
        t=self.torch
        model=self.tiny();ema=copy.deepcopy(model).eval().requires_grad_(False)
        optimizer=t.optim.SGD(model.parameters(),lr=.1,momentum=.9)
        template=preflight.inventory_template(model,ema,t)
        for parameter in model.parameters():optimizer.state[parameter]["momentum_buffer"]=t.ones_like(parameter)
        return model,ema,optimizer,template

    def test_full_inventory_rejects_missing_extra_mistyped_or_wrong_shape_state(self):
        t=self.torch
        for mutation in ("missing","extra","dtype","shape","ema"):
            model,ema,optimizer,template=self.populated();parameter=next(model.parameters())
            if mutation=="missing":optimizer.state.pop(parameter)
            elif mutation=="extra":optimizer.state[parameter]["extra"]=t.zeros(())
            elif mutation=="dtype":optimizer.state[parameter]["momentum_buffer"]=t.ones_like(parameter,dtype=t.float64)
            elif mutation=="shape":optimizer.state[parameter]["momentum_buffer"]=t.ones(1)
            else:ema.register_buffer("unexpected",t.zeros(()))
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                preflight.full_state_values(model,ema,optimizer,template,t)

    def test_full_inventory_all_elements_catches_nan_inf_and_keeps_maxfinite(self):
        t=self.torch
        for bad in (float("nan"),float("inf"),-float("inf"),t.finfo(t.float32).max):
            model,ema,optimizer,template=self.populated()
            momentum=optimizer.state[next(model.parameters())]["momentum_buffer"]
            momentum.flatten()[0]=bad
            values=preflight.full_state_values(model,ema,optimizer,template,t)
            self.assertEqual(bool(self.proof.all_finite(values).item()),bad==t.finfo(t.float32).max)

    def test_actual_wrn300_metadata_is_complete_480_and_no_initial_momentum_is_accepted(self):
        from model import build_model,count_parameters
        t=self.torch
        model=build_model(dict(depth=34,widen_factor=3,dropout=0.,num_classes=10))
        ema=copy.deepcopy(model).eval().requires_grad_(False)
        template=preflight.inventory_template(model,ema,t)
        self.assertEqual((count_parameters(model),len(list(model.parameters())),len(model.state_dict())),(4166298,98,191))
        self.assertEqual(len(template),480)
        self.assertEqual(sum(v[0]=="torch.float32" for v in template.values()),418)
        self.assertEqual(sum(v[0]=="torch.int64" for v in template.values()),62)
        optimizer=t.optim.SGD(model.parameters(),lr=.1,momentum=.9)
        with self.assertRaises(ValueError):preflight.full_state_values(model,ema,optimizer,template,t)


if __name__ == "__main__":
    unittest.main()
