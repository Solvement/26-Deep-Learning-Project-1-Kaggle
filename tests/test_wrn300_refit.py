"""Synthetic CPU WRN300 refit contracts; never real-data/GPU evidence."""
from contextlib import ExitStack,redirect_stdout
from datetime import datetime,timedelta,timezone
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import runtime
import numpy as np
import torch
from research import train_wrn300_refit as engine
from research import run_wrn300_refit as supervisor


def same_state(test,left,right):
    if isinstance(left,torch.Tensor):test.assertTrue(torch.equal(left,right))
    elif isinstance(left,np.ndarray):test.assertTrue(np.array_equal(left,right))
    elif isinstance(left,dict):
        test.assertEqual(set(left),set(right))
        for key in left:same_state(test,left[key],right[key])
    elif isinstance(left,(list,tuple)):
        test.assertEqual(type(left),type(right));test.assertEqual(len(left),len(right))
        for a,b in zip(left,right):same_state(test,a,b)
    else:test.assertEqual(left,right)


class WRN300RefitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)
    def setUp(self):runtime.seed_all(2026,threads=1)
    def plan(self):
        config=engine.fixed_config()
        return {"config":config,"plan_sha256":"a"*64,"parent_delivery_sha256":supervisor.PARENT_DELIVERY_SHA,
            "source_git_commit":"e"*40,
            "frozen_manifest_sha256":supervisor.PARENT_FROZEN_SHA,"source_sha256":{"synthetic_CPU_fixture":"c"*64},
            "core_source_sha256":{"train.py":"d"*64},"input_config_sha256":"b"*64}
    def tiny(self):
        return torch.nn.Sequential(torch.nn.Conv2d(3,4,3,padding=1),torch.nn.BatchNorm2d(4),torch.nn.ReLU(),
            torch.nn.Dropout(.2),torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten(),torch.nn.Linear(4,10))
    def tiny_setup(self,template):
        model=copy.deepcopy(template).train();ema=copy.deepcopy(model).eval().requires_grad_(False)
        optimizer=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9,weight_decay=.0005,nesterov=True)
        scaler=torch.amp.GradScaler("cpu",init_scale=65536.,enabled=True)
        return model,ema,optimizer,scaler

    def test_fixed_recipe_parameter_count_and_no_old_full_training_route(self):
        from research.train_shadow import validate_config as old_validate
        config=engine.fixed_config();engine.validate_config(config)
        with self.assertRaises(ValueError):old_validate(config)
        model,ema,optimizer,scaler,dtype=engine.setup(config,torch.device("cpu"),"fp32")
        self.assertEqual(sum(p.numel() for p in model.parameters()),4166298)
        self.assertFalse(scaler.is_enabled());self.assertEqual(dtype,torch.float32)
        for changes in ({"seed":2027},{"epochs":298},{"final_variant":"ema"},{"train_all":False},{"batch_size":64},{"seed":2026.}):
            with self.assertRaises(ValueError):engine.validate_config({**config,**changes})

    def test_full_official_coverage_uses_no_evaluation_or_hidden_loader(self):
        config=engine.fixed_config();images=np.broadcast_to(np.zeros((1,32,32,3),dtype=np.uint8),(50000,32,32,3))
        labels=np.arange(50000,dtype=np.int64)%10
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);(folder/"training_manifest.json").write_text(json.dumps({"split":{"seed":2026,"indices_sha256":engine.SIGNATURE["split_sha256"]}}))
            arrays=(images,labels,np.arange(40000),np.arange(40000,45000))
            with patch.object(engine,"load_training_arrays",return_value=arrays),patch.object(engine,"training_source",return_value={"archive_sha256":engine.SIGNATURE["archive_sha256"]}), \
                patch("research.train_shadow.evaluate",side_effect=AssertionError("no evaluation")):
                loaded=engine.load_full_training(folder,config)
                self.assertIs(loaded[0],images);self.assertIs(loaded[1],labels);self.assertEqual(loaded[2],engine.SIGNATURE)
                for wrong in ((images[:49999],labels[:49999],*arrays[2:]),(images,labels,np.arange(40000),np.arange(35000,40000))):
                    with patch.object(engine,"load_training_arrays",return_value=wrong):
                        with self.assertRaises(ValueError):engine.load_full_training(folder,config)
                with patch.object(engine,"training_source",return_value={"archive_sha256":"0"*64}):
                    with self.assertRaises(ValueError):engine.load_full_training(folder,config)

    def test_real_CPU_scaler_four_steps_equals_saved_two_plus_two_with_all_RNG_SGD_EMA(self):
        template=self.tiny();images=torch.randint(0,256,(2,32,32,3),dtype=torch.uint8);targets=torch.tensor([1,2]);device=torch.device("cpu")
        initial=engine.capture_rng(device)
        def run(model,ema,optimizer,scaler,first,last):
            for index in range(first,last):engine.step(model,ema,optimizer,scaler,images,targets,engine.fixed_config(),device,torch.float32,index)
        a,ea,oa,sa=self.tiny_setup(template);engine.restore_rng(initial,device);run(a,ea,oa,sa,0,4)
        final_rng=engine.capture_rng(device)
        b,eb,ob,sb=self.tiny_setup(template);engine.restore_rng(initial,device);run(b,eb,ob,sb,0,2)
        saved=copy.deepcopy({"model":b.state_dict(),"ema":eb.state_dict(),"optimizer":ob.state_dict(),"scaler":sb.state_dict(),"rng":engine.capture_rng(device)})
        c,ec,oc,sc=self.tiny_setup(template);c.load_state_dict(saved["model"]);ec.load_state_dict(saved["ema"])
        oc.load_state_dict(saved["optimizer"]);sc.load_state_dict(saved["scaler"]);engine.restore_rng(saved["rng"],device);run(c,ec,oc,sc,2,4)
        for left,right in ((a.state_dict(),c.state_dict()),(ea.state_dict(),ec.state_dict()),(oa.state_dict(),oc.state_dict()),(sa.state_dict(),sc.state_dict()),(final_rng,engine.capture_rng(device))):
            same_state(self,left,right)
        self.assertEqual(sc.state_dict()["_growth_tracker"],4)

    def test_real_CPU_overflow_skip_preserves_full_model_optimizer_and_EMA(self):
        model,ema,optimizer,scaler=self.tiny_setup(self.tiny());images=torch.randint(0,256,(2,32,32,3),dtype=torch.uint8);targets=torch.tensor([0,1])
        config=engine.fixed_config();device=torch.device("cpu")
        engine.step(model,ema,optimizer,scaler,images,targets,config,device,torch.float32,0)
        result=engine.overflow_smoke(model,ema,optimizer,scaler,images,targets,config,device,torch.float32)
        self.assertFalse(result["updated"]);self.assertEqual(result["scale_before"],65536.);self.assertEqual(result["scale_after"],32768.)
        self.assertTrue(result["all_parameters_unchanged"] and result["optimizer_state_unchanged"] and result["ema_parameters_unchanged"])

    def test_actual_tail80_step_is_one_forward_and_successful_CPU_update(self):
        model,ema,optimizer,scaler=self.tiny_setup(self.tiny())
        images=torch.randint(0,256,(80,32,32,3),dtype=torch.uint8);targets=torch.arange(80)%10
        report=engine.step(model,ema,optimizer,scaler,images,targets,engine.fixed_config(),torch.device("cpu"),torch.float32,390)
        self.assertEqual((report["source_examples"],report["forward_calls"]),(80,1));self.assertTrue(report["updated"])

    def history(self,epochs=300):
        return [{"epoch":epoch,"successful_updates":390,"skipped_updates":1,"attempted_updates":391,"source_examples":50000,"forward_calls":391,
            "source_order_sha256":"a"*64,
            "last_source_batch_size":80,"selected_variant":"model",**{key:None for key in engine.VALIDATION_KEYS},
            "finite_positive_gradient_steps":390,"actual_parameter_changed_steps":390,"skipped_probe_preserved_steps":1,
            "train_loss":1.,"train_accuracy":.8,"lr":engine.schedule(391*epoch-1,391*300,391,engine.fixed_config()),"epoch_seconds":10.,"elapsed_seconds":10.*epoch,"grad_scaler_scale":32768.} for epoch in range(1,epochs+1)]

    def test_full_counts_tail80_and_no_validation_history_are_strict(self):
        history=self.history();self.assertEqual(engine.validate_history(history,300),(117000,300))
        for key,value in (("source_examples",40000),("last_source_batch_size",64),("attempted_updates",390),("selected_variant","ema"),("validation_accuracy",.99),("finite_positive_gradient_steps",391),("skipped_updates",True)):
            changed=copy.deepcopy(history);changed[1][key]=value
            with self.assertRaises(ValueError):engine.validate_history(changed,300)
        with self.assertRaises(ValueError):engine.validate_history(history[:-1],300)
        partial=self.history(1);partial[0].update(successful_updates=387,skipped_updates=4,
            finite_positive_gradient_steps=387,actual_parameter_changed_steps=387,skipped_probe_preserved_steps=4)
        self.assertEqual(engine.validate_history(partial,1),(387,4))
        excessive=self.history()
        for row in excessive[:4]:row.update(successful_updates=1,skipped_updates=390,finite_positive_gradient_steps=1,actual_parameter_changed_steps=1,skipped_probe_preserved_steps=390)
        with self.assertRaises(ValueError):engine.validate_history(excessive,300)

    def preflight(self,natural_skips=1):
        plan=self.plan();normal=[]
        for index in range(23+natural_skips):
            success=index>=natural_skips
            normal.append({"updated":success,"mixed":True,"source_examples":128,"forward_calls":1,"elapsed_seconds":.05,"loss":2.3,
                "lr":engine.schedule(index,391*300,391,engine.fixed_config()),
                "scale_before":65536. if index==0 else 32768.,"scale_after":32768.,"finite_positive_gradient_verified":success,
                "actual_parameter_changed":success,"skip_probes_preserved":not success})
        report={"schema":engine.PREFLIGHT_SCHEMA,"status":"complete","synthetic_only":False,"training_data_loaded":True,"official_training_only":True,
            "research_training":engine.marker(plan,"fp16"),"source_sha256":plan["source_sha256"],"input_config_sha256":plan["input_config_sha256"],"data_signature":engine.SIGNATURE,
            "training_examples":50000,"validation_examples":0,"parameters":4166298,"batch_size":128,"warmup_steps":3,"timed_steps":20,
            "attempted_updates":len(normal),"successful_updates":23,"skipped_updates":natural_skips,"skipped_probe_preserved_steps":natural_skips,
            "actual_mixed_batches":len(normal),
            "source_examples_seen":128*len(normal),"source_ids_unique_count":128*len(normal),"forward_calls":len(normal),
            "finite_positive_gradient_steps":23,"actual_parameter_changed_steps":23,"full_training_resident_examples":50000,"initial_grad_scaler_scale":65536.,
            "validation_evaluated":False,"confirmation_evaluated":False,"hidden_images_loaded":False,"external_data_loaded":False,"normal_attempts":normal,
            "source_ids_sha256":"a"*64,"step_seconds":[.05]*20,"median_step_seconds":.05,"required_training_and_finalization_seconds":1.5*.05*391*300+300,
            "precision":"fp16","environment":{"device":"cuda","amp_dtype":"torch.float16","gpu":"synthetic metadata only","packages":{"torch":"synthetic"}},
            "overflow_smoke":{"status":"passed","updated":False,"source_examples":128,"forward_calls":1,"finite_loss":True,"infinite_gradient_injected":True,
                "all_parameters_unchanged":True,"optimizer_state_unchanged":True,"ema_parameters_unchanged":True,"scale_before":32768.,"scale_after":16384.}}
        return report,plan

    def test_actual_preflight_contract_natural_skip_burnin_original65536_and_391guard(self):
        report,plan=self.preflight();engine.validate_preflight(report,plan)
        for change in (lambda d:d.update(synthetic_only=True),lambda d:d.update(initial_grad_scaler_scale=256.),
            lambda d:d.update(attempted_updates=65),lambda d:d.update(full_training_resident_examples=40000),
            lambda d:d["normal_attempts"][0].update(skip_probes_preserved=False),lambda d:d["normal_attempts"][0].update(scale_before=256.),
            lambda d:d.update(required_training_and_finalization_seconds=1.5*.05*313*300+300),
            lambda d:d["research_training"]["source_sha256"].update(synthetic_CPU_fixture="0"*64),lambda d:d["overflow_smoke"].update(optimizer_state_unchanged=False)):
            modified=copy.deepcopy(report);change(modified)
            with self.assertRaises(ValueError):engine.validate_preflight(modified,plan)

    def final_fixture(self):
        from model import build_model
        plan=self.plan();config=plan["config"];model=build_model({k:config[k] for k in ("depth","widen_factor","dropout","num_classes")})
        params=list(model.parameters());state=model.state_dict();history=self.history();provenance={"source_sha256":plan["core_source_sha256"],"git_commit":plan["source_git_commit"],"research_training":engine.marker(plan,"fp16")}
        for key,value in state.items():
            if key.endswith("num_batches_tracked"):value.fill_(117300)
        optimizer=torch.optim.SGD(params,lr=history[-1]["lr"],momentum=.9,weight_decay=.0005,nesterov=True).state_dict()
        optimizer["state"]={index:{"momentum_buffer":torch.zeros_like(param)} for index,param in enumerate(params)}
        last={"schema":engine.SCHEMA,"model":state,"ema":copy.deepcopy(state),"optimizer":optimizer,
            "grad_scaler":{"scale":32768.,"growth_factor":2.,"backoff_factor":.5,"growth_interval":2000,"_growth_tracker":0},
            "config":config,"model_config":{k:config[k] for k in ("depth","widen_factor","dropout","num_classes")},"parameter_count":4166298,
            "data_signature":engine.SIGNATURE,"provenance":provenance,"epoch":300,"history":history,"updates":117000,"skipped_updates":300,
            "attempted_updates":117300,"source_examples_seen":15000000,"best_accuracy":None,"best_loss":None,**engine.capture_rng(torch.device("cpu"))}
        last["cuda_rng"]=[last["torch_rng"].clone()]
        best={key:last[key] for key in ("schema","model","model_config","config","parameter_count","data_signature","provenance","epoch")}
        best.update(variant="model",validation_accuracy=None,validation_loss=None)
        metrics={"schema":engine.SCHEMA,"status":"complete","epochs_completed":300,"config":config,"parameter_count":4166298,
            "training_examples":50000,"validation_examples":0,"best_validation_accuracy":None,"best_validation_loss":None,
            "research_training":provenance["research_training"],"successful_updates":117000,"skipped_updates":300,"attempted_updates":117300,
            "forward_calls":117300,"source_examples_seen":15000000}
        document={"schema":engine.SCHEMA,"code":provenance,"data_signature":engine.SIGNATURE,"parameter_count":4166298,"train_examples":50000,"validation_examples":0}
        env={"schema":engine.SCHEMA,"research_training":provenance["research_training"],"training_precision":"fp16","amp_dtype":"torch.float16",
            "grad_scaler_enabled":True,"initial_grad_scaler_scale":65536.,"device":"cuda"}
        return plan,best,last,metrics,history,document,env

    def test_deep_final50k_audit_rejects_counts_protocol_and_checkpoint_selection(self):
        original=self.final_fixture();supervisor.validate_records(*original)
        for change in (lambda d:d[1].update(epoch=298),lambda d:d[1].update(variant="ema"),lambda d:d[3].update(source_examples_seen=12000000),
            lambda d:d[3].update(validation_examples=5000),lambda d:d[2]["optimizer"]["state"][0].update(momentum_buffer=torch.zeros(1)),
            lambda d:d[2].update(grad_scaler={}),lambda d:d[6].update(amp_dtype="torch.bfloat16"),
            lambda d:d[1].update(validation_accuracy=.96)):
            docs=copy.deepcopy(original);change(docs)
            with self.assertRaises(ValueError):supervisor.validate_records(*docs)

    def policy(self):return {"concurrent_local_gpu_jobs":1,"local_gpu_seconds_per_utc_day":21600,"local_job_timeout_seconds":10800,"stops_at":"2026-11-05T05:00:00+00:00"}
    def test_fixed_budget_daily_hardstop_and_new_measured_guard(self):
        policy=self.policy();current=datetime(2026,11,1,tzinfo=timezone.utc)
        self.assertEqual(supervisor.admission_budget(policy,current,0),10800)
        self.assertEqual(supervisor.admission_budget(policy,current,21000),600)
        stop=datetime.fromisoformat(policy["stops_at"]);self.assertEqual(supervisor.admission_budget(policy,stop-timedelta(seconds=10),0),10)
        for used in (True,-1,float("nan"),float("inf"),21600):
            with self.assertRaises(ValueError):supervisor.admission_budget(policy,current,used)
        with self.assertRaises(ValueError):supervisor.admission_budget(policy,stop,0)
        with self.assertRaises(ValueError):supervisor.admission_budget(policy,current,0,{"median_step_seconds":.1})
        self.assertEqual(supervisor.admission_budget(policy,current,0,{"median_step_seconds":.05}),10800)

    def test_help_and_direct_worker_fail_before_native_or_data_work(self):
        with redirect_stdout(io.StringIO()),patch.object(supervisor,"parent") as parent,patch.object(supervisor,"worker") as worker:
            supervisor.main([]);parent.assert_not_called();worker.assert_not_called()
        with self.assertRaises(ValueError):supervisor.main(["--_worker","."])
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):supervisor.worker(Path(temp),"0"*64)

    def contract(self,root):
        folder=root/"results/research/synthetic_pf";folder.mkdir(parents=True)
        plan=root/"synthetic_plan.json";plan.write_text("synthetic plan bytes, no training")
        plan_sha=supervisor.sha256(plan);binding={str(plan):plan_sha};source={"synthetic_CPU_fixture":"a"*64}
        manifest={"mode":"preflight","source_sha256":source,"input_sha256":binding,"plan_path":str(plan),"plan_sha256":plan_sha}
        (folder/"refit_inputs.json").write_text(json.dumps(manifest));digest=supervisor.sha256(folder/"refit_inputs.json")
        moment=datetime.now(timezone.utc);parent=SimpleNamespace(pid=202,create_time=lambda:9.);worker=SimpleNamespace(pid=101,create_time=lambda:10.)
        receipt={"schema":supervisor.SCHEMA,"mode":"preflight","status":"benchmarking","source_sha256":source,"input_sha256":binding,
            "input_manifest_sha256":digest,"plan_sha256":plan_sha,"started_at":moment.isoformat(),"reserved_seconds":120.,"deadline_unix":moment.timestamp()+120,
            "pid":202,"process_created_at":9.,"child_pid":101,"child_created_at":10.}
        (folder/"supervisor.json").write_text(json.dumps(receipt));lease=root/"results/research/local_lease.json";lease.write_text(json.dumps({"pid":202,"process_created_at":9.}))
        directory=folder/("process_tree_"+"a"*32);directory.mkdir()
        spec={"schema":"owned_python_process_tree_v1","nonce":"a"*32,"host_pid":202,"host_created_at":9.,"cwd":str(root),
            "script":str(Path(supervisor.__file__).resolve()),"script_sha256":supervisor.sha256(supervisor.__file__),
            "guard_source_sha256":supervisor.BASE_SOURCES["research/process_tree_guard.py"],
            "arguments":["--_worker",str(folder),"--_manifest-sha",digest],"deadline_unix":receipt["deadline_unix"],
            "job_name":"Local\\SaberAugMix_"+"a"*32 if supervisor.os.name=="nt" else None}
        ready={"schema":spec["schema"],"nonce":spec["nonce"],"pid":101,"created_at":10.,"script_sha256":spec["script_sha256"],
            "method":"windows_job_v1" if supervisor.os.name=="nt" else "posix_session_v1"}
        (directory/"spec.json").write_text(json.dumps(spec));(directory/"ready.json").write_text(json.dumps(ready))
        worker.cmdline=lambda:[supervisor.sys.executable,str(root/"research/process_tree_guard.py"),"--bootstrap",str(directory/"spec.json")]
        return folder,digest,worker,parent,lease,directory,spec,source

    def test_worker_guard_live_canonical_lease_source_pid_nonce_and_OS_membership(self):
        for change in ("none","missing","foreign","unknown","nonce","command","source","budget","plan","membership"):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
                root=Path(temp);folder,digest,worker,parent,lease,directory,spec,source=self.contract(root)
                original=supervisor.sha256
                stack.enter_context(patch.object(supervisor,"ROOT",root));stack.enter_context(patch.object(supervisor,"source_snapshot",return_value=source))
                stack.enter_context(patch.object(supervisor,"sha256",side_effect=lambda p:spec["guard_source_sha256"] if Path(p)==root/"research/process_tree_guard.py" else original(p)))
                stack.enter_context(patch.object(supervisor.psutil,"Process",side_effect=lambda pid=None:worker if pid is None else parent))
                stack.enter_context(patch.object(supervisor.os,"getppid",return_value=202))
                native=stack.enter_context(patch("research.profile_augmix.verify_os_boundary"))
                if change=="missing":lease.unlink()
                elif change=="foreign":lease.write_text(json.dumps({"pid":999,"process_created_at":9.}))
                elif change=="unknown":lease.write_text(json.dumps({"pid":202,"process_created_at":9.,"cleanup_requires_reconciliation":True}))
                elif change=="nonce":
                    ready=json.loads((directory/"ready.json").read_text());ready["nonce"]="b"*32;(directory/"ready.json").write_text(json.dumps(ready))
                elif change=="command":worker.cmdline=lambda:[supervisor.sys.executable,str(supervisor.__file__)]
                elif change=="source":spec["script_sha256"]="0"*64;(directory/"spec.json").write_text(json.dumps(spec))
                elif change=="budget":
                    receipt=supervisor.read(folder/"supervisor.json");receipt["reserved_seconds"]=10800;(folder/"supervisor.json").write_text(json.dumps(receipt))
                elif change=="plan":(root/"synthetic_plan.json").write_text("changed")
                elif change=="membership":native.side_effect=ValueError("not an actual OS boundary")
                if change=="none":supervisor.validate_worker(folder,digest);native.assert_called_once()
                else:
                    with self.assertRaises((ValueError,FileNotFoundError)):supervisor.validate_worker(folder,digest)
                    if change not in ("plan","membership"):native.assert_not_called()

    def completed_guard(self,root):
        folder,digest,worker,parent,lease,directory,spec,source=self.contract(root)
        receipt=supervisor.read(folder/"supervisor.json");ready=supervisor.read(directory/"ready.json")
        completed={**ready,"exit_code":0}
        host={"schema":spec["schema"],"nonce":spec["nonce"],"method":ready["method"],"guard_directory":str(directory),
            "guard_source_sha256":spec["guard_source_sha256"],"cleanup_verified":True,"child_pid":101,"child_created_at":10.,
            "child_returncode":0,"timed_out":False,"completion_authenticated":True,"completion_exit_code":0}
        (directory/"completion.json").write_text(json.dumps(completed));(directory/"host_result.json").write_text(json.dumps(host))
        receipt.update(process_tree_cleanup_verified=True,process_tree_guard_results=[host])
        return folder,directory,receipt

    def test_durable_guard_proof_and_original_attempt_manifest_are_deeply_bound(self):
        for change in ("none","lost_inputs","old_backup","nonce","exit","method","command","cleanup","deadline"):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);folder,directory,receipt=self.completed_guard(root)
                if change in ("lost_inputs","old_backup"):
                    original=(folder/"refit_inputs.json").read_bytes()
                    if change=="old_backup":(folder/"refit_inputs_attempt_1.json").write_bytes(original)
                    (folder/"refit_inputs.json").write_text("later resume manifest bytes")
                elif change in ("nonce","exit","method"):
                    done=supervisor.read(directory/"completion.json")
                    done[{"nonce":"nonce","exit":"exit_code","method":"method"}[change]]={"nonce":"b"*32,"exit":1,"method":"fake"}[change]
                    (directory/"completion.json").write_text(json.dumps(done))
                elif change in ("command","deadline"):
                    spec=supervisor.read(directory/"spec.json")
                    spec["arguments" if change=="command" else "deadline_unix"]=["--run","unowned"] if change=="command" else receipt["deadline_unix"]+100
                    (directory/"spec.json").write_text(json.dumps(spec))
                elif change=="cleanup":receipt["process_tree_cleanup_verified"]=False
                with patch.object(supervisor,"ROOT",root):
                    if change in ("none","old_backup"):
                        paths=supervisor.guard_evidence(folder,receipt)
                        self.assertIn(str((directory/"completion.json").resolve()),paths)
                        if change=="old_backup":self.assertIn(str((folder/"refit_inputs_attempt_1.json").resolve()),paths)
                    else:
                        with self.assertRaises(ValueError):supervisor.guard_evidence(folder,receipt)

    def test_resume_state_plan_source_and_exact_epoch_ledgers_rejected_when_changed(self):
        plan,best,last,metrics,history,provenance,env=self.final_fixture()
        last=copy.deepcopy(last);last.update(epoch=1,history=self.history(1),updates=390,skipped_updates=1,attempted_updates=391,source_examples_seen=50000)
        for kind in ("model","ema"):
            for key,value in last[kind].items():
                if key.endswith("num_batches_tracked"):value.fill_(391)
        last["optimizer"]["param_groups"][0]["lr"]=last["history"][-1]["lr"]
        current=last["provenance"]
        engine.validate_resume(last,plan["config"],engine.SIGNATURE,current)
        for change in (lambda d:d.update(attempted_updates=313),lambda d:d.update(source_examples_seen=40000),lambda d:d["provenance"]["research_training"].update(plan_sha256="0"*64),
            lambda d:d.update(best_accuracy=.97),lambda d:d.update(epoch=300),lambda d:d.pop("torch_rng")):
            modified=copy.deepcopy(last);change(modified)
            with self.assertRaises(ValueError):engine.validate_resume(modified,plan["config"],engine.SIGNATURE,current)

    def test_attempt_budget_time_contract_rejects_bool_extension_and_hardstop(self):
        plan={**self.plan(),"prepared_at":"2026-11-01T00:00:00+00:00"}
        receipt={"schema":supervisor.SCHEMA,"backend":"local","mode":"preflight","source_sha256":plan["source_sha256"],"plan_sha256":plan["plan_sha256"],
            "started_at":"2026-11-01T01:00:00+00:00","finished_at":"2026-11-01T01:00:20+00:00","reserved_seconds":180,
            "elapsed_seconds":20.,"status":"complete","deadline_unix":datetime(2026,11,1,1,3,tzinfo=timezone.utc).timestamp()}
        supervisor.receipt_timing(receipt,plan)
        for fields in ({"elapsed_seconds":True},{"reserved_seconds":10801},{"elapsed_seconds":181.1},
            {"finished_at":"2026-11-05T05:01:00+00:00"},{"plan_sha256":"0"*64}):
            with self.assertRaises(ValueError):supervisor.receipt_timing({**receipt,**fields},plan)
        with self.assertRaises(ValueError):supervisor.admission_budget({**self.policy(),"stops_at":"2026-11-06T05:00:00+00:00"},datetime(2026,11,1,tzinfo=timezone.utc),0)

    def test_prepare_and_validate_new_plan_changes_only_coverage_and_final_variant(self):
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            root=Path(temp);(root/"research").mkdir();(root/"research/campaign.json").write_text(json.dumps(self.policy()))
            parent=root/"synthetic_parent.json";parent.write_text("synthetic parent receipt bytes")
            original={**engine.fixed_config(),"train_all":False,"final_variant":"ema"}
            frozen={"config":original,"data_signature":engine.SIGNATURE,"data_dir":str(root/"data"),
                "evaluation_context":{"device_type":"cuda","amp_dtype":"torch.bfloat16","torch_version":"synthetic CPU fixture"}}
            document={"frozen_manifest_path":str(root/"synthetic_frozen.json")}
            stack.enter_context(patch.object(supervisor,"ROOT",root))
            stack.enter_context(patch.object(supervisor,"now",return_value=datetime(2026,11,1,tzinfo=timezone.utc)))
            stack.enter_context(patch.object(supervisor,"load_parent",return_value=({},document,frozen,[str(parent)])))
            stack.enter_context(patch.object(supervisor,"source_snapshot",return_value={"synthetic.py":"a"*64}))
            commit=stack.enter_context(patch.object(supervisor,"committed_source",return_value={"synthetic.py":{
                "raw_sha256":"a"*64,"git_blob_content_sha256":"a"*64,"serialization":"exact_bytes"}}))
            stack.enter_context(patch("train.code_provenance",return_value={"git_commit":"e"*40}))
            path=supervisor.prepare("synthetic_refit",parent)
            plan=supervisor.validate_plan(path)
            self.assertEqual({k:v for k,v in plan["config"].items() if k not in ("train_all","final_variant")},
                {k:v for k,v in original.items() if k not in ("train_all","final_variant")})
            self.assertTrue(plan["config"]["train_all"]);self.assertEqual(plan["weight_variant"],"model")
            self.assertFalse(plan["old_refit_helpers_support_this_extension"])
            before=path.read_bytes()
            with self.assertRaises(ValueError):supervisor.prepare("synthetic_refit",parent)
            self.assertEqual(path.read_bytes(),before)
            commit.assert_called()
            config_path=path.parent/"input_config.json";config_path.write_text("{}")
            with self.assertRaises(ValueError):supervisor.validate_plan(path)

    def test_recorded_commit_hashes_are_checked_without_requiring_current_HEAD(self):
        raw=b"synthetic committed source";fingerprint=hashlib.sha256(raw).hexdigest()
        with patch.object(supervisor.subprocess,"check_output",return_value=raw) as git:
            record=supervisor.committed_source("e"*40,{"synthetic.py":fingerprint})
            self.assertEqual(record,{"synthetic.py":{"raw_sha256":fingerprint,"git_blob_content_sha256":fingerprint,"serialization":"exact_bytes"}})
            self.assertEqual(git.call_args.args[0],["git","show","e"*40+":synthetic.py"])
            with self.assertRaises(ValueError):supervisor.committed_source("e"*40,{"synthetic.py":"0"*64})
        for value in (None,"HEAD","e"*39):
            with self.assertRaises(ValueError):supervisor.committed_source(value,{})

    def test_only_two_pinned_CRLF_serializations_are_allowed_and_one_byte_drift_is_rejected(self):
        for name,(raw_sha,blob_sha) in supervisor.FROZEN_CRLF.items():
            blob=(supervisor.ROOT/name).read_bytes().replace(b"\r\n",b"\n")
            raw=blob.replace(b"\n",b"\r\n")
            self.assertEqual(hashlib.sha256(raw).hexdigest(),raw_sha)
            self.assertEqual(hashlib.sha256(blob).hexdigest(),blob_sha)
            with self.subTest(name=name),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);path=root/name;path.parent.mkdir(parents=True);path.write_bytes(raw)
                with patch.object(supervisor,"ROOT",root),patch.object(supervisor.subprocess,"check_output",return_value=blob):
                    records=supervisor.committed_source("e"*40,{name:raw_sha})
                    self.assertEqual(records[name],{"raw_sha256":raw_sha,"git_blob_content_sha256":blob_sha,"serialization":"frozen_CRLF_to_LF"})
                    changed=raw[:-3]+b"X"+raw[-2:];path.write_bytes(changed)
                    with self.assertRaises(ValueError):supervisor.committed_source("e"*40,{name:raw_sha})
                    with self.assertRaises(ValueError):supervisor.committed_source("e"*40,{name:hashlib.sha256(changed).hexdigest()})
                    path.write_bytes(raw)
                    with self.assertRaises(ValueError):supervisor.committed_source("e"*40,{"research/unregistered.py":raw_sha})
                with patch.object(supervisor,"ROOT",root),patch.object(supervisor.subprocess,"check_output",return_value=blob+b"X"):
                    with self.assertRaises(ValueError):supervisor.committed_source("e"*40,{name:raw_sha})

    def parent_fixture(self,root,stack,used,guarded,*,resume=False):
        from research import run_local,process_tree_guard
        (root/"research").mkdir();(root/"research/campaign.json").write_text(json.dumps(self.policy()))
        (root/"synthetic.py").write_text("synthetic source")
        data=root/"data";data.mkdir();(data/"pixels").write_bytes(b"synthetic bytes, never loaded")
        (data/"training_manifest.json").write_text(json.dumps({"files_sha256":{"pixels":"synthetic"}}))
        folder=root/"results/research/synthetic_refit";folder.mkdir(parents=True)
        plan_path=folder/"refit_plan.json";plan_path.write_text("synthetic prepared plan, never a real artifact")
        raw=folder/"input_config.json";raw.write_text(json.dumps(engine.fixed_config()))
        plan={**self.plan(),"run":folder.name,"plan_path":str(plan_path),"plan_sha256":supervisor.sha256(plan_path),"data_dir":str(data),
            "input_config_path":str(raw),"input_config_sha256":supervisor.sha256(raw),"parent_evidence_sha256":{},
            "source_sha256":{"synthetic.py":supervisor.sha256(root/"synthetic.py")},"prepared_at":"2026-11-01T00:00:00+00:00"}
        preflight=root/"synthetic_pref.json";preflight.write_text("synthetic preflight receipt, never admission")
        args=SimpleNamespace(plan=plan_path,mode="train" if resume else "preflight",run=None if resume else "synthetic_pf",preflight=preflight if resume else None,resume=resume)
        binding={str(p.resolve()):supervisor.sha256(p) for p in (plan_path,raw,data/"training_manifest.json",data/"pixels",root/"synthetic.py")}
        stack.enter_context(patch.object(supervisor,"ROOT",root));stack.enter_context(patch.object(supervisor,"validate_plan",return_value=plan))
        stack.enter_context(patch.object(supervisor,"source_snapshot",return_value=plan["source_sha256"]))
        stack.enter_context(patch.object(supervisor,"now",return_value=datetime(2026,11,1,12,tzinfo=timezone.utc)))
        stack.enter_context(patch.object(run_local,"used_seconds",side_effect=used if isinstance(used,list) else None,return_value=used if not isinstance(used,list) else None))
        stack.enter_context(patch.object(supervisor,"checked_preflight",return_value=({"median_step_seconds":.05,"environment":{"packages":{"torch":"synthetic"}}},{})))
        stack.enter_context(patch("runtime.environment",return_value={"packages":{"torch":"synthetic"}}))
        stack.enter_context(patch.object(engine,"validate_preflight",return_value={}))
        guard=stack.enter_context(patch.object(process_tree_guard,"run_guarded",side_effect=guarded))
        return args,plan,binding,guard

    def test_preflight_parent_uses_shared_guard_180_and_unknown_cleanup_locks_lease(self):
        for unknown in (False,True):
            with self.subTest(unknown=unknown),tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
                root=Path(temp)
                def guarded(command,log,seconds,receipt_path,receipt,cwd):
                    self.assertLessEqual(seconds,180);self.assertIn("--_worker",command)
                    receipt["process_tree_cleanup_verified"]=not unknown
                    if unknown:raise RuntimeError("unknown owned tree")
                    (receipt_path.parent/"native_preflight.json").write_text(json.dumps({"status":"complete"}))
                args,plan,binding,guard=self.parent_fixture(root,stack,100,guarded)
                if unknown:
                    with self.assertRaisesRegex(RuntimeError,"unknown owned tree"):supervisor.parent(args)
                    lease=supervisor.read(root/"results/research/local_lease.json");self.assertTrue(lease["cleanup_requires_reconciliation"])
                else:
                    supervisor.parent(args);self.assertFalse((root/"results/research/local_lease.json").exists())
                    receipt=supervisor.read(root/"results/research/synthetic_pf/supervisor.json")
                    self.assertEqual(receipt["status"],"complete");self.assertFalse(receipt["submitted"])

    def test_resume_budget_refusal_keeps_old_receipt_and_attempt_input_bytes(self):
        for usage in ([21600],[0,21600]):
            with self.subTest(usage=usage),tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
                root=Path(temp);args,plan,binding,guard=self.parent_fixture(root,stack,usage,lambda *a,**k:None,resume=True)
                folder=args.plan.parent;(folder/"last.pt").write_bytes(b"synthetic checkpoint, not loaded")
                old_inputs=json.dumps({"mode":"train","plan_sha256":plan["plan_sha256"],"source_sha256":plan["source_sha256"],"input_sha256":binding}).encode()
                (folder/"refit_inputs.json").write_bytes(old_inputs)
                previous={"schema":supervisor.SCHEMA,"backend":"local","mode":"train","status":"failed","run":folder.name,
                    "plan_sha256":plan["plan_sha256"],"source_sha256":plan["source_sha256"],"input_sha256":binding,
                    "preflight_sha256":supervisor.sha256(args.preflight),"input_manifest_sha256":hashlib.sha256(old_inputs).hexdigest(),
                    "pid":2147483647,"process_created_at":0.,"process_tree_cleanup_verified":True,
                    "started_at":"2026-11-01T01:00:00+00:00","finished_at":"2026-11-01T02:00:00+00:00","reserved_seconds":10800,"elapsed_seconds":3600.}
                receipt=folder/"supervisor.json";receipt.write_text(json.dumps(previous));original=receipt.read_bytes()
                with self.assertRaisesRegex(ValueError,"budget|remaining"):supervisor.parent(args)
                guard.assert_not_called();self.assertFalse((root/"results/research/local_lease.json").exists())
                if len(usage)==1:
                    self.assertEqual(receipt.read_bytes(),original);self.assertFalse((folder/"supervisor_raw_1.json").exists())
                else:
                    self.assertEqual((folder/"supervisor_raw_1.json").read_bytes(),original)
                    self.assertEqual((folder/"refit_inputs_attempt_1.json").read_bytes(),old_inputs)
                    current=supervisor.read(receipt);self.assertEqual(current["previous_attempt"],"supervisor_attempt_1.json");self.assertEqual(current["reserved_seconds"],0)


if __name__=="__main__":unittest.main()
