"""Synthetic CPU protocol tests; fixtures are never native GPU evidence."""
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
from research import augmix_nativebn as protocol
from research import run_augmix_nativebn as supervisor
from research import benchmark_augmix_nativebn as audit
from research import train_augmix as engine
from research import benchmark_augmix as old_audit
import test_augmix_training as fixtures


class NativeBNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)
    def setUp(self):
        runtime.seed_all(2026,threads=1)
        # CI never has the private actual-GPU receipt. Unit contracts bind a
        # declared synthetic source snapshot, never pretend it is that receipt.
        self.real_bound_sources=protocol.bound_sources
        source={key:protocol.sha256(protocol.ROOT/key) for key in protocol.NEW_SOURCES}
        source["synthetic_CPU_fixture_only"]="a"*64
        binding=patch.object(protocol,"bound_sources",return_value=source)
        binding.start();self.addCleanup(binding.stop)
    def tiny(self):
        return torch.nn.Sequential(torch.nn.Conv2d(3,4,3,padding=1),torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten(),torch.nn.Linear(4,10))
    def raw(self): return (protocol.ROOT/supervisor.CONFIG).read_bytes()
    def proof(self,attempts,successes,skipped):
        result = protocol.new_proof()
        result.update(attempted_updates=attempts,successful_updates=successes,skipped_updates=skipped,
            training_bn_forward_calls=19*attempts,bn_single_update_verified_calls=19*attempts,
            finite_positive_gradient_steps=successes,actual_parameter_changed_steps=successes,
            skipped_probe_preserved_steps=skipped,convolution_cudnn_enabled_calls=22*attempts,
            observed_training_bn_nodes=["NativeBatchNormBackward0"],real_cuda_backend_verified=True)
        return result

    def test_exact_source_config_diagnostic_and_parameter_count(self):
        from model import build_model,count_parameters
        self.assertEqual(hashlib.sha256(self.raw()).hexdigest(),protocol.CONFIG_SHA)
        source = protocol.bound_sources()
        self.assertTrue(set(protocol.NEW_SOURCES) <= set(source))
        model = build_model({"depth":22,"widen_factor":4,"dropout":0.,"num_classes":10})
        before = {k:v.clone() for k,v in model.state_dict().items()}
        ids = [id(p) for p in model.parameters()]
        with protocol.native_training_context(engine.fixed_config(),protocol.CONFIG_SHA):
            self.assertEqual(count_parameters(model),4298970)
            self.assertEqual(sum(isinstance(m,torch.nn.BatchNorm2d) for m in model.modules()),19)
            self.assertEqual(ids,[id(p) for p in model.parameters()])
            self.assertTrue(all(torch.equal(before[k],v) for k,v in model.state_dict().items()))
        # Exercise the real source/receipt guard entirely with temporary bytes.
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for key in (*protocol.NEW_SOURCES,"research/benchmark_augmix.py","research/benchmark.py"):
                path=root/key;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b"synthetic CPU source")
            receipt=root/protocol.DIAGNOSTIC;receipt.parent.mkdir(parents=True);receipt.write_bytes(b"synthetic CPU receipt")
            with patch.object(protocol,"ROOT",root),patch.object(protocol,"DIAGNOSTIC_SHA",protocol.sha256(receipt)), \
                patch("research.profile_augmix.source_snapshot",return_value={"base":"f"*64}):
                self.assertIn(protocol.NEW_SOURCES[0],self.real_bound_sources())
                receipt.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError,"diagnostic changed"):self.real_bound_sources()

    def test_training_only_native_scope_eval_original_and_exception_restore(self):
        model = self.tiny();seen=[]
        original = torch.nn.functional.batch_norm
        original_bn,original_conv,original_step = torch.nn.BatchNorm2d.forward,torch.nn.Conv2d.forward,engine.train_step
        def observe(*args,**kwargs):
            seen.append(torch.backends.cudnn.enabled)
            return original(*args,**kwargs)
        with protocol.native_training_context(engine.fixed_config(),protocol.CONFIG_SHA,strict=False) as proof:
            with patch.object(torch.nn.functional,"batch_norm",side_effect=observe):
                model.train()(torch.randn(6,3,32,32)).sum().backward()
                model.eval()(torch.randn(6,3,32,32))
            self.assertEqual(seen,[False,True])
            self.assertEqual(proof["training_bn_forward_calls"],1)
            self.assertEqual(proof["evaluation_bn_forward_calls"],1)
            with patch.object(torch.nn.functional,"batch_norm",side_effect=RuntimeError("synthetic")):
                with self.assertRaisesRegex(RuntimeError,"synthetic"): model.train()(torch.randn(6,3,32,32))
            self.assertTrue(torch.backends.cudnn.enabled)
        self.assertIs(torch.nn.BatchNorm2d.forward,original_bn)
        self.assertIs(torch.nn.Conv2d.forward,original_conv)
        self.assertIs(engine.train_step,original_step)
        with self.assertRaisesRegex(RuntimeError,"exit"):
            with protocol.native_training_context(engine.fixed_config(),protocol.CONFIG_SHA): raise RuntimeError("exit")
        self.assertIs(torch.nn.BatchNorm2d.forward,original_bn)

    def test_original_step_numeric_rng_sgd_ema_unchanged_on_CPU(self):
        model = self.tiny();other=copy.deepcopy(model)
        ema,other_ema=copy.deepcopy(model).eval(),copy.deepcopy(model).eval()
        optimizers=[torch.optim.SGD(m.parameters(),lr=.1,momentum=.9,weight_decay=.0005,nesterov=True) for m in (model,other)]
        prepared=(torch.randn(6,3,32,32),torch.tensor([1,2]))
        rng=np.random.Generator(np.random.PCG64(2026));sampler,loader=engine.pipeline_generators()
        before=engine.capture_state_rng(rng,torch.device("cpu"),sampler,loader)
        first=engine.train_step(model,optimizers[0],torch.amp.GradScaler("cpu",enabled=False),prepared,torch.device("cpu"),torch.float32,ema=ema)
        engine.restore_state_rng(before,rng,torch.device("cpu"),sampler,loader)
        with protocol.native_training_context(engine.fixed_config(),protocol.CONFIG_SHA,strict=False) as proof:
            second=engine.train_step(other,optimizers[1],torch.amp.GradScaler("cpu",enabled=False),prepared,torch.device("cpu"),torch.float32,ema=other_ema)
        self.assertEqual(first,second)
        fixtures.assert_state_equal(self,model.state_dict(),other.state_dict())
        fixtures.assert_state_equal(self,ema.state_dict(),other_ema.state_dict())
        fixtures.assert_state_equal(self,optimizers[0].state_dict(),optimizers[1].state_dict())
        self.assertEqual(proof["actual_parameter_changed_steps"],1)
        self.assertFalse(proof["real_cuda_backend_verified"])
        after=engine.capture_state_rng(rng,torch.device("cpu"),sampler,loader)
        fixtures.assert_state_equal(self,before,after)

    def test_skip_keeps_probes_and_allows_native_bn_single_update(self):
        model=self.tiny();ema=copy.deepcopy(model).eval();optimizer=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9)
        prepared=(torch.randn(6,3,32,32),torch.tensor([0,1]))
        scaler=fixtures.ForcedSkipScaler()
        before=copy.deepcopy(model.state_dict())
        with protocol.native_training_context(engine.fixed_config(),protocol.CONFIG_SHA,strict=False) as proof:
            report=engine.train_step(model,optimizer,scaler,prepared,torch.device("cpu"),torch.float32,ema=ema)
        self.assertFalse(report["updated"])
        self.assertEqual(proof["skipped_probe_preserved_steps"],1)
        self.assertEqual(proof["finite_positive_gradient_steps"],0)
        self.assertEqual(int(model[1].num_batches_tracked),int(before["1.num_batches_tracked"])+1)

    def complete(self,skipped=1):
        documents=list(fixtures.AugMixSourceAuditTests().complete_fixture("fp16",skipped))
        config=documents[0]["config"]
        documents[5]=self.raw()
        marker=protocol.research_provenance("fp16",config,protocol.CONFIG_SHA)
        for index in (0,1,3,4,6):documents[index]["schema"]=protocol.SCHEMA
        documents[0]["provenance"]["research_training"]=copy.deepcopy(marker)
        documents[1]["research_training"]=copy.deepcopy(marker)
        documents[3]["code"]["research_training"]=copy.deepcopy(marker)
        documents[4]["research_training"]=copy.deepcopy(marker)
        documents[6]["provenance"]["research_training"]=copy.deepcopy(marker)
        proof=self.proof(50080,50080-skipped,skipped)
        for index in (0,1,6):documents[index]["nativebn_execution_proof"]=copy.deepcopy(proof)
        return documents

    def test_new_validator_deep_normalizes_metadata_old_validator_rejects_raw(self):
        documents=self.complete()
        old_marker=copy.deepcopy(documents[0]["provenance"])
        model_pointer=documents[0]["model"]
        with self.assertRaises(ValueError):old_audit.validate_augmix(*documents)
        result=audit.validate_nativebn(*documents)
        self.assertEqual(result["schema"],"research_augmix_nativebn_integrity_v1")
        self.assertEqual((result["attempted_updates"],result["source_examples_seen"],result["model_views_seen"]),(50080,6400000,19200000))
        self.assertEqual(result["training"]["method"],"augmix_jsd_nativebn")
        self.assertIs(documents[0]["model"],model_pointer)
        self.assertEqual(documents[0]["provenance"],old_marker)

    def test_new_audit_rejects_wrong_backend_source_partial_overlap_counts_and_proofs(self):
        original=self.complete()
        changes=[lambda d:d[0]["provenance"]["research_training"]["bn_backend_extension"].update(evaluation_batchnorm_backend="native"),
            lambda d:d[1].update(epochs_completed=159),lambda d:d[1].update(training_examples=50000),
            lambda d:d[1].update(source_examples_seen=6400001),lambda d:d[2][5].update(model_views=80000),
            lambda d:d[6]["nativebn_execution_proof"].update(training_bn_forward_calls=50080),
            lambda d:d[0]["nativebn_execution_proof"].update(real_cuda_backend_verified=False),
            lambda d:d[0]["provenance"]["research_training"]["bn_backend_extension"]["source_sha256"].update({protocol.NEW_SOURCES[0]:"0"*64}),
            lambda d:d[6]["optimizer"]["state"][0].update(momentum_buffer=torch.zeros(1))]
        for index,change in enumerate(changes):
            with self.subTest(index=index):
                docs=copy.deepcopy(original);change(docs)
                with self.assertRaises(ValueError):audit.validate_nativebn(*docs)

    def test_future_seed27_and_old_protocol_resume_are_not_admitted(self):
        config,signature,provenance,last=fixtures.AugMixSourceAuditTests().resume_fixture()
        raw_sha=protocol.CONFIG_SHA
        with protocol.native_training_context(config,raw_sha):
            current={**provenance,"research_training":engine.research_provenance("fp32",config,raw_sha)}
            with self.assertRaises(ValueError):engine.validate_resume(last,config,signature,current)
        with self.assertRaisesRegex(ValueError,"primary2026"):
            protocol.run_engine("train",engine.fixed_config(2027),Path("unused"),Path("unused"),torch.device("cpu"),self.raw())

    def test_exact_epoch_resume_keeps_rng_and_marker(self):
        config,signature,provenance,last=fixtures.AugMixSourceAuditTests().resume_fixture()
        marker=protocol.research_provenance("fp32",config,protocol.CONFIG_SHA)
        last["schema"]=protocol.SCHEMA
        last["provenance"]["research_training"]=marker
        provenance["research_training"]=copy.deepcopy(marker)
        rng_before={k:copy.deepcopy(last[k]) for k in ("python_rng","numpy_rng","torch_rng","cuda_rng","augmix_rng","sampler_rng","loader_rng")}
        with protocol.native_training_context(config,protocol.CONFIG_SHA):
            engine.validate_resume(last,config,signature,provenance)
        fixtures.assert_state_equal(self,rng_before,{k:last[k] for k in rng_before})

    def test_continuous_and_saved_boundary_replay_model_sgd_ema_rng_and_counters(self):
        initial=self.tiny()
        # Dropout makes exact Torch RNG replay observable without official arrays.
        initial.insert(3,torch.nn.Dropout(.2))
        prepared=(torch.randn(6,3,32,32),torch.tensor([1,2]))
        config=engine.fixed_config();device=torch.device("cpu")
        rng=np.random.Generator(np.random.PCG64(2026));sampler,loader=engine.pipeline_generators()
        start=engine.capture_state_rng(rng,device,sampler,loader)
        def setup():
            model=copy.deepcopy(initial);ema=copy.deepcopy(model).eval()
            optimizer=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9,weight_decay=.0005,nesterov=True)
            return model,ema,optimizer,torch.amp.GradScaler("cpu",enabled=False)
        continuous,ema_a,opt_a,scale_a=setup()
        engine.restore_state_rng(start,rng,device,sampler,loader)
        with protocol.native_training_context(config,protocol.CONFIG_SHA,strict=False) as proof_a:
            for index in range(4):
                rng.random();torch.randperm(8,generator=sampler);torch.rand(1,generator=loader)
                engine.train_step(continuous,opt_a,scale_a,prepared,device,torch.float32,ema=ema_a,successful_updates=index)
        final_rng=engine.capture_state_rng(rng,device,sampler,loader)
        interrupted,ema_b,opt_b,scale_b=setup()
        engine.restore_state_rng(start,rng,device,sampler,loader)
        with protocol.native_training_context(config,protocol.CONFIG_SHA,strict=False) as first_proof:
            for index in range(2):
                rng.random();torch.randperm(8,generator=sampler);torch.rand(1,generator=loader)
                engine.train_step(interrupted,opt_b,scale_b,prepared,device,torch.float32,ema=ema_b,successful_updates=index)
        saved={"model":copy.deepcopy(interrupted.state_dict()),"ema":copy.deepcopy(ema_b.state_dict()),
            "optimizer":copy.deepcopy(opt_b.state_dict()),"scaler":scale_b.state_dict(),"proof":copy.deepcopy(first_proof),
            "rng":engine.capture_state_rng(rng,device,sampler,loader)}
        resumed,ema_c,opt_c,scale_c=setup()
        resumed.load_state_dict(saved["model"]);ema_c.load_state_dict(saved["ema"])
        opt_c.load_state_dict(saved["optimizer"]);scale_c.load_state_dict(saved["scaler"])
        engine.restore_state_rng(saved["rng"],rng,device,sampler,loader)
        with protocol.native_training_context(config,protocol.CONFIG_SHA,strict=False,initial_proof=saved["proof"]) as proof_c:
            for index in range(2,4):
                rng.random();torch.randperm(8,generator=sampler);torch.rand(1,generator=loader)
                engine.train_step(resumed,opt_c,scale_c,prepared,device,torch.float32,ema=ema_c,successful_updates=index)
        for left,right in ((continuous.state_dict(),resumed.state_dict()),(ema_a.state_dict(),ema_c.state_dict()),
            (opt_a.state_dict(),opt_c.state_dict()),(final_rng,engine.capture_state_rng(rng,device,sampler,loader)),(proof_a,proof_c)):
            fixtures.assert_state_equal(self,left,right)

    def test_checkpoint_and_metrics_proof_snapshots_are_independent_of_live_counters(self):
        config=engine.fixed_config();written={};saved={}
        with patch.object(engine,"write_json",side_effect=lambda path,doc:written.update({str(path):doc})), \
            patch.object(engine,"save_checkpoint",side_effect=lambda path,doc:saved.update({str(path):doc})):
            with protocol.native_training_context(config,protocol.CONFIG_SHA,strict=False) as proof:
                marker=engine.research_provenance("fp32",config,protocol.CONFIG_SHA)
                engine.save_checkpoint("best.pt",{"schema":engine.SCHEMA,"provenance":{"research_training":marker}})
                engine.write_json("metrics.json",{"schema":engine.SCHEMA,"research_training":marker})
                proof["attempted_updates"]=99
                self.assertEqual(saved["best.pt"]["nativebn_execution_proof"]["attempted_updates"],0)
                self.assertEqual(written["metrics.json"]["nativebn_execution_proof"]["attempted_updates"],0)
                self.assertNotIn("attempted_updates",marker)

    def envelope(self):
        report,config,_,_=fixtures.AugMixPreflightTests().fixture()
        report["input_config_sha256"]=protocol.CONFIG_SHA
        report["research_training"]=protocol.research_provenance("fp16",config,protocol.CONFIG_SHA)
        result={"schema":protocol.PREFLIGHT_SCHEMA,"status":"complete","source_sha256":protocol.bound_sources(),
            "config_file_sha256":protocol.CONFIG_SHA,"actual_backend_diagnostic_sha256":protocol.DIAGNOSTIC_SHA,
            "engine_result":report,"engine_result_sha256":hashlib.sha256(json.dumps(report,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
            "execution_proof":self.proof(24,23,1)}
        return result,config

    def test_envelope_requires_new_marker_actual_fullpipeline_and_complete24proof(self):
        envelope,config=self.envelope()
        report=protocol.validate_preflight(envelope,config,protocol.CONFIG_SHA)
        with self.assertRaises(ValueError):engine.validate_preflight(report,config,protocol.CONFIG_SHA)
        changes=[lambda e:e["execution_proof"].update(skipped_probe_preserved_steps=0),
            lambda e:e["engine_result"].update(synthetic_only=True),lambda e:e["engine_result"].update(model_batch_size=128),
            lambda e:e.update(actual_backend_diagnostic_sha256="0"*64),lambda e:e["engine_result"].update(median_step_seconds=.01),
            lambda e:e["execution_proof"].update(observed_training_bn_nodes=["CudnnBatchNormBackward0"])]
        for change in changes:
            altered=copy.deepcopy(envelope);change(altered)
            with self.assertRaises(ValueError):protocol.validate_preflight(altered,config,protocol.CONFIG_SHA)

    def policy(self):return {"concurrent_local_gpu_jobs":1,"local_gpu_seconds_per_utc_day":21600,"local_job_timeout_seconds":10800,"stops_at":"2026-11-05T05:00:00+00:00"}

    def test_shared_budget_exact_margin_hardstop_and_invalid_inputs(self):
        current=datetime(2026,11,1,tzinfo=timezone.utc);policy=self.policy()
        self.assertEqual(supervisor.admission_budget(policy,current,0),10800)
        self.assertEqual(supervisor.admission_budget(policy,current,21000),600)
        stop=datetime.fromisoformat(policy["stops_at"])
        self.assertEqual(supervisor.admission_budget(policy,stop-timedelta(seconds=9),0),9)
        with self.assertRaises(ValueError):supervisor.admission_budget(policy,current,0,{"median_step_seconds":.4})
        with self.assertRaises(ValueError):supervisor.admission_budget(policy,current,21600)
        with self.assertRaises(ValueError):supervisor.admission_budget(policy,stop,0)
        for value in (True,-1,float("nan"),float("inf")):
            with self.assertRaises(ValueError):supervisor.admission_budget(policy,current,value)
        for field,value in (("concurrent_local_gpu_jobs",2),("local_job_timeout_seconds",20000),("local_gpu_seconds_per_utc_day",99999)):
            with self.assertRaises(ValueError):supervisor.admission_budget({**policy,field:value},current,0)

    def test_help_only_and_bare_worker_rejected_before_Torch_or_data(self):
        with redirect_stdout(io.StringIO()),patch.object(supervisor,"parent") as parent,patch.object(supervisor,"worker") as worker:
            supervisor.main([]);parent.assert_not_called();worker.assert_not_called()
        with self.assertRaises(ValueError):supervisor.main(["--_worker","."])
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):supervisor.worker(Path(folder),"0"*64)
        with self.assertRaisesRegex(ValueError,"shared GPU lease"):
            audit.main(["--checkpoint","unused","--output","unused","--baseline","unused"])

    def contract(self,root):
        folder=root/"results/research/synthetic_guard";folder.mkdir(parents=True)
        config=root/supervisor.CONFIG;config.parent.mkdir(parents=True);config.write_bytes(self.raw())
        binding={str(config.resolve()):protocol.CONFIG_SHA}
        manifest={"source_sha256":{"source":"same"},"config_sha256":protocol.CONFIG_SHA,
            "diagnostic_sha256":protocol.DIAGNOSTIC_SHA,"input_sha256":binding,"mode":"preflight"}
        (folder/"nativebn_inputs.json").write_text(json.dumps(manifest))
        digest=supervisor.sha256(folder/"nativebn_inputs.json")
        now=datetime.now(timezone.utc)
        parent=SimpleNamespace(pid=202,create_time=lambda:9.)
        worker=SimpleNamespace(pid=101,create_time=lambda:10.)
        receipt={"schema":supervisor.SCHEMA,"mode":"preflight","status":"benchmarking","source_sha256":manifest["source_sha256"],
            "input_sha256":binding,
            "input_manifest_sha256":digest,"started_at":now.isoformat(),"reserved_seconds":120.,"deadline_unix":now.timestamp()+120,
            "pid":202,"process_created_at":9.,"child_pid":101,"child_created_at":10.}
        (folder/"supervisor.json").write_text(json.dumps(receipt))
        lease=root/"results/research/local_lease.json";lease.write_text(json.dumps({"pid":202,"process_created_at":9.}))
        path=folder/("process_tree_"+"a"*32);path.mkdir()
        from research.profile_augmix import PINNED_SOURCES
        spec={"schema":"owned_python_process_tree_v1","nonce":"a"*32,"host_pid":202,"host_created_at":9.,"cwd":str(root),
            "script":str(Path(supervisor.__file__).resolve()),"script_sha256":supervisor.sha256(supervisor.__file__),
            "guard_source_sha256":PINNED_SOURCES["research/process_tree_guard.py"],
            "arguments":["--_worker",str(folder),"--_manifest-sha",digest],"deadline_unix":receipt["deadline_unix"],
            "job_name":"Local\\SaberAugMix_"+"a"*32 if supervisor.os.name=="nt" else None}
        ready={"schema":spec["schema"],"nonce":spec["nonce"],"pid":101,"created_at":10.,"script_sha256":spec["script_sha256"],
            "method":"windows_job_v1" if supervisor.os.name=="nt" else "posix_session_v1"}
        (path/"spec.json").write_text(json.dumps(spec));(path/"ready.json").write_text(json.dumps(ready))
        worker.cmdline=lambda:[supervisor.sys.executable,str(root/"research/process_tree_guard.py"),"--bootstrap",str(path/"spec.json")]
        return folder,digest,worker,parent,lease,path,spec

    def test_worker_live_guard_contract_and_canonical_lease_rejections(self):
        for change in ("none","missing_lease","foreign_lease","unknown_lease","nonce","source","ready","command","long_budget","membership"):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
                root=Path(temp);folder,digest,worker,parent,lease,path,spec=self.contract(root)
                original=supervisor.sha256
                stack.enter_context(patch.object(supervisor,"ROOT",root))
                stack.enter_context(patch.object(supervisor,"bound_sources",return_value={"source":"same"}))
                stack.enter_context(patch.object(supervisor,"sha256",side_effect=lambda p:spec["guard_source_sha256"] if Path(p)==root/"research/process_tree_guard.py" else original(p)))
                stack.enter_context(patch.object(supervisor.psutil,"Process",side_effect=lambda pid=None:worker if pid is None else parent))
                stack.enter_context(patch.object(supervisor.os,"getppid",return_value=202))
                boundary=stack.enter_context(patch.object(supervisor,"verify_os_boundary"))
                if change=="missing_lease":lease.unlink()
                elif change=="foreign_lease":lease.write_text(json.dumps({"pid":999,"process_created_at":9.}))
                elif change=="unknown_lease":lease.write_text(json.dumps({"pid":202,"process_created_at":9.,"cleanup_requires_reconciliation":True}))
                elif change=="nonce":
                    ready=json.loads((path/"ready.json").read_text());ready["nonce"]="b"*32;(path/"ready.json").write_text(json.dumps(ready))
                elif change=="source":spec["script_sha256"]="0"*64;(path/"spec.json").write_text(json.dumps(spec))
                elif change=="ready":(path/"ready.json").unlink()
                elif change=="command":worker.cmdline=lambda:[supervisor.sys.executable,str(supervisor.__file__)]
                elif change=="long_budget":
                    receipt=json.loads((folder/"supervisor.json").read_text());receipt["reserved_seconds"]=10800;(folder/"supervisor.json").write_text(json.dumps(receipt))
                elif change=="membership":boundary.side_effect=ValueError("no actual OS membership")
                if change=="none":
                    supervisor.validate_worker(folder,digest);boundary.assert_called_once()
                else:
                    with self.assertRaises((ValueError,FileNotFoundError)):supervisor.validate_worker(folder,digest)
                    if change!="membership":boundary.assert_not_called()

    def test_complete_v3_preserves_gates_and_rejects_partial_policy_or_case(self):
        def result():
            from research.benchmark import CASES,POLICIES
            return {"schema":"official_development_stress_v3","status":"complete","examples":5000,"bn_recalibrated":False,
                "stress_cases":list(CASES),"policies":{policy:{"cases":{case:{"examples":5000,"input_pixel_sha256":"a"*64} for case in CASES}} for policy in POLICIES},
                "fixed_primary_policy":"center80_shift20","promotion":{"center80_shift20":{"passed":False}},"research_training_integrity":{}}
        for change in ("none","partial","policy","case","pixels"):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as temp:
                output=Path(temp)/"benchmark.json";document=result()
                if change=="partial":document["examples"]=100
                elif change=="policy":document["policies"].pop("plain")
                elif change=="case":document["policies"]["flip"]["cases"]["clean"]["examples"]=4999
                elif change=="pixels":document["policies"]["flip"]["cases"]["clean"]["input_pixel_sha256"]="bad"
                with patch.object(old_audit,"main",side_effect=lambda:output.write_text(json.dumps(document))):
                    if change=="none":
                        actual=audit.benchmark(Path("synthetic"),output,Path("baseline"),Path("unused"),"cpu")
                        self.assertFalse(actual["promotion"]["center80_shift20"]["passed"])
                    else:
                        with self.assertRaises(ValueError):audit.benchmark(Path("synthetic"),output,Path("baseline"),Path("unused"),"cpu")

    def parent_fixture(self,stack,root,guarded,used=100):
        from research import run_local,process_tree_guard
        (root/"research/configs").mkdir(parents=True);(root/"results/research").mkdir(parents=True)
        (root/supervisor.CONFIG).write_bytes(self.raw())
        policy={**self.policy(),"baseline_benchmark":"baseline.json"}
        (root/"research/campaign.json").write_text(json.dumps(policy));(root/"baseline.json").write_text("synthetic baseline")
        data=root/"data";data.mkdir();(data/"pixels").write_bytes(b"synthetic placeholder, never loaded")
        (data/"training_manifest.json").write_text(json.dumps({"files_sha256":{"pixels":"synthetic fixture"}}))
        source={"synthetic":"a"*64}
        stack.enter_context(patch.object(supervisor,"ROOT",root))
        stack.enter_context(patch.object(supervisor,"bound_sources",return_value=source))
        stack.enter_context(patch.object(run_local,"now",return_value=datetime(2026,11,1,12,tzinfo=timezone.utc)))
        stack.enter_context(patch.object(run_local,"used_seconds",side_effect=used if isinstance(used,list) else None,return_value=used if not isinstance(used,list) else None))
        guard=stack.enter_context(patch.object(process_tree_guard,"run_guarded",side_effect=guarded))
        stack.enter_context(patch.object(protocol,"validate_preflight",return_value={"synthetic_fixture":True}))
        args=SimpleNamespace(run="synthetic_nativebn",mode="preflight",resume=False,preflight=None,checkpoint=None,data_dir=data)
        binding={str(path.resolve()):supervisor.sha256(path) for path in (root/supervisor.CONFIG,root/"baseline.json",data/"training_manifest.json",data/"pixels")}
        return args,guard,source,binding

    def test_parent_preflight_guard_bounded_and_verified_cleanup_releases_shared_lease(self):
        def guarded(command,log,seconds,receipt_path,receipt,cwd):
            self.assertLessEqual(seconds,180);self.assertIn("--_worker",command)
            receipt["process_tree_cleanup_verified"]=True
            (receipt_path.parent/"nativebn_preflight.json").write_text(json.dumps({"status":"complete"}))
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            root=Path(temp);args,guard,source,binding=self.parent_fixture(stack,root,guarded)
            supervisor.parent(args)
            guard.assert_called_once();self.assertFalse((root/"results/research/local_lease.json").exists())
            receipt=json.loads((root/"results/research"/args.run/"supervisor.json").read_text())
            self.assertEqual(receipt["status"],"complete");self.assertFalse(receipt["submitted"])

    def test_parent_unknown_cleanup_retains_locked_canonical_lease(self):
        def guarded(command,log,seconds,receipt_path,receipt,cwd):
            receipt["process_tree_cleanup_verified"]=False;raise RuntimeError("unknown tree")
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            root=Path(temp);args,guard,source,binding=self.parent_fixture(stack,root,guarded)
            with self.assertRaisesRegex(RuntimeError,"unknown tree"):supervisor.parent(args)
            lease=json.loads((root/"results/research/local_lease.json").read_text())
            self.assertTrue(lease["cleanup_requires_reconciliation"])
            self.assertFalse(lease["process_tree_cleanup_verified"])

    def test_parent_existing_lease_refuses_without_touching_foreign_owner(self):
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            root=Path(temp);args,guard,source,binding=self.parent_fixture(stack,root,lambda *a,**k:None)
            lease=root/"results/research/local_lease.json";lease.write_bytes(b'{"pid":999}')
            with self.assertRaisesRegex(ValueError,"free local lease"):supervisor.parent(args)
            guard.assert_not_called();self.assertEqual(lease.read_bytes(),b'{"pid":999}')

    def test_resume_insufficient_budget_preserves_original_receipt_before_and_after_archive(self):
        for usage in ([21600],[0,21600]):
            with self.subTest(usage=usage),tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
                root=Path(temp);args,guard,source,binding=self.parent_fixture(stack,root,lambda *a,**k:None,used=usage)
                args.mode="train";args.resume=True;args.preflight=root/"unused-native-pref.json"
                stack.enter_context(patch("runtime.environment",return_value={"packages":{"torch":"synthetic"}}))
                # Supply matching synthetic environment only; no hardware query.
                stack.enter_context(patch.object(supervisor,"checked_preflight",return_value=({"median_step_seconds":.06,"environment":{"packages":{"torch":"synthetic"}}},{})))
                folder=root/"results/research"/args.run;folder.mkdir();(folder/"last.pt").write_bytes(b"synthetic checkpoint, never loaded")
                previous={"schema":supervisor.SCHEMA,"mode":"train","backend":"local","run":args.run,"status":"failed",
                    "source_sha256":source,"input_sha256":binding,"config_sha256":protocol.CONFIG_SHA,"pid":2147483647,
                    "process_created_at":0.,"started_at":"2026-11-01T00:00:00+00:00","finished_at":"2026-11-01T01:00:00+00:00",
                    "reserved_seconds":10800,"elapsed_seconds":3600.,"process_tree_cleanup_verified":True}
                receipt=folder/"supervisor.json";receipt.write_text(json.dumps(previous));original=receipt.read_bytes()
                with self.assertRaisesRegex(ValueError,"budget"):supervisor.parent(args)
                guard.assert_not_called();self.assertFalse((root/"results/research/local_lease.json").exists())
                if len(usage)==1:
                    self.assertEqual(receipt.read_bytes(),original);self.assertFalse((folder/"supervisor_raw_1.json").exists())
                else:
                    self.assertEqual((folder/"supervisor_raw_1.json").read_bytes(),original)
                    current=json.loads(receipt.read_text());self.assertEqual(current["previous_attempt"],"supervisor_attempt_1.json")
                    self.assertEqual(current["status"],"failed");self.assertEqual(current["reserved_seconds"],0)


if __name__ == "__main__":unittest.main()
