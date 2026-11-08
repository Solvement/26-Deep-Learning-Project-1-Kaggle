"""Pure contracts locally; actual Torch/state topology checks are Linux-CI only."""
import ast
from contextlib import ExitStack,redirect_stdout
import copy
from datetime import datetime,timedelta,timezone
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from research import preflight_wrn16_300 as preflight
from tests import test_preflight_core_control as old_tests


class PureContracts(unittest.TestCase):
    def test_help_and_direct_worker_do_not_import_runtime_or_torch(self):
        code="import sys;from research import preflight_wrn16_300 as p;p.main([]);assert not {'torch','numpy','PIL','runtime'}&set(sys.modules)"
        subprocess.check_output([sys.executable,'-c',code],cwd=preflight.ROOT,text=True)
        with patch.object(preflight,'parent') as parent,patch.object(preflight,'worker') as worker,redirect_stdout(io.StringIO()):preflight.main([])
        parent.assert_not_called();worker.assert_not_called()
        with tempfile.TemporaryDirectory() as temp,self.assertRaises(ValueError):preflight.validate_worker(Path(temp),'a'*64)

    def test_explicit_seed_and_repeat_freeze_boundary(self):
        for args in (['--run','fresh'],['--seed','2026'],['--run','fresh','--seed','2027'],['--run','fresh','--seed','2026','--frozen-manifest','x.json']):
            with self.assertRaises(ValueError):preflight.main(args)
        for seed,extra in [(2026,[]),(2027,['--frozen-manifest','freeze.json'])]:
            with patch.object(preflight,'parent') as parent:
                preflight.main(['--run','fresh','--seed',str(seed),*extra]);self.assertEqual(parent.call_args.args[0].recipe,preflight.NAME)
        self.assertEqual(preflight.repeat_plan(2026,None,{}),([],None))
        with self.assertRaises(ValueError):preflight.repeat_plan(2027,None,{})

    def test_exact_raw_schedule_only_change_and_frozen_source_closure(self):
        primary,repeat=preflight.configuration(2026),preflight.configuration(2027)
        self.assertEqual(repeat,{**primary,'seed':2027});self.assertEqual(primary,{**preflight.control.configuration(2026),'epochs':300})
        self.assertEqual((primary['depth'],primary['widen_factor'],primary['epochs'],primary['batch_size']),(16,5,300,128))
        sources=preflight.source_snapshot();self.assertIn('research/preflight_wrn16_300.py',sources)
        self.assertIn(preflight.PLAN,sources);self.assertIs(preflight.core_step,preflight.control.core_step)
        for bad in (True,2028):
            with self.assertRaises(ValueError):preflight.configuration(bad)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name in ('train.py','research/configs/wrn16_5.json',*(v[0] for v in preflight.CONFIGS.values())):
                (root/name).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(preflight.ROOT/name,root/name)
            p=root/preflight.CONFIGS[2027][0];p.write_bytes(p.read_bytes()+b' ')
            with self.assertRaises(ValueError):preflight.configuration(2026,root)

    def fixture(self):
        report,manifest=old_tests.PureContracts().report_fixture();manifest.update(recipe=preflight.NAME,effective_config=preflight.configuration(2026),config_sha256=preflight.CONFIGS[2026][1],
            comparison_sha256={k:v[1] for k,v in preflight.COMPARISONS.items()},frozen_manifest_sha256=None)
        manifest['effective_config_sha256']=preflight.json_sha(manifest['effective_config'])
        report.update(schema=preflight.REPORT_SCHEMA,recipe=preflight.NAME,epochs=300,planned_formal_attempted_updates=93900,
            config_sha256=manifest['config_sha256'],effective_config_sha256=manifest['effective_config_sha256'],gradient_tensor_count=44,state_tensor_count=210,
            comparison_sha256=manifest['comparison_sha256'],frozen_manifest_sha256=None,conservative_pipeline_seconds=1.5*.1*313*300+300)
        for step in report['steps']:step.update(gradient_tensor_count=44,state_tensor_count=210)
        report['environment']={'device':'cuda','gpu':'NVIDIA GeForce RTX 5060 Laptop GPU','amp_dtype':'torch.bfloat16','python_version':'3.14.3','python':sys.executable,
            'platform':'synthetic unit fixture','cuda_runtime':'13.0','cudnn':92000,'cudnn_benchmark':True,'deterministic_algorithms':False,
            'gpu_memory_GiB':8.,'process_local_cudnn_path_entries_removed':0,'packages':{k:'unit' for k in ('torch','torchvision','numpy','Pillow','matplotlib','nbformat','reportlab')}}
        report['environment']['packages']['torch']='2.12.0+cu130';report['environment_sha256']=preflight.json_sha(report['environment'])
        return report,manifest

    def test_all23_steps_full_counts_and_300_guard_are_required(self):
        report,manifest=self.fixture();preflight.validate_report(report,manifest,'b'*64)
        for key,value in [('epochs',160),('planned_formal_attempted_updates',50080),('gradient_tensor_count',98),('state_tensor_count',480),('bn_forward_counts',[24]*13),
            ('training_precision','fp16'),('formal_training_admitted',True),('conservative_pipeline_seconds',1.5*.1*313*160+300),('comparison_sha256',{})]:
            with self.subTest(key=key),self.assertRaises(ValueError):preflight.validate_report({**report,key:value},manifest,'b'*64)
        for key,value in [('state_tensor_count',209),('gradient_tensor_count',43),('gradient_norm',float('nan')),('parameter_probe_max_delta',0),('ema_probe_max_delta',0),
                          ('updates',True),('lr',.1),('bn_single_forward',False)]:
            bad=copy.deepcopy(report);bad['steps'][4][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):preflight.validate_report(bad,manifest,'b'*64)

    def test_resource_guard_uses_300_and_actual_day_and_hardstop(self):
        now=datetime(2026,11,3,tzinfo=timezone.utc);policy={'concurrent_local_gpu_jobs':1,'local_gpu_seconds_per_utc_day':21600,'local_job_timeout_seconds':10800,'stops_at':preflight.HARD_STOP.isoformat()}
        report={'median_step_seconds':.05,'conservative_pipeline_seconds':1.5*.05*313*300+300}
        self.assertAlmostEqual(preflight.formal_guard_seconds(report,policy,now,5116),7342.5)
        for used in [15000,-1,float('nan')]:
            with self.assertRaises(ValueError):preflight.formal_guard_seconds(report,policy,now,used)
        with self.assertRaises(ValueError):preflight.formal_guard_seconds({**report,'median_step_seconds':.08,'conservative_pipeline_seconds':1.5*.08*313*300+300},policy,now,0)
        with self.assertRaises(ValueError):preflight.formal_guard_seconds(report,{**policy,'stops_at':(now+timedelta(seconds=7000)).isoformat()},now,0)
        self.assertEqual(preflight.bounded_seconds(policy,now,0),180)
        with self.assertRaises(ValueError):preflight.bounded_seconds(policy,now,21500)
        with self.assertRaises(ValueError):preflight.bounded_seconds({**policy,'stops_at':(preflight.HARD_STOP+timedelta(days=1)).isoformat()},now,0)

    def test_midnight_uses_current_day_and_prorates_live_elapsed(self):
        current=datetime(2026,11,4,0,0,20,tzinfo=timezone.utc);started=current-timedelta(seconds=40)
        calls=[]
        def used(base,left,right,exclude=None):calls.append((left,right,exclude));return 100.
        value=preflight.day_usage({'used_seconds':used},Path('unused'),current,exclude=Path('own'),started=started,elapsed=44.)
        self.assertEqual(value,122.);self.assertEqual(calls[0][0],datetime(2026,11,4,tzinfo=timezone.utc))
        with self.assertRaises(ValueError):preflight.day_usage({'used_seconds':used},Path('unused'),current,started=current,elapsed=44.)

    def test_full_native_environment_and_nonbool_finite_times_required(self):
        report,manifest=self.fixture()
        for key,value in [('cudnn',False),('cuda_runtime',None),('cudnn_benchmark',1),('gpu','NVIDIA GeForce RTX 5060'),('packages',{}),('process_local_cudnn_path_entries_removed',True)]:
            bad=copy.deepcopy(report);bad['environment'][key]=value;bad['environment_sha256']=preflight.json_sha(bad['environment'])
            with self.subTest(key=key),self.assertRaises(ValueError):preflight.validate_report(bad,manifest,'b'*64)
        for key in ('median_step_seconds','full_step_seconds','setup_seconds','conservative_pipeline_seconds'):
            with self.subTest(key=key),self.assertRaises(ValueError):preflight.validate_report({**report,key:True},manifest,'b'*64)

    def test_atomic_unknown_lease_and_cleanup_never_released(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'local_lease.json';path.write_text('{"status":"unknown"}');raw=path.read_bytes()
            with self.assertRaises(FileExistsError):preflight.claim_lease(path)
            self.assertEqual(path.read_bytes(),raw)
        for owner in [{'status':'unknown'},{'cleanup_requires_reconciliation':True},{'process_tree_cleanup_verified':False}]:
            self.assertFalse(preflight.lease_can_release(owner,{'process_tree_cleanup_verified':True}))

    def test_worker_authentication_precedes_runtime_and_reuses_unchanged_math(self):
        tree=ast.parse(Path(preflight.__file__).read_text());worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
        auth=next(i for i,n in enumerate(worker.body) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and getattr(n.value.func,'id',None)=='validate_worker')
        runtime=next(i for i,n in enumerate(worker.body) if isinstance(n,ast.Import) and n.names[0].name=='runtime')
        torch=next(i for i,n in enumerate(worker.body) if isinstance(n,ast.Import) and n.names[0].name=='torch')
        self.assertLess(auth,runtime);self.assertLess(runtime,torch)
        self.assertFalse(any(isinstance(n,ast.FunctionDef) and n.name=='core_step' for n in tree.body))
        loop=next(n for n in worker.body if isinstance(n,ast.For))
        calls=[getattr(n.func,'id',None) for n in ast.walk(loop) if isinstance(n,ast.Call)]
        self.assertIn('core_step',calls)

    def test_repeat_plan_requires_exact_predeclared_source_and_config(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'frozen.json';sources={'unit.py':'a'*64};now=datetime.now(timezone.utc)
            document={'schema':'research_candidate_freeze_v2','status':'frozen','config':preflight.configuration(2026),'parameter_count':4289754,'policy':'center80_shift20',
                'confirmation_evaluated':False,'repeat_prediction_authorized':False,'full_refit_supported':False,'frozen_at':(now-timedelta(seconds=1)).isoformat(),
                'primary_decision':{'seed':2026,'planned_repeat_seed':2027,'planned_repeat_config_sha256':'a'*64,'planned_repeat_source_sha256':sources,'decided_at':(now-timedelta(seconds=2)).isoformat()}}
            p.write_text(json.dumps(document))
            with self.assertRaises(ValueError):preflight.repeat_plan(2027,p,sources)
            document['primary_decision']['planned_repeat_config_sha256']=preflight.CONFIGS[2027][1];document['primary_decision']['planned_repeat_source_sha256']={}
            p.write_text(json.dumps(document))
            with self.assertRaises(ValueError):preflight.repeat_plan(2027,p,sources)


    def test_guarded_worker_checks_lease_nonce_bootstrap_source_deadline_and_os_membership(self):
        from research import profile_augmix
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root/"results/research/fresh"
            directory = folder/("process_tree_"+"a"*32)
            directory.mkdir(parents=True)
            guard = root/"research/process_tree_guard.py"
            guard.parent.mkdir()
            shutil.copyfile(preflight.ROOT/"research/process_tree_guard.py",guard)
            (root/"train.py").write_text("unit fixture")
            config_path = root/preflight.CONFIGS[2026][0]
            config_path.parent.mkdir(parents=True)
            config_path.write_text("unit fixture")
            data_dir = root/"data"
            data_dir.mkdir()
            data_manifest = data_dir/"training_manifest.json"
            data_manifest.write_text("unit fixture")
            sources = {"train.py":preflight.sha256(root/"train.py")}
            config = preflight.configuration(2026)
            inputs = {str(p.resolve()):preflight.sha256(p) for p in (data_manifest,config_path,root/"train.py")}
            manifest = {"schema":preflight.INPUT_SCHEMA,"recipe":preflight.NAME,"seed":2026,"run":"fresh","data_dir":str(data_dir),
                "source_sha256":sources,"source_git_commit":"a"*40,"config_sha256":preflight.CONFIGS[2026][1],
                "effective_config":config,"effective_config_sha256":preflight.json_sha(config),
                "data_identity":preflight.DATA_IDENTITY,"input_sha256":inputs,"comparison_sha256":{},"frozen_manifest_path":None,"frozen_manifest_sha256":None}
            preflight.write(folder/"inputs.json",manifest)
            digest = preflight.sha256(folder/"inputs.json")
            now = datetime.now(timezone.utc)
            receipt = {"schema":preflight.SCHEMA,"status":"benchmarking","backend":"local","mode":"preflight","precision":"bf16","run":"fresh","recipe":preflight.NAME,
                "started_at":now.isoformat(),"reserved_seconds":180,"deadline_unix":now.timestamp()+180,
                "pid":11,"process_created_at":1.5,"child_pid":12,"child_created_at":2.5,
                "input_manifest_sha256":digest,"source_sha256":sources,"config_sha256":manifest["config_sha256"],"input_sha256":inputs,
                "confirmation_evaluated":False,"submitted":False}
            lease = {"pid":11,"process_created_at":1.5}
            spec = {"schema":"owned_python_process_tree_v1","nonce":"a"*32,"host_pid":11,"host_created_at":1.5,
                "cwd":str(root),"script":str(Path(preflight.__file__).resolve()),"script_sha256":preflight.sha256(preflight.__file__),
                "guard_source_sha256":preflight.PINS["research/process_tree_guard.py"],"arguments":["--_worker",str(folder),"--_manifest-sha",digest],
                "deadline_unix":now.timestamp()+170,"job_name":"Local\\SaberAugMix_"+"a"*32 if os.name == "nt" else None}
            ready = {"schema":spec["schema"],"nonce":spec["nonce"],"pid":12,"created_at":2.5,"script_sha256":spec["script_sha256"],
                "method":"windows_job_v1" if os.name == "nt" else "posix_session_v1"}
            process = SimpleNamespace(pid=12,create_time=lambda:2.5,cmdline=lambda:[sys.executable,str(guard),"--bootstrap",str(directory/"spec.json")])
            parent = SimpleNamespace(pid=11,create_time=lambda:1.5)
            with ExitStack() as stack:
                stack.enter_context(patch.object(preflight,"ROOT",root))
                stack.enter_context(patch.object(preflight,"source_snapshot",return_value=sources))
                stack.enter_context(patch.object(preflight,"configuration",return_value=config))
                stack.enter_context(patch.object(preflight,"official_inputs",return_value=([data_manifest],preflight.DATA_IDENTITY)))
                stack.enter_context(patch.object(preflight,"comparison_inputs",return_value=([],{})))
                stack.enter_context(patch.object(preflight,"repeat_plan",return_value=([],None)))
                stack.enter_context(patch.object(preflight.os,"getppid",return_value=11))
                stack.enter_context(patch.object(preflight.psutil,"Process",side_effect=lambda pid=None:process if pid is None else parent))
                boundary = stack.enter_context(patch.object(profile_augmix,"verify_os_boundary"))
                def save():
                    for path,document in ((folder/"supervisor.json",receipt),(root/"results/research/local_lease.json",lease),
                            (directory/"spec.json",spec),(directory/"ready.json",ready)):
                        preflight.write(path,document)
                save()
                self.assertEqual(preflight.validate_worker(folder,digest),manifest)
                boundary.assert_called_once()
                for target,key,value in ((lease,"cleanup_requires_reconciliation",True),(lease,"pid",99),(spec,"nonce","b"*32),
                        (spec,"script_sha256","b"*64),(spec,"deadline_unix",now.timestamp()+200),(ready,"pid",99),
                        (receipt,"child_created_at",99),(receipt,"reserved_seconds",181)):
                    old = target.get(key)
                    target[key] = value
                    save()
                    with self.assertRaises(ValueError): preflight.validate_worker(folder,digest)
                    if old is None: target.pop(key)
                    else: target[key] = old
                save()
                boundary.side_effect = ValueError("not a real OS member")
                with self.assertRaises(ValueError): preflight.validate_worker(folder,digest)


class TorchTopologyContracts(unittest.TestCase):
    def test_actual_wrn16_has44_gradients_and210_complete_state_tensors(self):
        import runtime
        import torch
        import copy
        from model import build_model,count_parameters
        torch.set_num_threads(1);config=preflight.configuration(2026)
        model=build_model({k:config[k] for k in ('depth','widen_factor','dropout','num_classes')})
        self.assertEqual(count_parameters(model),4289754);self.assertEqual(count_parameters(model,True),4289754)
        self.assertEqual(len(list(model.parameters())),44);self.assertEqual(len(model.state_dict()),83)
        self.assertEqual(sum(isinstance(m,torch.nn.BatchNorm2d) for m in model.modules()),13)
        ema=copy.deepcopy(model).eval().requires_grad_(False)
        optimizer=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9,nesterov=True)
        loss=torch.nn.functional.cross_entropy(model(torch.randn(2,3,32,32)),torch.tensor([0,1]));loss.backward();optimizer.step()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        states=[*model.state_dict().values(),*ema.state_dict().values(),*(v for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v))]
        self.assertEqual(len(states),210)


if __name__=='__main__':unittest.main()
