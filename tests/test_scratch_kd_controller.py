"""Pure synthetic controller/native-receipt contracts; never execute numerics."""
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
from research import run_scratch_kd as controller


class PureContracts(unittest.TestCase):
    def test_bare_help_and_direct_worker_never_import_numerical_modules(self):
        code="import sys;from research import run_scratch_kd as c;c.main([]);assert not {'torch','numpy','PIL','runtime'}&set(sys.modules)"
        subprocess.check_output([sys.executable,'-c',code],cwd=controller.ROOT,text=True)
        with patch.object(controller,'parent') as parent,redirect_stdout(io.StringIO()):controller.main([])
        parent.assert_not_called()
        with tempfile.TemporaryDirectory() as temp,self.assertRaises(ValueError):controller.validate_worker(Path(temp),'a'*64)
        for args in (['--mode','formal'],['--role','control26'],['--run','fresh','--mode','formal']):
            with self.assertRaises(ValueError):controller.main(args)

    def test_all_frozen_source_bytes_and_role_configs(self):
        sources=controller.source_snapshot();self.assertIn('research/run_scratch_kd.py',sources);self.assertIn(controller.PLAN,sources)
        self.assertTrue(set(controller.engine.SOURCE_KEYS)<=set(sources))
        a,b=controller.configuration('control26'),controller.configuration('primary26');self.assertEqual(b,{**a,'kd_alpha':.5})
        for role in ('repeat27','candidate','unknown'):
            with self.assertRaises(ValueError):controller.configuration(role)

    def policy(self):return {'stops_at':controller.HARD_STOP.isoformat(),'concurrent_local_gpu_jobs':1,'local_job_timeout_seconds':10800,'local_gpu_seconds_per_utc_day':21600}

    def test_job_day_hardstop_and_each_roles_actual_measurement(self):
        now=datetime(2026,11,3,tzinfo=timezone.utc);p=self.policy()
        self.assertEqual(controller.budget(p,now,0,'preflight'),180)
        self.assertEqual(controller.budget(p,now,0,'formal',{'median_step_seconds':.1}),10800)
        for mode,used,pref in [('preflight',21500,None),('formal',18000,{'median_step_seconds':.1}),('formal',0,{'median_step_seconds':.15}),('formal',0,None)]:
            with self.assertRaises(ValueError):controller.budget(p,now,used,mode,pref)
        with self.assertRaises(ValueError):controller.budget({**p,'stops_at':(controller.HARD_STOP+timedelta(days=1)).isoformat()},now,0,'preflight')
        with self.assertRaises(ValueError):controller.budget(p,controller.HARD_STOP-timedelta(seconds=179),0,'preflight')
        calls=[]
        account={'used_seconds':lambda base,left,right:calls.append((left,right)) or 7.}
        current=datetime(2026,11,4,0,0,1,tzinfo=timezone.utc)
        self.assertEqual(controller.current_usage(account,Path('unused'),current),7.)
        self.assertEqual(calls[0][0],datetime(2026,11,4,tzinfo=timezone.utc))

    def fixture(self,role='control26'):
        config=controller.configuration(role);ref={'unit':'synthetic metadata only'}
        manifest={'role':role,'source_sha256':{'unit.py':'a'*64},'config_sha256':controller.PINS[controller.ROLES[role]['config']],
            'teacher_reference':ref,'data_identity':controller.c.DATA_IDENTITY}
        env={'python':sys.executable,'python_version':'3.14.3','platform':'unit fixture','device':'cuda','cuda_runtime':'13.0','cudnn':92000,
            'gpu':'NVIDIA GeForce RTX 5060 Laptop GPU','gpu_memory_GiB':7.96,'amp_dtype':'torch.bfloat16','cudnn_benchmark':True,'deterministic_algorithms':False,
            'process_local_cudnn_path_entries_removed':1,'packages':{key:'unit' for key in ('torch','torchvision','numpy','Pillow','matplotlib','nbformat','reportlab')}}
        env['packages']['torch']='2.12.0+cu130';teacher=role=='primary26'
        report={'schema':controller.PREF_SCHEMA,'status':'complete','mode':'preflight','role':role,'seed':2026,'epochs':160,'engineering_only':True,
            'accuracy_evaluated':False,'formal_training_admitted':False,'confirmation_evaluated':False,'hidden_images_loaded':False,'external_data_loaded':False,
            'native_bf16_without_emulation':True,'training_precision':'bf16','parameter_count':4289754,'trainable_parameter_count':4289754,'teacher_parameters':4166298,
            'teacher_state_tensors':191,'teacher_bn_modules':31,'gradient_tensor_count':44,'state_tensor_count':210,'bn_modules':13,
            'master_parameter_dtype':'torch.float32','ema_parameter_dtype':'torch.float32','grad_scaler_enabled':False,'gradient_clipping':False,
            'batch_size':128,'training_pool_examples':40000,'evaluation_examples':0,'warmup_updates':3,'timed_updates':20,'successful_updates':23,'skipped_updates':0,
            'source_examples_seen':2944,'unique_source_examples':2944,'student_forward_passes':23,'teacher_forward_passes':23 if teacher else 0,
            'student_inputs_seen':2944,'teacher_inputs_seen':2944 if teacher else 0,'planned_formal_attempted_updates':50080,'bn_forward_counts':[23]*13,
            'ema_buffer_copies':0,'preflight_state_discarded':True,'source_sha256':manifest['source_sha256'],'config_sha256':manifest['config_sha256'],
            'effective_config_sha256':controller.c.json_sha(config),'input_manifest_sha256':'b'*64,'teacher_reference_sha256':controller.engine.json_sha(ref),
            'data_identity':manifest['data_identity'],'timing_scope':controller.TIMING_SCOPE,'environment':env,'environment_sha256':controller.engine.json_sha(env),
            'step_seconds':[.1]*23,'median_step_seconds':.1,'conservative_pipeline_seconds':controller.guard_seconds(.1),'setup_seconds':4.,'full_step_seconds':sum([.1]*23),
            'peak_gpu_memory_bytes':1024,'ordered_training_indices_sha256':'c'*64,'ordered_official_ids_sha256':'d'*64,'initial_student_state_sha256':'e'*64,
            'steps':[{'updates':i,'gradient_tensor_count':44,'state_tensor_count':210,'source_examples':128,'student_forward_passes':1,'teacher_forward_passes':int(teacher),
                'teacher_inputs':128 if teacher else 0,'bn_counters_after':[i]*13,'teacher_immutable_verified':True,'teacher_eval_no_grad':True,'rollback_snapshots_in_timer':True,
                'logits_dtype':'torch.bfloat16','mixed':True,'loss':1.,'hard_loss':1.,'soft_loss':.1 if teacher else None,'gradient_norm':1.,'parameter_probe_max_delta':.01,
                'ema_probe_max_delta':.01,'lr':.1*i/(5*313)} for i in range(1,24)]}
        return report,manifest

    def test_complete_native_role_proof_and_rollback_work_cannot_be_partial(self):
        for role in controller.ROLES:
            report,manifest=self.fixture(role);controller.validate_preflight_report(report,manifest,'b'*64)
            for field,value in [('state_tensor_count',209),('teacher_state_tensors',190),('successful_updates',22),('grad_scaler_enabled',True),('master_parameter_dtype','torch.bfloat16')]:
                with self.subTest(role=role,field=field),self.assertRaises(ValueError):controller.validate_preflight_report({**report,field:value},manifest,'b'*64)
            for field,value in [('logits_dtype','torch.float32'),('rollback_snapshots_in_timer',False),('teacher_eval_no_grad',False),('gradient_norm',float('nan'))]:
                bad=copy.deepcopy(report);bad['steps'][7][field]=value
                with self.subTest(field=field),self.assertRaises(ValueError):controller.validate_preflight_report(bad,manifest,'b'*64)

    def test_cohort_requires_actual_initialization_input_order_and_native_environment(self):
        a,_=self.fixture('control26');b,_=self.fixture('primary26')
        with patch.object(controller,'validate_preflight_bundle',side_effect=lambda folder,sources,role:(a if role=='control26' else b,[])):
            controller.cohort(controller.ROOT/'results/research/controlpref',controller.ROOT/'results/research/kdpref',{})
            for field in ('initial_student_state_sha256','ordered_training_indices_sha256','ordered_official_ids_sha256'):
                old=b[field];b[field]='f'*64
                with self.assertRaises(ValueError):controller.cohort(controller.ROOT/'results/research/controlpref',controller.ROOT/'results/research/kdpref',{})
                b[field]=old
            b['environment']['cudnn']=91000
            with self.assertRaises(ValueError):controller.cohort(controller.ROOT/'results/research/controlpref',controller.ROOT/'results/research/kdpref',{})

    def test_stage_inputs_add_future_control_evidence_without_polluting_pref_base(self):
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            root=Path(temp).resolve();basefile=root/'base';pref=root/'pref';control=root/'control'
            for path in (basefile,pref,control):path.write_text(path.name)
            binding={str(basefile):controller.sha(basefile)}
            stack.enter_context(patch.object(controller,'base_binding',return_value=(binding,{},{})))
            cohort=stack.enter_context(patch.object(controller,'cohort',return_value={}))
            stack.enter_context(patch.object(controller,'validate_preflight_bundle',return_value=({},[pref])))
            control_check=stack.enter_context(patch.object(controller,'validate_control_bundle',return_value=({'point_and_case_gates_passed':False},[control])))
            stack.enter_context(patch.object(controller,'owned_folder',side_effect=lambda value:Path(value)))
            actual,_,_=controller.stage_binding(root,{},'preflight','control26');self.assertEqual(actual,binding);cohort.assert_not_called()
            with self.assertRaises(ValueError):controller.stage_binding(root,{},'preflight','control26',pref,pref)
            actual,_,_=controller.stage_binding(root,{},'formal','control26',root/'a',root/'b');self.assertIn(str(pref),actual);control_check.assert_not_called()
            with self.assertRaises(ValueError):controller.stage_binding(root,{},'formal','primary26',root/'a',root/'b')
            actual,_,_=controller.stage_binding(root,{},'formal','primary26',root/'a',root/'b',control)
            self.assertIn(str(control),actual);control_check.assert_called_once()
            # False candidate flags on a complete control do not reject the stage.
            self.assertEqual(control_check.return_value[0]['point_and_case_gates_passed'],False)

    def guard_fixture(self,root,folder,digest,receipt):
        directory=folder/('process_tree_'+'a'*32);directory.mkdir()
        spec={'schema':'owned_python_process_tree_v1','nonce':'a'*32,'host_pid':receipt['pid'],'host_created_at':receipt['process_created_at'],
            'cwd':str(root),'script':str(Path(controller.__file__).resolve()),'script_sha256':controller.sha(controller.__file__),
            'guard_source_sha256':controller.PINS['research/process_tree_guard.py'],'arguments':['--_worker',str(folder),'--_manifest-sha',digest],
            'deadline_unix':receipt['deadline_unix']-2,'job_name':'Local\\SaberAugMix_'+'a'*32}
        ready={'schema':spec['schema'],'nonce':spec['nonce'],'pid':receipt['child_pid'],'created_at':receipt['child_created_at'],'script_sha256':spec['script_sha256'],'method':'windows_job_v1'}
        completion={**ready,'exit_code':0};proof={'schema':spec['schema'],'nonce':spec['nonce'],'method':'windows_job_v1','guard_directory':str(directory),
            'guard_source_sha256':spec['guard_source_sha256'],'cleanup_verified':True,'timed_out':False,'child_pid':receipt['child_pid'],'child_created_at':receipt['child_created_at'],
            'child_returncode':0,'completion_authenticated':True,'completion_exit_code':0};receipt['process_tree_guard_results']=[proof]
        def save():
            for name,obj in [('spec.json',spec),('ready.json',ready),('completion.json',completion),('host_result.json',proof)]:controller.write(directory/name,obj)
        save();return proof,spec,ready,completion,save

    def test_closed_native_real_fieldnames_host_identity_and_timestamp(self):
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            root=Path(temp).resolve();folder=root/'results/research/pref';folder.mkdir(parents=True);now=datetime.now(timezone.utc)-timedelta(seconds=10)
            receipt={'pid':11,'process_created_at':1.5,'child_pid':12,'child_created_at':2.5,'started_at':now.isoformat(),'deadline_unix':now.timestamp()+180}
            proof,spec,ready,done,save=self.guard_fixture(root,folder,'b'*64,receipt);stack.enter_context(patch.object(controller,'ROOT',root))
            self.assertEqual(len(controller.completed_guard_paths(folder,receipt,'b'*64)),4)
            for obj,key,value in [(proof,'child_created_at',99.),(spec,'script_sha256','d'*64),(ready,'pid',99),(done,'exit_code',1),(proof,'cleanup_verified',False)]:
                old=obj[key];obj[key]=value;save()
                with self.assertRaises(ValueError):controller.completed_guard_paths(folder,receipt,'b'*64)
                obj[key]=old
            save();controller.write(Path(proof['guard_directory'])/'host_result.json',{**proof,'timed_out':True})
            with self.assertRaises(ValueError):controller.completed_guard_paths(folder,receipt,'b'*64)

    def test_live_guard_auth_uses_real_sha256_fields_before_runtime(self):
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            root=Path(temp).resolve();folder=root/'results/research/pref';folder.mkdir(parents=True)
            (root/'research').mkdir();shutil.copyfile(controller.ROOT/'research/process_tree_guard.py',root/'research/process_tree_guard.py')
            controller.write(folder/'inputs.json',{'mode':'preflight'});digest=controller.sha(folder/'inputs.json');now=datetime.now(timezone.utc)
            receipt={'schema':controller.SCHEMA,'status':'benchmarking','input_manifest_sha256':digest,'pid':11,'process_created_at':1.5,'child_pid':12,'child_created_at':2.5,
                'reserved_seconds':180,'started_at':now.isoformat(),'deadline_unix':now.timestamp()+180}
            proof,spec,ready,done,save=self.guard_fixture(root,folder,digest,receipt)
            if os.name!='nt':spec['job_name']=None;ready['method']='posix_session_v1';save()
            lease={'pid':11,'process_created_at':1.5};controller.write(root/'results/research/local_lease.json',lease);controller.write(folder/'supervisor.json',receipt)
            process=SimpleNamespace(pid=12,create_time=lambda:2.5,cmdline=lambda:[sys.executable,str(root/'research/process_tree_guard.py'),'--bootstrap',str(Path(proof['guard_directory'])/'spec.json')])
            parent=SimpleNamespace(pid=11,create_time=lambda:1.5)
            stack.enter_context(patch.object(controller,'ROOT',root));stack.enter_context(patch.object(controller.psutil,'Process',side_effect=lambda pid=None:process if pid is None else parent))
            stack.enter_context(patch.object(controller.os,'getppid',return_value=11));boundary=stack.enter_context(patch.object(controller.p,'verify_os_boundary'))
            bound=stack.enter_context(patch.object(controller,'validate_live_binding',return_value={'ok':True}))
            self.assertEqual(controller.validate_worker(folder,digest),{'ok':True});boundary.assert_called_once();bound.assert_called_once()
            for obj,key,value,path in [(spec,'script_sha256','e'*64,'spec.json'),(spec,'arguments',['wrong'],'spec.json'),(ready,'created_at',9.,'ready.json')]:
                old=obj[key];obj[key]=value;controller.write(Path(proof['guard_directory'])/path,obj)
                with self.assertRaises(ValueError):controller.validate_worker(folder,digest)
                obj[key]=old;save()
            receipt['input_manifest_sha256']='a'*64;controller.write(folder/'supervisor.json',receipt)
            with self.assertRaises(ValueError):controller.validate_worker(folder,digest)

    def test_engine_math_only_and_initial_fingerprint_inside_setup_charge(self):
        tree=ast.parse(Path(controller.__file__).read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='preflight_worker')
        calls=[n.func.attr for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        for name in ('setup','start_epoch','train_step'):self.assertIn(name,calls)
        imports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
        self.assertFalse(any((n.module if isinstance(n,ast.ImportFrom) else n.names[0].name) in ('runtime','torch','numpy') for n in imports))
        self.assertNotIn('exec(',Path(controller.__file__).read_text())


if __name__=='__main__':unittest.main()
