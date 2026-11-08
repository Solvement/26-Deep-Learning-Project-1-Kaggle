"""Pure delivery contracts plus one CPU-only tensor-loader boundary test for CI."""
import ast
from contextlib import ExitStack,redirect_stdout
import copy
from datetime import datetime,timedelta,timezone
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from research import trivialaugment_delivery as delivery
from scripts import submit_verified


class DeliveryPureContracts(unittest.TestCase):
    def test_fresh_import_and_actual_new_source_snapshot_do_not_import_runtime(self):
        code="""import sys,tempfile
from pathlib import Path
from research import trivialaugment_delivery as d
with tempfile.TemporaryDirectory() as temp:
 d.FROZEN_PATH=Path(temp)/'synthetic-freeze.json'
 d.write(d.FROZEN_PATH,{'planned_repeat_source_sha256':d.repeat.source_snapshot()})
 d.FROZEN_SHA=d.sha(d.FROZEN_PATH)
 d.source_snapshot()
assert not {'torch','numpy','runtime'}&set(sys.modules)
print('stdlib-only')
"""
        self.assertEqual(subprocess.check_output([sys.executable,'-c',code],cwd=delivery.ROOT,text=True).strip(),'stdlib-only')

    def test_authentication_and_both_tensor_audits_precede_hidden_decoder(self):
        before=set(sys.modules)
        with tempfile.TemporaryDirectory() as temp,self.assertRaises(ValueError):delivery.worker(Path(temp),'b'*64)
        self.assertFalse({'torch','numpy','runtime'}&(set(sys.modules)-before))
        tree=ast.parse(Path(delivery.__file__).read_text());worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
        self.assertEqual(worker.body[0].value.func.id,'validate_worker')
        imports=[n.names[0].name for n in worker.body if isinstance(n,ast.Import)]
        self.assertEqual(imports[0],'runtime');self.assertLess(imports.index('runtime'),imports.index('torch'))
        calls={n.func.id:n.lineno for n in ast.walk(worker) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
        self.assertLess(calls['audit_tensors'],calls['load_kaggle_test'])
        self.assertNotIn('torch.load',ast.get_source_segment(Path(delivery.__file__).read_text(),worker))
        for n in tree.body:
            if isinstance(n,(ast.Import,ast.ImportFrom)):
                name=n.module if isinstance(n,ast.ImportFrom) else n.names[0].name
                self.assertNotIn(name.split('.')[0],('torch','numpy','runtime','data','model','kaggle'))

    def test_failed_qualification_creates_no_lease_or_prediction_and_loads_no_runtime(self):
        before=set(sys.modules)
        with patch.object(delivery,'binding_for',side_effect=ValueError('Incomplete/negative repeat')),patch.object(delivery.control,'claim_lease') as lease:
            with self.assertRaisesRegex(ValueError,'Incomplete'):delivery.parent('future_prediction')
            lease.assert_not_called()
        self.assertFalse({'torch','numpy','runtime'}&(set(sys.modules)-before))

    def fixture(self,root,stack):
        base=root/'results/research';folder=base/'prediction_unit';folder.mkdir(parents=True)
        data=root/'data';data.mkdir();primary=base/'wrn34_trivialaugment_bf16_s2026';primary.mkdir()
        repeat=base/'wrn34_trivialaugment_bf16_s2027';repeat.mkdir()
        now=datetime.now(timezone.utc);start=now-timedelta(seconds=60);end=now-timedelta(seconds=10)
        for d,seed in ((primary,26),(repeat,27)):
            (d/'best.pt').write_bytes(f'actual-{seed}-selected-fixture'.encode());(d/'last.pt').write_bytes(f'actual-{seed}-last-fixture'.encode())
        env={key:'same' for key in delivery.paired.ENVIRONMENT_KEYS}
        env.update(packages={'torch':'2.12.0+cu130'},device='cuda',amp_dtype='torch.bfloat16',cudnn=92000,cuda_runtime='13.0',gpu_memory_GiB=7.96,
                   cudnn_benchmark=True,deterministic_algorithms=False,process_local_cudnn_path_entries_removed=1)
        delivery.write(primary/'environment.json',env)
        delivery.write(repeat/'supervisor.json',{'schema':delivery.repeat.SCHEMA,'finished_at':(start-timedelta(seconds=20)).isoformat()})
        with (data/'sample_submission.csv').open('w',newline='') as f:
            f.write('id,label\n');f.writelines(f'{i},0\n' for i in range(2000))
        sample_sha=delivery.sha(data/'sample_submission.csv')
        csv_path=folder/'submission.csv';csv_path.write_bytes((data/'sample_submission.csv').read_bytes())
        frozen_path=base/'frozen.json';delivery.write(frozen_path,{'frozen':True})
        frozen={'checkpoint_path':str(primary/'best.pt'),'repeat_run_path':str(repeat),'data_dir':str(data),
                'data_signature':delivery.aggregate.DATA,'evaluation_context':delivery.aggregate.CONTEXT}
        documents={'primary':{'research_training_integrity':{'seed':2026}},'repeat':{'checkpoint_sha256':delivery.sha(repeat/'best.pt'),'research_training_integrity':{'seed':2027}}}
        qualify=root/'qualified_evidence.json';delivery.write(qualify,{'audited':True});binding={str(qualify):delivery.sha(qualify)};sources={'owned.py':'a'*64}
        (root/'research').mkdir();(root/'research/benchmark_trivialaugment.py').write_bytes((delivery.ROOT/'research/benchmark_trivialaugment.py').read_bytes())
        stack.enter_context(patch.object(delivery,'ROOT',root));stack.enter_context(patch.object(delivery,'FROZEN_PATH',frozen_path))
        stack.enter_context(patch.object(delivery,'binding_for',return_value=(frozen,documents,binding,sources)))
        stack.enter_context(patch.object(delivery,'validate_csv',side_effect=lambda path,sample:submit_verified.validate_csv(path,sample,sample_sha)))
        stack.enter_context(patch('research.process_tree_guard.matching_process',return_value=None))
        delivery.write(root/'research/campaign.json',{'stops_at':'2099-01-01T00:00:00+00:00'})
        inputs=delivery.prediction_inputs(folder.name,frozen,documents,binding,sources);delivery.write(folder/'inputs.json',inputs);digest=delivery.sha(folder/'inputs.json')
        audit=delivery.tensor_audit_expected(frozen,documents);delivery.write(folder/'tensor_audit.json',audit)
        nonce='a'*32;directory=folder/('process_tree_'+nonce);directory.mkdir()
        spec={'schema':'owned_python_process_tree_v1','nonce':nonce,'host_pid':101,'host_created_at':10.,'cwd':str(root),'script':str(Path(delivery.__file__).resolve()),
            'script_sha256':delivery.sha(delivery.__file__),'guard_source_sha256':delivery.PINS['research/process_tree_guard.py'],
            'arguments':['--_worker',str(folder),'--_manifest-sha',digest],'deadline_unix':start.timestamp()+178,
            'job_name':'Local\\SaberAugMix_'+nonce if os.name=='nt' else None}
        method='windows_job_v1' if os.name=='nt' else 'posix_session_v1'
        ready={'schema':spec['schema'],'nonce':nonce,'pid':202,'created_at':20.,'script_sha256':spec['script_sha256'],'method':method}
        done={**ready,'exit_code':0}
        proof={'schema':spec['schema'],'nonce':nonce,'method':method,'guard_directory':str(directory),'guard_source_sha256':spec['guard_source_sha256'],
            'child_pid':202,'child_created_at':20.,'cleanup_verified':True,'timed_out':False,'completion_authenticated':True,'completion_exit_code':0,'child_returncode':0}
        for name,value in (('spec.json',spec),('ready.json',ready),('completion.json',done),('host_result.json',proof)):delivery.write(directory/name,value)
        document={'schema':delivery.DELIVERY_SCHEMA,'status':'complete','created_at':(start+timedelta(seconds=30)).isoformat(),
            'run':primary.name,'competition':delivery.COMPETITION,'frozen_manifest_path':str(frozen_path),'frozen_manifest_sha256':delivery.FROZEN_SHA,
            'checkpoint_path':str(primary/'best.pt'),'checkpoint_sha256':delivery.paired.PRIMARY_CHECKPOINT_SHA,'checkpoint_epoch':159,'weight_variant':'ema',
            'parameter_count':4166298,'policy':delivery.POLICY,'seed':2026,'training_examples':40000,'validation_examples':5000,
            'repeat_run_path':str(repeat),'repeat_checkpoint_sha256':documents['repeat']['checkpoint_sha256'],'repeat_receipt_schema':delivery.repeat.SCHEMA,
            'repeat_receipt_sha256':delivery.sha(repeat/'supervisor.json'),'repeat_validator_source_sha256':delivery.PINS['research/run_trivialaugment_repeat.py'],
            'prediction_input_manifest_sha256':digest,'source_sha256':sources,'input_sha256':binding,'tensor_audit_sha256':delivery.sha(folder/'tensor_audit.json'),
            'actual_environment':env,'evaluation_context':frozen['evaluation_context'],'csv_path':str(csv_path),'submission_sha256':delivery.sha(csv_path),
            'rows':2000,'columns':['id','label'],'ids_match_official_order':True,'generator_source_sha256':delivery.sha(delivery.__file__),
            'confirmation_evaluated':False,'hidden_test_labels_loaded':False,'repeat_weights_used_for_hidden_prediction':False,'external_data_loaded':False,'bn_recalibrated':False}
        delivery.write(folder/'delivery.json',document)
        receipt={'schema':delivery.SCHEMA,'run':folder.name,'mode':'predict','backend':'local','status':'complete','pid':101,'process_created_at':10.,
            'child_pid':202,'child_created_at':20.,'started_at':start.isoformat(),'finished_at':end.isoformat(),'elapsed_seconds':50.,'reserved_seconds':180,
            'deadline_unix':start.timestamp()+180,'source_sha256':sources,'input_sha256':binding,'input_manifest_sha256':digest,'confirmation_evaluated':False,
            'submitted':False,'process_tree_cleanup_verified':True,'process_tree_guard_results':[proof],'artifact':'delivery.json','artifact_sha256':delivery.sha(folder/'delivery.json')}
        delivery.write(folder/'supervisor.json',receipt)
        return folder,document,receipt,frozen,documents,directory,qualify

    def update_document(self,folder,document,receipt):
        delivery.write(folder/'delivery.json',document);receipt={**receipt,'artifact_sha256':delivery.sha(folder/'delivery.json')};delivery.write(folder/'supervisor.json',receipt)
        return receipt

    def test_complete_pure_delivery_details_are_exact_and_do_not_import_runtime(self):
        before=set(sys.modules)
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            folder,doc,receipt,*_=self.fixture(Path(temp).resolve(),stack)
            result=delivery.validate_delivery(folder/'delivery.json')
            self.assertEqual(result['seed'],2026);self.assertEqual(result['checkpoint_sha256'],delivery.paired.PRIMARY_CHECKPOINT_SHA)
            self.assertEqual(result['repeat_receipt_schema'],delivery.repeat.SCHEMA)
            self.assertEqual(result['repeat_validator_source_sha256'],delivery.PINS['research/run_trivialaugment_repeat.py'])
            self.assertEqual(result['submission_sha256'],delivery.sha(folder/'submission.csv'))
        self.assertFalse({'torch','numpy','runtime'}&(set(sys.modules)-before))

    def test_false_tensor_audit_unknown_schema_repeat_weights_and_metadata_never_qualify(self):
        for key,value in (('schema','research_prediction_delivery_v2'),('checkpoint_sha256','f'*64),('seed',2027),('weight_variant','model'),
                          ('checkpoint_epoch',160),('parameter_count',5000001),('policy','flip'),('training_examples',50000),
                          ('repeat_weights_used_for_hidden_prediction',True),('hidden_test_labels_loaded',True),('rows',True)):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
                folder,doc,receipt,*_=self.fixture(Path(temp).resolve(),stack)
                receipt=self.update_document(folder,{**doc,key:value},receipt)
                with self.assertRaises(ValueError):delivery.validate_delivery(folder/'delivery.json')
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            folder,doc,receipt,*_=self.fixture(Path(temp).resolve(),stack)
            audit=delivery.read(folder/'tensor_audit.json');audit['performed_in_authenticated_prediction_worker']=False;delivery.write(folder/'tensor_audit.json',audit)
            self.update_document(folder,{**doc,'tensor_audit_sha256':delivery.sha(folder/'tensor_audit.json')},receipt)
            with self.assertRaisesRegex(ValueError,'tensor-audit'):delivery.validate_delivery(folder/'delivery.json')

    def test_unknown_cleanup_or_missing_host_and_live_owner_reject(self):
        for kind in ('flag','host','pid','source','deadline'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
                folder,doc,receipt,frozen,documents,directory,_=self.fixture(Path(temp).resolve(),stack)
                if kind=='flag':delivery.write(folder/'supervisor.json',{**receipt,'process_tree_cleanup_verified':False})
                if kind=='host':delivery.write(directory/'host_result.json',{**delivery.read(directory/'host_result.json'),'cleanup_verified':False})
                if kind=='pid':stack.enter_context(patch('research.process_tree_guard.matching_process',return_value=object()))
                if kind=='source':
                    spec=delivery.read(directory/'spec.json');delivery.write(directory/'spec.json',{**spec,'script':str(delivery.ROOT/'research/trivialaugment_paired_v2.py')})
                if kind=='deadline':delivery.write(folder/'supervisor.json',{**receipt,'reserved_seconds':181})
                with self.assertRaises(ValueError):delivery.validate_delivery(folder/'delivery.json')

    def test_actual_csv_reordered_duplicate_noninteger_and_changed_input_reject(self):
        for kind in ('order','duplicate','label','input'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
                folder,doc,receipt,frozen,documents,directory,qualify=self.fixture(Path(temp).resolve(),stack)
                if kind=='input':qualify.write_text('changed after actual prediction')
                else:
                    lines=(folder/'submission.csv').read_text().splitlines()
                    if kind=='order':lines[1],lines[2]=lines[2],lines[1]
                    if kind=='duplicate':lines[2]=lines[1]
                    if kind=='label':lines[1]='0,10'
                    (folder/'submission.csv').write_text('\n'.join(lines)+'\n')
                    self.update_document(folder,{**doc,'submission_sha256':delivery.sha(folder/'submission.csv')},receipt)
                with self.assertRaises(ValueError):delivery.validate_delivery(folder/'delivery.json')

    def test_prediction_must_follow_repeat_and_old_freeze_is_not_prediction(self):
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            folder,doc,receipt,frozen,*_=self.fixture(Path(temp).resolve(),stack)
            rp=Path(frozen['repeat_run_path'])/'supervisor.json';delivery.write(rp,{'schema':delivery.repeat.SCHEMA,'finished_at':receipt['finished_at']})
            self.update_document(folder,{**doc,'repeat_receipt_sha256':delivery.sha(rp)},receipt)
            with self.assertRaisesRegex(ValueError,'follow qualifying repeat'):delivery.validate_delivery(folder/'delivery.json')
            with self.assertRaises(ValueError):delivery.validate_delivery(folder/'delivery.json',require_prediction=False)

    def test_no_action_starts_no_prediction_and_inexact_worker_argv_rejects(self):
        with patch.object(delivery,'parent') as parent,patch.object(delivery,'worker') as worker,redirect_stdout(io.StringIO()):
            delivery.main([]);parent.assert_not_called();worker.assert_not_called()
            with self.assertRaises(ValueError):delivery.main(['--run','orphan'])
            with self.assertRaises(ValueError):delivery.main(['--_worker','x','--_manifest-sha'])

    def test_prediction_budget_and_unknown_cleanup_fail_closed(self):
        from research.process_tree_guard import TreeCleanupError
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            root=Path(temp);(root/'research').mkdir()
            delivery.write(root/'research/campaign.json',{'stops_at':'2026-11-05T00:00:00-05:00','concurrent_local_gpu_jobs':1,
                'local_gpu_seconds_per_utc_day':21600,'local_job_timeout_seconds':10800})
            now=datetime(2026,11,3,tzinfo=timezone.utc)
            frozen={'data_signature':{},'evaluation_context':{}};documents={'repeat':{'checkpoint_sha256':'a'*64}}
            stack.enter_context(patch.object(delivery,'ROOT',root));stack.enter_context(patch.object(delivery,'binding_for',return_value=(frozen,documents,{},{})))
            stack.enter_context(patch.object(delivery.time,'time',return_value=now.timestamp()))
            with patch.object(delivery.control,'accounting',return_value={'now':lambda:now,'used_seconds':lambda *a:21600}),patch('research.process_tree_guard.run_guarded') as launch:
                with self.assertRaises(ValueError):delivery.parent('fresh_prediction')
                launch.assert_not_called();self.assertFalse((root/'results/research/local_lease.json').exists())
            def unknown(*args,**kwargs):
                args[4].update(child_pid=999,process_tree_cleanup_verified=False);raise TreeCleanupError('synthetic unknown cleanup')
            with patch.object(delivery.control,'accounting',return_value={'now':lambda:now,'used_seconds':lambda *a:0}),patch('research.process_tree_guard.run_guarded',side_effect=unknown):
                with self.assertRaises(TreeCleanupError):delivery.parent('fresh_prediction')
            self.assertTrue(delivery.read(root/'results/research/local_lease.json')['cleanup_requires_reconciliation'])
            self.assertEqual(delivery.read(root/'results/research/fresh_prediction/supervisor.json')['status'],'cleanup_unverified')


class DeliveryTensorBoundaryCI(unittest.TestCase):
    def test_actual_CPU_checkpoint_loads_both_roles_and_returns_primary_not_repeat(self):
        """CPU-only CI boundary test; existing typed-validator suite checks its math."""
        import runtime  # Runtime remains first, also in the CPU-only CI fixture.
        import torch
        from research import benchmark_trivialaugment as typed
        from research import benchmark as core
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folders={role:root/role for role in ('primary','repeat')}
            documents={}
            for role,seed in (('primary',2026),('repeat',2027)):
                folder=folders[role];folder.mkdir();integrity={'actual_fixture_role':role};documents[role]={'config':{'seed':seed},'research_training_integrity':integrity}
                torch.save({'config':{'seed':seed},'model':{'sentinel':torch.tensor([float(seed)])}},folder/'best.pt')
                torch.save({'last_role':role,'optimizer':{'state':{'momentum':torch.tensor([1.])}}},folder/'last.pt')
                for name,value in (('metrics.json',{'role':role}),('history.json',[{'epoch':160}]),('provenance.json',{}),('environment.json',{}),('config.json',{'seed':seed})):
                    delivery.write(folder/name,value)
                (folder/'input_config.json').write_text(json.dumps({'seed':seed}))
            frozen={'checkpoint_path':str(folders['primary']/'best.pt'),'repeat_run_path':str(folders['repeat']),'data_signature':{'official':'fixture'}}
            seen=[]
            def typed_check(checkpoint,metrics,history,provenance,env,last,raw):
                role=metrics['role'];self.assertEqual(last['last_role'],role);self.assertIn('optimizer',last);seen.append(role)
                return documents[role]['research_training_integrity']
            with patch.object(typed,'validate_training',side_effect=typed_check) as validator,patch.object(core,'validate_checkpoint') as core_validator:
                checkpoint,audit=delivery.audit_tensors(frozen,documents,object(),object())
            self.assertEqual(seen,['primary','repeat']);self.assertEqual(validator.call_count,2);self.assertEqual(core_validator.call_count,2)
            self.assertEqual(checkpoint['config']['seed'],2026);self.assertEqual(checkpoint['model']['sentinel'].item(),2026.)
            self.assertTrue(audit['artifacts']['repeat']['actual_selected_last_optimizer_parent_and_TA_RNG_reloaded'])


if __name__=='__main__':unittest.main()
