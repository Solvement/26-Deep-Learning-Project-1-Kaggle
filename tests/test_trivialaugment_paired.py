"""Stdlib contracts; no numerical runtime, GPU or private inputs needed."""
import ast
from contextlib import ExitStack,redirect_stdout
import copy
from datetime import datetime,timezone
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from research import trivialaugment_paired as pair
from tests import test_incumbent_compare as fixtures


class TrivialAugmentPairedTests(unittest.TestCase):
    def test_early_own_authentication_precedes_all_numerical_imports(self):
        before=set(sys.modules)
        with tempfile.TemporaryDirectory() as temp,self.assertRaises(ValueError):pair.worker(Path(temp),'b'*64)
        self.assertFalse({'torch','numpy','runtime'}&(set(sys.modules)-before))
        tree=ast.parse(Path(pair.__file__).read_text());worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
        self.assertEqual(worker.body[0].value.func.id,'validate_worker')
        self.assertTrue(any(isinstance(n,ast.FunctionDef) and n.name=='validate_owner' for n in tree.body))
        self.assertTrue(any(isinstance(n,ast.FunctionDef) and n.name=='parent' for n in tree.body))
        self.assertFalse(any(isinstance(n,ast.FunctionDef) and n.name=='supervisor_primitives' for n in tree.body))
        for n in tree.body:
            if isinstance(n,(ast.Import,ast.ImportFrom)):
                name=n.module if isinstance(n,ast.ImportFrom) else n.names[0].name
                self.assertNotIn(name.split('.')[0],('torch','numpy','runtime','train','data','model'))

    def test_frozen_sources_and_exact_primary_repeat_config_are_runtime_free(self):
        before=set(sys.modules);sources=pair.source_snapshot()
        self.assertEqual(sources['research/train_trivialaugment.py'],'0d4e1458b8a0bc8c30805371f43dfff42a690398d07cfca405e07bfe8e5e0f0c')
        self.assertEqual(pair.primary.configuration(2027),{**pair.primary.configuration(2026),'seed':2027})
        self.assertFalse({'torch','numpy','runtime'}&(set(sys.modules)-before))

    def test_point_gate_matches_frozen_actual_function_AST(self):
        tree=ast.parse((pair.ROOT/'research/benchmark.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='promotion_gate');namespace={}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'frozen_gate','exec'),namespace)
        b={'clean_accuracy':.9654,'stress_accuracy':.79716}
        for clean,stress in ((.967,.91168),(.9664,.79516),(.9634,.80216),(.962,.94),(.965,.796)):
            a={'clean_accuracy':clean,'stress_accuracy':stress}
            self.assertEqual(pair.point_gate(a,b),namespace['promotion_gate'](a,b))

    def screening(self):
        f=fixtures.IncumbentComparisonTests();inc=f.fixture(incumbent=True);candidate=f.fixture()
        for row in candidate['policies'].values():
            for case in pair.aggregate.CASES:row['cases'][case]['accuracy']=min(1,inc['policies'][pair.POLICY]['cases'][case]['accuracy']+.01)
            row['clean_accuracy']=row['cases']['clean']['accuracy'];row['clean_loss']=row['cases']['clean']['loss']
            row['stress_accuracy']=sum(row['cases'][c]['accuracy'] for c in pair.aggregate.CASES[1:])/5
        baseline=copy.deepcopy(inc)
        candidate['research_training_integrity']={'training':{'schema':'research_trivialaugment_training_v1','training_seed':2026,'precision':'bf16',
            'source_sha256':{n:pair.PINS[n] for n in ('research/train_trivialaugment.py','research/trivialaugment.py','research/train_shadow.py','research/augmentations.py')}}}
        candidate['fixed_primary_policy']=pair.POLICY;candidate['promotion']={pair.POLICY:pair.point_gate(candidate['policies'][pair.POLICY],baseline['policies']['flip'])}
        summary=pair.aggregate.compare_documents(candidate,inc);deltas={c:summary['cases'][c]['accuracy_delta'] for c in pair.aggregate.CASES[1:]}
        candidate['prospective_incumbent']={'policy':pair.POLICY,'incumbent_checkpoint_sha256':pair.aggregate.INCUMBENT_CHECKPOINT_SHA,
           'incumbent_benchmark_sha256':pair.aggregate.INCUMBENT_BENCHMARK_SHA,'mean_gate':pair.point_gate(candidate['policies'][pair.POLICY],inc['policies'][pair.POLICY]),
           'clean_nll_delta':summary['same_policy_deltas']['clean_nll'],'case_accuracy_deltas':deltas,'noise_accuracy_delta':deltas['noise_004'],
           'maximum_allowed_case_accuracy_drop':.01,'per_case_noninferiority_passed':True,'failed_cases':[],'passed':True,
           'prospective_plan_sha256':pair.primary.PINS['research/incumbent-promotion-plan.md'],'automatic_delivery_authorization':False}
        candidate['point_and_case_gates_passed']=True
        return candidate,inc,baseline

    def test_only_true_TA_primary_all_weighted_and_case_gates_can_pair(self):
        candidate,inc,baseline=self.screening();pair.validate_screening(candidate,inc,baseline)
        for field,value in (('schema','research_shadow_training_v1'),('training_seed',2027),('precision','fp16')):
            bad=copy.deepcopy(candidate);bad['research_training_integrity']['training'][field]=value
            with self.assertRaises(ValueError):pair.validate_screening(bad,inc,baseline)
        for key,value in (('fixed_primary_policy','flip'),('point_and_case_gates_passed',False)):
            with self.assertRaises(ValueError):pair.validate_screening({**candidate,key:value},inc,baseline)
        bad=copy.deepcopy(candidate);bad['prospective_incumbent']['passed']=False
        with self.assertRaises(ValueError):pair.validate_screening(bad,inc,baseline)
        bad=copy.deepcopy(candidate);bad['policies'][pair.POLICY]['cases']['noise_004']['accuracy']-=.04
        bad['policies'][pair.POLICY]['stress_accuracy']=sum(bad['policies'][pair.POLICY]['cases'][c]['accuracy'] for c in pair.aggregate.CASES[1:])/5
        with self.assertRaises(ValueError):pair.validate_screening(bad,inc,baseline)

    def bootstrap(self):
        row={'delta_accuracy':0.,'percentile_ci95':[-.001,.001],'candidate_only_correct':0,'baseline_only_correct':0,'both_correct':4800,'both_wrong':200,
             'original_image_net_wins':0,'original_image_net_losses':0}
        return {'original_images':5000,'replicates':10000,'seed':pair.SEED,'sampling_unit':'original_image_with_all_five_stress_views','confidence_level':.95,
                'clean':row,'stress':{**row,'both_correct':20000,'both_wrong':5000}}

    def test_cluster_contract_keeps_CI_crossing_zero_and_rejects_bad_counts(self):
        value=self.bootstrap();pair.validate_bootstrap(value,0.,0.)
        for k,v in (('original_images',25000),('replicates',9999),('sampling_unit','views'),('seed',27)):
            with self.assertRaises(ValueError):pair.validate_bootstrap({**value,k:v},0.,0.)
        for k,v in (('both_correct',True),('delta_accuracy',.1),('percentile_ci95',[float('nan'),1])):
            bad=copy.deepcopy(value);bad['clean'][k]=v
            with self.assertRaises(ValueError):pair.validate_bootstrap(bad,0.,0.)

    def owner_fixture(self,root):
        folder=root/'results/research/unit_pair';folder.mkdir(parents=True)
        guard=root/'research/process_tree_guard.py';guard.parent.mkdir(parents=True);guard.write_bytes((pair.ROOT/'research/process_tree_guard.py').read_bytes())
        directory=folder/('process_tree_'+'a'*32);directory.mkdir();current=1793635200.
        receipt=dict(schema=pair.SCHEMA,status='benchmarking',mode='paired-incumbent',backend='local',run=folder.name,input_manifest_sha256='b'*64,
           started_at=datetime.fromtimestamp(current-1,timezone.utc).isoformat(),reserved_seconds=180,deadline_unix=current+170,pid=101,process_created_at=10.,child_pid=202,child_created_at=20.)
        spec=dict(schema='owned_python_process_tree_v1',nonce='a'*32,host_pid=101,host_created_at=10.,cwd=str(root),script=str(Path(pair.__file__).resolve()),
           script_sha256=pair.sha(pair.__file__),guard_source_sha256=pair.PINS['research/process_tree_guard.py'],arguments=['--_worker',str(folder),'--_manifest-sha','b'*64],
           deadline_unix=current+169,job_name='Local\\SaberAugMix_'+'a'*32 if os.name=='nt' else None)
        ready=dict(schema=spec['schema'],nonce=spec['nonce'],pid=202,created_at=20.,script_sha256=spec['script_sha256'],method='windows_job_v1' if os.name=='nt' else 'posix_session_v1')
        pair.write(directory/'spec.json',spec);pair.write(directory/'ready.json',ready);pair.write(root/'results/research/local_lease.json',dict(pid=101,process_created_at=10.))
        process=SimpleNamespace(pid=202,create_time=lambda:20.,cmdline=lambda:[sys.executable,str(guard),'--bootstrap',str(directory/'spec.json')])
        return folder,receipt,spec,ready,process,SimpleNamespace(pid=101,create_time=lambda:10.),current,directory

    def test_explicit_own_OS_identity_rejects_oldscript_nonce_argv_and_PID_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder,receipt,spec,ready,process,parent,current,directory=self.owner_fixture(root)
            with patch.object(pair,'ROOT',root),patch.object(pair.time,'time',return_value=current),patch.object(pair.guard_checks,'verify_os_boundary'):
                pair.validate_owner(folder,'b'*64,{'run':folder.name},receipt,process,parent)
                for key,value in (('script',str(root/'research/paired_incumbent.py')),('nonce','c'*32),('arguments',['--_worker','wrong']),('deadline_unix',current+200)):
                    pair.write(directory/'spec.json',{**spec,key:value})
                    with self.assertRaises(ValueError):pair.validate_owner(folder,'b'*64,{'run':folder.name},receipt,process,parent)
                pair.write(directory/'spec.json',spec)
                with self.assertRaises(ValueError):pair.validate_owner(folder,'b'*64,{'run':folder.name},{**receipt,'process_created_at':11.},process,parent)

    def test_budget_or_unknown_cleanup_cannot_launch_unbounded_or_release_lease(self):
        from research.process_tree_guard import TreeCleanupError
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            root=Path(temp);(root/'research').mkdir();pair.write(root/'research/campaign.json',dict(stops_at='2026-11-05T00:00:00-05:00',concurrent_local_gpu_jobs=1,
                local_gpu_seconds_per_utc_day=21600,local_job_timeout_seconds=10800))
            now=datetime(2026,11,2,tzinfo=timezone.utc)
            stack.enter_context(patch.object(pair,'ROOT',root));stack.enter_context(patch.object(pair,'source_snapshot',return_value={}))
            stack.enter_context(patch.object(pair,'binding_for',return_value=({}, {}, {})));stack.enter_context(patch.object(pair.subprocess,'check_output',return_value='a'*40))
            stack.enter_context(patch.object(pair.time,'time',return_value=now.timestamp()))
            with patch.object(pair.control,'accounting',return_value={'now':lambda:now,'used_seconds':lambda *a:21600}),patch('research.process_tree_guard.run_guarded') as launch:
                with self.assertRaises(ValueError):pair.parent('unit',root/'best.pt',root/'benchmark.json',root)
                launch.assert_not_called()
            def fail(*args,**kwargs):
                args[4].update(child_pid=999,process_tree_cleanup_verified=False);raise TreeCleanupError('unknown cleanup')
            with patch.object(pair.control,'accounting',return_value={'now':lambda:now,'used_seconds':lambda *a:0}),patch('research.process_tree_guard.run_guarded',side_effect=fail):
                with self.assertRaises(TreeCleanupError):pair.parent('unit',root/'best.pt',root/'benchmark.json',root)
            self.assertTrue(pair.read(root/'results/research/local_lease.json')['cleanup_requires_reconciliation'])
            self.assertEqual(pair.read(root/'results/research/unit/supervisor.json')['status'],'cleanup_unverified')

    def test_no_arguments_do_not_start_and_inexact_worker_arguments_reject(self):
        with patch.object(pair,'parent') as parent,patch.object(pair,'worker') as worker,redirect_stdout(io.StringIO()):
            pair.main([]);parent.assert_not_called();worker.assert_not_called()
            with self.assertRaises(ValueError):pair.main(['--_worker','x','--_manifest-sha'])

    def freeze_fixture(self,root,stack):
        raw=root/'research/configs/repeat.json';raw.parent.mkdir(parents=True);raw.write_text('{"seed":2027}')
        config={'seed':2026,'epochs':160};repeat={**config,'seed':2027}
        sources={}
        for name in ('research/run_trivialaugment_repeat.py','research/trivialaugment-repeat-plan.md'):
            p=root/name;p.write_text(name);sources[name]=pair.sha(p)
        stack.enter_context(patch.object(pair,'ROOT',root));stack.enter_context(patch.object(pair,'source_snapshot',return_value={}))
        stack.enter_context(patch.object(pair.primary,'CONFIGS',{2027:('research/configs/repeat.json',pair.sha(raw))}))
        stack.enter_context(patch.object(pair.primary,'configuration',return_value=repeat))
        folder=root/'results/research/paired';folder.mkdir(parents=True)
        paired=folder/'paired_incumbent.json';pair.write(paired,{'schema':pair.RESULT_SCHEMA})
        checkpoint=folder/'best.pt';checkpoint.write_bytes(b'primary')
        benchmark=folder/'benchmark.json';pair.write(benchmark,{'config':config,'model_config':{'depth':34},'parameter_count':4166298,
            'checkpoint_epoch':159,'checkpoint_variant':'ema','promotion':{pair.POLICY:{'clean_delta':.0042}}})
        manifest={'candidate_checkpoint':str(checkpoint),'candidate_benchmark':str(benchmark),'data_dir':str(root/'data')}
        evidence={str(paired):pair.sha(paired),str(benchmark):pair.sha(benchmark)}
        stack.enter_context(patch.object(pair,'completed_pair',return_value=(manifest,{'legacy_bootstrap':self.bootstrap()},evidence)))
        decision={'decided_at':'2026-11-02T00:00:00+00:00','checkpoint_sha256':pair.PRIMARY_CHECKPOINT_SHA,
            'benchmark_sha256':pair.PRIMARY_BENCHMARK_SHA,'paired_sha256':pair.sha(paired),'policy':pair.POLICY,
            'planned_repeat_seed':2027,'planned_repeat_config_path':str(raw),'planned_repeat_config_sha256':pair.sha(raw),
            'planned_repeat_effective_config':repeat,'planned_repeat_effective_config_sha256':pair.control.json_sha(repeat),
            'planned_repeat_source_sha256':sources,'repeat_run_path':str(root/'results/research/fresh27')}
        dp=folder/'decision.json';pair.write(dp,decision)
        return folder,dp,decision,folder/'frozen.json'

    def test_freeze_explicit_sources_and_repeat_config_are_bound_and_revalidated(self):
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            folder,decision_path,decision,output=self.freeze_fixture(Path(temp).resolve(),stack)
            value=pair.freeze(folder,decision_path,output);self.assertEqual(pair.validate_frozen(output),value)
            # A later formal repeat folder may exist; read-only validation remains usable.
            Path(value['repeat_run_path']).mkdir();self.assertEqual(pair.validate_frozen(output),value)
            for key,val in (('policy','flip'),('parameter_count',5_000_001),('selected_epoch',160),('weight_variant','model'),
                            ('planned_repeat_config_sha256','0'*64),('planned_repeat_effective_config',{'seed':2026})):
                pair.write(output,{**value,key:val})
                with self.assertRaises(ValueError):pair.validate_frozen(output)
            pair.write(output,value)
            source=Path(temp)/'research/run_trivialaugment_repeat.py';source.write_text('changed after freeze')
            with self.assertRaises(ValueError):pair.validate_frozen(output)

    def test_freeze_rejects_missing_fake_source_or_started_repeat_and_never_overwrites(self):
        for variant in ('missing','fake','started','wrongseed'):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
                folder,dp,decision,output=self.freeze_fixture(Path(temp).resolve(),stack)
                if variant=='missing':decision['planned_repeat_source_sha256']={}
                if variant=='fake':decision['planned_repeat_source_sha256']['research/run_trivialaugment_repeat.py']='f'*64
                if variant=='started':Path(decision['repeat_run_path']).mkdir()
                if variant=='wrongseed':decision['planned_repeat_seed']=2026
                pair.write(dp,decision)
                with self.assertRaises(ValueError):pair.freeze(folder,dp,output)
                self.assertFalse(output.exists())
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            folder,dp,decision,output=self.freeze_fixture(Path(temp).resolve(),stack);pair.freeze(folder,dp,output)
            with self.assertRaises(ValueError):pair.freeze(folder,dp,output)

    def test_complete_pair_unknown_cleanup_is_rejected_without_any_torch(self):
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            root=Path(temp);folder=root/'results/research/unit';folder.mkdir(parents=True)
            manifest={'schema':pair.INPUT_SCHEMA,'run':'unit','data_dir':str(root),'candidate_checkpoint':str(root/'best.pt'),
                'candidate_benchmark':str(root/'benchmark.json'),'source_sha256':{},'input_sha256':{},'data_identity':{},'comparison':{}}
            pair.write(folder/'inputs.json',manifest);pair.write(folder/'paired_incumbent.json',{})
            receipt={'schema':pair.SCHEMA,'status':'complete','run':'unit','input_manifest_sha256':pair.sha(folder/'inputs.json'),
                'source_sha256':{},'input_sha256':{},'artifact':'paired_incumbent.json','artifact_sha256':pair.sha(folder/'paired_incumbent.json'),
                'process_tree_cleanup_verified':False}
            pair.write(folder/'supervisor.json',receipt)
            stack.enter_context(patch.object(pair,'ROOT',root));stack.enter_context(patch.object(pair,'source_snapshot',return_value={}))
            stack.enter_context(patch.object(pair,'binding_for',return_value=({}, {}, {})))
            before=set(sys.modules)
            with self.assertRaises(ValueError):pair.completed_pair(folder)
            self.assertFalse({'torch','numpy','runtime'}&(set(sys.modules)-before))

    def test_completed_guard_duration_identity_cleanup_and_PID_reuse_boundaries(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder,receipt,spec,ready,process,parent,current,directory=self.owner_fixture(root)
            start=datetime.now(timezone.utc).timestamp()-60
            receipt.update(started_at=datetime.fromtimestamp(start,timezone.utc).isoformat(),finished_at=datetime.fromtimestamp(start+50,timezone.utc).isoformat(),
                elapsed_seconds=50.,deadline_unix=start+180,process_tree_cleanup_verified=True)
            spec['deadline_unix']=start+178
            done={**ready,'exit_code':0}
            proof=dict(schema=spec['schema'],nonce=spec['nonce'],method=ready['method'],guard_directory=str(directory),
                guard_source_sha256=spec['guard_source_sha256'],cleanup_verified=True,timed_out=False,
                completion_authenticated=True,completion_exit_code=0,child_returncode=0)
            with patch.object(pair,'ROOT',root),patch('research.process_tree_guard.matching_process',return_value=None) as matching:
                pair.validate_completed_guard(folder,receipt,spec,ready,done,proof,'b'*64)
                self.assertEqual(matching.call_args_list[0].args,(101,10.))
                self.assertEqual(matching.call_args_list[1].args,(202,20.))
                for key,val in (('reserved_seconds',181),('elapsed_seconds',48.),('elapsed_seconds',float('nan')),('backend','cloud'),
                                ('mode','old'),('finished_at',receipt['started_at']),('deadline_unix',start+181)):
                    with self.subTest(key=key,val=val),self.assertRaises(ValueError):pair.validate_completed_guard(folder,{**receipt,key:val},spec,ready,done,proof,'b'*64)
                for key,val in (('guard_source_sha256','0'*64),('nonce','c'*32),('deadline_unix',start+181)):
                    with self.assertRaises(ValueError):pair.validate_completed_guard(folder,receipt,{**spec,key:val},ready,done,proof,'b'*64)
                for key,val in (('method','old'),('created_at',21.),('pid',203)):
                    with self.assertRaises(ValueError):pair.validate_completed_guard(folder,receipt,spec,{**ready,key:val},done,proof,'b'*64)
                # Identity lookup compares creation time; a reused numeric PID is absent.
                matching.return_value=object()
                with self.assertRaises(ValueError):pair.validate_completed_guard(folder,receipt,spec,ready,done,proof,'b'*64)


if __name__=='__main__':unittest.main()
