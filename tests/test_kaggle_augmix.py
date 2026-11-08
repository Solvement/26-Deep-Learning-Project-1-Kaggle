"""Offline public-source/synthetic-baseline fixtures; no account, data or GPU."""
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import shutil
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import warnings
from unittest.mock import patch
import zipfile

from research import run_kaggle_augmix as runner
from research.run_colab_augmix import SIGNATURE, CACHE_SHA
from research import kaggle_research_transport as transport


class FakeAccount:
    """Typed-boundary fake, no SDK import or network; preserves private bytes."""
    def __init__(self,folder,stage_unknown=False,push_unknown=False,parent_drift=False,post_parent_drift=False):
        self.folder=folder; self.calls=[]; self.stage_unknown=stage_unknown; self.push_unknown=push_unknown
        self.parent_drift=parent_drift; self.post_parent_drift=post_parent_drift
    def call(self,action,payload=None):
        self.calls.append(action); plan=json.loads((self.folder/'kaggle_plan.json').read_text())
        if action in ('dataset-exists','kernel-exists'): return {'exists':False}
        if action=='kernel-inventory':
            assert not plan.get('reuse_parent'),'Explicit positive parent mode must not infer global inventory absence'
            return {'exists':False,'inventory_complete':True,'authenticated_owner':'kevin1220'}
        if action=='verify-reuse-parent':
            if self.parent_drift: raise RuntimeError('reuse_parent_drift')
            return {'status':'verified','kernel_id':transport.REUSE_ID,'ref':transport.REUSE_REF,'latest_version':1,
                'parent_source_unchanged':True,'parent_metadata_unchanged':True,'parent_complete_outputs_unchanged':True}
        if action=='quota': return {'remaining_seconds':90000.,'reserved_seconds':0.,'pay_to_scale_enabled':False}
        if action=='verify-train': return {'ref':runner.TRAIN_EXPORT,'version':2,'is_private':True}
        if action in ('stage','verify-baseline'):
            intent=json.loads(Path(payload['intent_path']).read_text())
            assert runner.digest(payload['intent_path'])==payload['intent_sha256']
            if action=='stage':
                assert Path(intent['lease_path']).exists()
                if self.stage_unknown: raise RuntimeError('unknown metadata result')
                return {'ref':plan['baseline_dataset_id'],'version':1,'metadata_create_returned':True}
            return {'ref':plan['baseline_dataset_id'],'version':1,'is_private':True,'files_sha256':intent['payload_sha256'],'actual_download_verified':True}
        if action=='push':
            intent=json.loads(Path(payload['intent_path']).read_text()); assert Path(intent['lease_path']).exists()
            assert Path(intent['ledger_path']).exists(); assert runner.digest(payload['intent_path'])==payload['intent_sha256']
            metadata=json.loads((Path(intent['kernel_folder'])/'kernel-metadata.json').read_text())
            assert metadata['dataset_sources']==[plan['baseline_dataset_id']+'/1'] and metadata['is_private'] is True and metadata['enable_internet'] is False
            if plan.get('reuse_parent'):
                assert metadata['id_no']==transport.REUSE_ID and metadata['id']==transport.REUSE_REF and metadata['title']==transport.REUSE_TITLE
            if self.push_unknown: raise RuntimeError('ambiguous push response')
            if self.post_parent_drift: raise RuntimeError('postpush parent v1 output drift')
            return {'ref':plan['kernel_id'],'version':2 if plan.get('reuse_parent') else 1,'kernel_id':transport.REUSE_ID if plan.get('reuse_parent') else 999,
                'private_requested':True,'private_verified':True,'parent_v1_content_unchanged':True if plan.get('reuse_parent') else None}
        if action=='status': return {'ref':plan['kernel_id'],'version':2 if plan.get('reuse_parent') else 1,'status':'ERROR'}
        if action=='output':
            folder=Path(payload['folder']); folder.mkdir()
            # Official kernels_output also writes its fixed slug.log even
            # when file_pattern selected only the two artifact files.
            (folder/(plan['kernel_id'].split('/')[1]+'.log')).write_text('private synthetic kernel log')
            remote={'schema':'research_kaggle_augmix_remote_v1','status':'failed','input_plan_sha256':plan['input_plan_sha256'],
                'source_sha256':plan['source_sha256'],'confirmation_evaluated':False,'hidden_images_loaded':False,'external_data_loaded':False}
            blobs={'remote_receipt.json':json.dumps(remote).encode(),'controller.log':b'predeclared budget admission failed'}
            archive=folder/'augmix_artifacts.private.zip'
            with zipfile.ZipFile(archive,'w') as z:
                for name,raw in blobs.items(): z.writestr(name,raw)
            (folder/'augmix_output_manifest.json').write_text(json.dumps({'schema':'research_kaggle_augmix_outputs_v1','plan_sha256':plan['input_plan_sha256'],
                'archive_sha256':runner.digest(archive),'members_sha256':{n:hashlib.sha256(b).hexdigest() for n,b in blobs.items()},'remote_status':'failed'}))
            return {'files':[str(p) for p in folder.iterdir()]}
        raise AssertionError('Unexpected fake account action '+action)


class KaggleAugMixTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root=Path(temp.name); self.output=self.root/'results/research/offline_kaggle'
        for name in runner.source_names():
            path=self.root/name; path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes((runner.ROOT/name).read_bytes())
        base=self.root/'results/regularized'; base.mkdir(parents=True)
        for name in runner.BASELINE_FILES: (base/name).write_bytes(('synthetic own baseline '+name).encode())
        self.baseline_sha=runner.digest(base/'best.pt')
        data=self.root/'data'; data.mkdir()
        (data/'training_manifest.json').write_text(json.dumps({'source_provenance':{'official_train_npz_sha256':SIGNATURE['archive_sha256']},
            'split':{'indices_sha256':SIGNATURE['split_sha256'],'train_indices_sha256':'f'*64},'files_sha256':CACHE_SHA}))
        def git(args,**kwargs):
            if args[1]=='rev-parse': return 'a'*40+'\n'
            self.assertEqual(args[:2],['git','show'])
            return (self.root/args[2].split(':',1)[1]).read_bytes()
        self.git=patch.object(runner.subprocess,'check_output',side_effect=git); self.git.start(); self.addCleanup(self.git.stop)
        self.pin=patch.object(runner,'BASELINE_SHA',self.baseline_sha); self.pin.start(); self.addCleanup(self.pin.stop)
        self.meta=patch.object(runner,'TRAINING_MANIFEST_SHA',runner.digest(data/'training_manifest.json')); self.meta.start(); self.addCleanup(self.meta.stop)
        (self.root/'research/kaggle_research_transport.py').write_bytes((runner.ROOT/'research/kaggle_research_transport.py').read_bytes())

    def prepare(self): return runner.prepare(self.root,self.output,'offline_augmix')

    def ledger(self): return {'schema':'research_kaggle_gpu_ledger_v1','limit_seconds':90000,'initial_history_audited':True,'entries':[]}

    def init_audit(self):
        campaign=self.root/'research/campaign.json'; campaign.write_text('{}')
        probe=self.root/'quota.json'; probe.write_text('{"fixture":"typed raw quota"}')
        audit=self.root/'audit.json'; audit.write_text(json.dumps({'schema':'research_kaggle_opening_budget_audit_v1',
            'auditor':'root_agent_under_existing_campaign_authorization','additional_limit_seconds':90000,
            'official_gpu_quota_reserved_seconds':0,'historical_account_exclusivity_claimed':False,'unknown_old_account_jobs_reclaimed':False,
            'official_gpu_quota_time_used_seconds':4952.029,'conservative_opening_charge_seconds':4953,
            'campaign_sha256':runner.digest(campaign),'quota_probe':'quota.json','quota_probe_sha256':runner.digest(probe),
            'legacy_receipts':{},'research_receipt_scan':[]}))
        runner.initialize_ledger(self.root,audit,runner.digest(audit))
        return self.root/'results/research/kaggle_gpu_ledger.json'

    def parent_archive(self):
        folder=self.root/'results/research/approved_parent_v1'; folder.mkdir(parents=True)
        source=b'synthetic approved failed v1 source\n'
        pin=patch.object(transport,'REUSE_SOURCE_SHA',hashlib.sha256(source).hexdigest()); pin.start(); self.addCleanup(pin.stop)
        metadata={'id':transport.REUSE_ID,'ref':transport.REUSE_REF,'currentVersionNumber':1,'isPrivate':True,'enableInternet':False,'title':transport.REUSE_TITLE}
        def write(name,value):
            path=folder/name; path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(value if isinstance(value,bytes) else json.dumps(value).encode())
        for label in ('current','pinned_v1'):
            write(label+'.metadata.json',metadata); write(label+'.source.py',source)
        write('parent_verification.json',{'status':'readonly_mismatch_or_error','error_type':'AssertionError','reason':'Old cache was partial'})
        write('archive_hash_manifest.json',{'initial_snapshot_preserved':True})
        write('local_complete_archive_assessment.json',{'all_original_known_bytes_unchanged':True,'all_extra_source_bytes_explained':True,'old_cache_was_partial':True})
        members={transport.REUSE_REF.split('/')[1]+'.log','repeat-seed2027/launch_manifest.json','repeat-seed2027/repeat_status.json',
            *('saber/'+n for n in ('train.py','model.py','data.py','runtime.py','prepare_data.py','predict.py','configs/wrn_final.json','configs/repeat_seed2027.json','__pycache__/runtime.cpython-313.pyc'))}
        for name in members: write('fresh_pinned_v1_outputs/'+name,('synthetic original bytes '+name).encode())
        outputs={n:runner.digest(folder/'fresh_pinned_v1_outputs'/n) for n in members}
        write('parent_verification_reconciled.json',{'schema':'research_owned_failed_kernel_parent_reconciliation_v1','status':'verified_readonly_reconciled',
            'readonly':True,'account_mutations':0,'training_started':False,'confirmation_evaluated':False,'hidden_predictions_generated':False,'pinned_v1_status':'ERROR',
            'preserved_original_probe_report_sha256':runner.digest(folder/'parent_verification.json'),
            'preserved_initial_archive_manifest_sha256':runner.digest(folder/'archive_hash_manifest.json'),
            'local_assessment_sha256':runner.digest(folder/'local_complete_archive_assessment.json'),
            'fresh_complete_v1_output_sha256':outputs,'fresh_complete_v1_output_count':12,'raw_gpu_quota':{'time_reserved':0,'pay_to_scale_enabled':False}})
        self.reseal_parent(folder)
        return folder

    def reseal_parent(self,folder):
        files={str(p.relative_to(folder)).replace('\\','/'):runner.digest(p) for p in folder.rglob('*') if p.is_file() and p.name!='complete_reconciled_hash_manifest.json'}
        (folder/'complete_reconciled_hash_manifest.json').write_text(json.dumps({'schema':'private_parent_version_complete_archive_v1','readonly':True,'account_mutations':0,'files_sha256':files}))

    def reuse_prepare(self,archive):
        return runner.prepare(self.root,self.output,'explicit_reuse',reuse_owned_failed_kernel=archive)

    def test_offline_prepare_exact_private_train_only_source_and_separate_weights(self):
        with patch('numpy.load',side_effect=AssertionError('No official arrays permitted')):
            plan=self.prepare()
        self.assertEqual(runner.validate_prepared(self.output),plan)
        metadata=json.loads((self.output/'kernel-metadata.json').read_text())
        self.assertTrue(metadata['is_private']); self.assertFalse(metadata['enable_internet'])
        self.assertEqual(metadata['kernel_sources'],[runner.TRAIN_EXPORT]); self.assertEqual(metadata['competition_sources'],[])
        with zipfile.ZipFile(self.output/'source.private.zip') as archive:
            self.assertEqual(set(archive.namelist()),set(runner.source_names()))
            self.assertFalse(any(name.endswith('.pt') or name.startswith('data/') for name in archive.namelist()))
        self.assertEqual(set(p.name for p in (self.output/'baseline_dataset').iterdir()),set(runner.BASELINE_FILES)|{'saber_baseline_manifest.json','dataset-metadata.json'})
        compile((self.output/'augmix_entry.py').read_text(),'actual_generated_entry','exec')

    def test_remote_actual_stage_order_and_bounded_children(self):
        self.prepare(); source=(self.output/'augmix_entry.py').read_text()
        names=["stage('prepare_official_training')","stage('preflight')","stage('baseline_benchmark')","stage('scratch_training')","stage('candidate_benchmark')"]
        self.assertEqual([source.index(n) for n in names],sorted(source.index(n) for n in names))
        for required in ('from research.process_tree_guard import run_guarded','run_guarded([sys.executable,*args]','deadline-time.time()',"validate_preflight(pref,P['config']","needed=1.5*pref['median_step_seconds']*313*160+300",'needed+180>deadline-time.time()',"--preflight","--device','cuda'"):
            self.assertTrue(required in source,required)
        self.assertNotIn('official-test-export',source)

    def test_prepared_remote_is_disabled_before_any_path_or_torch_access(self):
        self.prepare(); source=(self.output/'augmix_entry.py').read_text()
        with patch.object(Path,'mkdir',side_effect=AssertionError('No remote work permitted')):
            with self.assertRaisesRegex(RuntimeError,'Offline-only workload template'): exec(compile(source,'disabled_entry','exec'),{'__name__':'__main__'})

    def test_platform_slug_and_title_fit_and_match(self):
        plan=self.prepare()
        for path,key in ((self.output/'kernel-metadata.json','kernel_id'),(self.output/'baseline_dataset/dataset-metadata.json','baseline_dataset_id')):
            metadata=json.loads(path.read_text()); self.assertEqual(metadata['title'],plan[key].split('/')[1]); self.assertLessEqual(len(metadata['title']),50)
        with self.assertRaisesRegex(ValueError,'2..30'): runner.prepare(self.root,self.root/'results/research/too_long','a'*31)

    def test_dirty_source_rejected_before_any_output(self):
        (self.root/'train.py').write_text('changed frozen trainer')
        with self.assertRaisesRegex(ValueError,'Frozen four-file core'): self.prepare()
        self.assertFalse(self.output.exists())

    def test_changed_baseline_rejected_without_artifacts(self):
        (self.root/'results/regularized/best.pt').write_bytes(b'wrong weights')
        with self.assertRaisesRegex(ValueError,'Trusted baseline'): self.prepare()
        self.assertFalse(self.output.exists())

    def test_wrong_official_metadata_rejected_before_output(self):
        path=self.root/'data/training_manifest.json'; value=json.loads(path.read_text()); value['split']['indices_sha256']='0'*64; path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'official training manifest fingerprint'): self.prepare()
        self.assertFalse(self.output.exists())

    def test_prepare_is_immutable_and_changed_payload_fails(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError,'immutable'): self.prepare()
        (self.output/'baseline_dataset/metrics.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'payload changed'): runner.validate_prepared(self.output)

    def test_launch_explicitly_blocked_before_api_intent_or_lease(self):
        self.prepare()
        with patch.object(runner.subprocess,'run',side_effect=AssertionError('No transport allowed')):
            with self.assertRaisesRegex(ValueError,'before authentication, intent, lease or push'): runner.launch(self.output)
        self.assertFalse((self.output/'launch_intent.json').exists())
        self.assertFalse((self.root/'results/research/cloud_lease.json').exists())

    def test_raw_timedelta_preserves_days_and_nonzero_reserved(self):
        response=SimpleNamespace(gpu_quota=SimpleNamespace(time_used=timedelta(hours=5),time_reserved=timedelta(seconds=.01),total_time_allowed=timedelta(hours=30),minimum_time_allowed=timedelta(),is_pay_to_scale_enabled=False))
        raw=runner.raw_gpu_quota(response)
        self.assertEqual(raw,{'remaining_seconds':90000.,'reserved_seconds':.01})
        with self.assertRaisesRegex(ValueError,'Another GPU reservation'): runner.validate_quota(raw,10800)
        response.gpu_quota.time_reserved=timedelta()
        runner.validate_quota(runner.raw_gpu_quota(response),10800)

    def test_raw_quota_missing_nan_paytoscale_failclosed(self):
        for raw in ({'remaining_seconds':10800.}, {'remaining_seconds':float('nan'),'reserved_seconds':0}, {'remaining_seconds':10799,'reserved_seconds':0}):
            with self.assertRaises(ValueError): runner.validate_quota(raw,10800)
        with self.assertRaises(ValueError): runner.raw_gpu_quota(SimpleNamespace())

    def test_cumulative_25h_reservations_do_not_reset_or_retry(self):
        ledger=self.ledger()
        for n in range(8): ledger=runner.reserve_ledger(ledger,str(n),10800,'a'*64)
        self.assertEqual(runner.validate_ledger(ledger,3600),86400)
        with self.assertRaisesRegex(ValueError,'25-GPU-hour'): runner.reserve_ledger(ledger,'9',10800,'b'*64)
        with self.assertRaisesRegex(ValueError,'cannot be retried'): runner.reserve_ledger(ledger,'1',300,'b'*64)
        ledger['entries'][0]['status']='push_unknown'
        with self.assertRaisesRegex(ValueError,'cannot be retried'): runner.reserve_ledger(ledger,'0',300,'b'*64)
        ledger['initial_history_audited']=False
        with self.assertRaisesRegex(ValueError,'audited cumulative'): runner.validate_ledger(ledger,300)

    def test_active_cloud_lease_refuses_and_hardstop_caps_server_timeout(self):
        lease=self.root/'cloud_lease.json'; plan={'max_seconds':10800}; raw={'remaining_seconds':90000,'reserved_seconds':0}
        now=runner.STOP-timedelta(seconds=900)
        result=runner.admission(plan,raw,self.ledger(),lease,now)
        self.assertEqual(result['server_session_timeout_seconds'],720)
        self.assertEqual(result['absolute_deadline_unix'],runner.STOP.timestamp()-180)
        lease.write_text('owned Colab job')
        with self.assertRaisesRegex(ValueError,'lease is occupied'): runner.admission(plan,raw,self.ledger(),lease,now)
        with self.assertRaisesRegex(ValueError,'hard stop'): runner.admission(plan,raw,self.ledger(),lease,runner.STOP)

    def test_actual_pipeline_budget_negative_stops_formal_before_resources(self):
        with self.assertRaisesRegex(ValueError,'no formal training'): runner.measured_budget(.410,10800)
        self.assertLess(runner.measured_budget(.10,10800)+180,10800)
        for value in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError): runner.measured_budget(value,10800)

    def test_pinned_terminal_and_delayed_raw_zero_only(self):
        intent={'ref':'kevin1220/saber-one','version':3}; state={**intent,'status':'COMPLETE'}
        self.assertTrue(runner.validate_terminal(intent,state,{'remaining_seconds':90000,'reserved_seconds':0},100,280))
        for value in ({**state,'version':4},{**state,'status':'CANCEL_REQUESTED'},{**state,'ref':'another/kernel'}):
            with self.assertRaises(ValueError): runner.validate_terminal(intent,value,{'remaining_seconds':90000,'reserved_seconds':0},100,280)
        with self.assertRaises(ValueError): runner.validate_terminal(intent,state,{'remaining_seconds':90000,'reserved_seconds':.001},100,280)
        with self.assertRaises(ValueError): runner.validate_terminal(intent,state,{'remaining_seconds':90000,'reserved_seconds':0},100,279)

    def test_zip_traversal_duplicate_and_symlink_refused(self):
        for members in ([('../escape',b'x')],[('safe',b'x'),('safe',b'x')]):
            buffer=io.BytesIO()
            with warnings.catch_warnings(),zipfile.ZipFile(buffer,'w') as z:
                warnings.simplefilter('ignore',UserWarning)
                for name,raw in members: z.writestr(name,raw)
            with zipfile.ZipFile(buffer) as z:
                with self.assertRaises(ValueError): runner.safe_zip(z,{'safe'},100)

    def test_negative_output_preserved_with_exact_hashes_and_no_cleanup_claim(self):
        plan=self.prepare(); remote={'schema':'research_kaggle_augmix_remote_v1','status':'failed','input_plan_sha256':plan['input_plan_sha256'],
            'source_sha256':plan['source_sha256'],'confirmation_evaluated':False,'hidden_images_loaded':False,'external_data_loaded':False}
        blobs={'remote_receipt.json':json.dumps(remote).encode(),'controller.log':b'fixed budget admission failed'}
        archive=self.root/'negative.zip'
        with zipfile.ZipFile(archive,'w') as z:
            for name,raw in blobs.items(): z.writestr(name,raw)
        manifest=self.root/'output.json'; manifest.write_text(json.dumps({'schema':'research_kaggle_augmix_outputs_v1','plan_sha256':plan['input_plan_sha256'],
            'archive_sha256':runner.digest(archive),'members_sha256':{n:hashlib.sha256(b).hexdigest() for n,b in blobs.items()},'remote_status':'failed'}))
        target=self.root/'downloaded'; _,actual=runner.extract_outputs(self.output,archive,manifest,target)
        self.assertEqual(actual['status'],'failed'); self.assertEqual((target/'controller.log').read_bytes(),blobs['controller.log'])
        with self.assertRaisesRegex(ValueError,'immutable'): runner.extract_outputs(self.output,archive,manifest,target)
        doc=json.loads(manifest.read_text()); doc['members_sha256']['controller.log']='0'*64; manifest.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError,'fingerprint'): runner.extract_outputs(self.output,archive,manifest,self.root/'wrong_hash')
        self.assertFalse((self.root/'wrong_hash').exists())

    def test_initial_ledger_requires_explicit_root_audit_hash_and_keeps_opening_usage(self):
        path=self.init_audit(); ledger=json.loads(path.read_text()); self.assertEqual(runner.validate_ledger(ledger,0),4953)
        with self.assertRaises(ValueError): runner.initialize_ledger(self.root,self.root/'audit.json','0'*64)
        with self.assertRaises(FileExistsError): runner.initialize_ledger(self.root,self.root/'audit.json',runner.digest(self.root/'audit.json'))

    def test_private_staging_one_metadata_attempt_and_actual_download_hash_receipt(self):
        self.prepare(); client=FakeAccount(self.output)
        result=runner.stage_baseline(self.root,self.output,client)
        self.assertEqual(result['status'],'complete'); self.assertEqual(client.calls.count('stage'),1)
        self.assertTrue(result['verified']['actual_download_verified']); self.assertEqual(result['verified']['files_sha256']['best.pt'],self.baseline_sha)
        self.assertFalse((self.root/'results/research/cloud_lease.json').exists())
        with self.assertRaisesRegex(ValueError,'forbids metadata retry'): runner.stage_baseline(self.root,self.output,client)
        self.assertEqual(client.calls.count('stage'),1)
        self.assertNotIn('dataset-exists',client.calls)

    def test_unknown_staging_retains_lease_and_never_uploads_again(self):
        self.prepare(); client=FakeAccount(self.output,stage_unknown=True)
        with self.assertRaises(RuntimeError): runner.stage_baseline(self.root,self.output,client)
        self.assertTrue((self.root/'results/research/cloud_lease.json').exists())
        with self.assertRaisesRegex(ValueError,'forbids metadata retry'): runner.stage_baseline(self.root,self.output,client)
        self.assertEqual(client.calls.count('stage'),1)

    def test_unknown_push_retains_source_intent_lease_and_full_reservation(self):
        self.prepare(); client=FakeAccount(self.output,push_unknown=True); runner.stage_baseline(self.root,self.output,client); ledger=self.init_audit()
        with self.assertRaises(RuntimeError): runner.launch(self.output,self.root,client,ledger,self.output/'baseline_receipt.json')
        self.assertTrue((self.root/'results/research/cloud_lease.json').exists())
        record=json.loads(ledger.read_text()); self.assertEqual(record['entries'][0]['status'],'push_unknown'); self.assertEqual(record['entries'][0]['reserved_seconds'],21600)
        with self.assertRaisesRegex(ValueError,'forbids retries'): runner.launch(self.output,self.root,client,ledger,self.output/'baseline_receipt.json')
        self.assertEqual(client.calls.count('push'),1)

    def test_pinned_negative_end_to_end_delays_cleanup_and_retains_charge(self):
        self.prepare(); client=FakeAccount(self.output); runner.stage_baseline(self.root,self.output,client); ledger=self.init_audit()
        clock=[datetime.now(timezone.utc).timestamp()]
        def sleep(seconds): clock[0]+=seconds
        with patch.object(runner.time,'time',side_effect=lambda:clock[0]),patch.object(runner.time,'sleep',side_effect=sleep):
            result=runner.launch(self.output,self.root,client,ledger,self.output/'baseline_receipt.json')
        self.assertEqual(result['status'],'terminal_negative'); self.assertTrue(result['runtime_stop_verified'])
        self.assertEqual(client.calls.count('push'),1); self.assertEqual(client.calls.count('status'),1)
        self.assertGreaterEqual(clock[0]-result['terminal_observed_unix'],180)
        self.assertFalse((self.root/'results/research/cloud_lease.json').exists())
        record=json.loads(ledger.read_text()); self.assertEqual(record['entries'][0]['status'],'terminal'); self.assertEqual(runner.validate_ledger(record,0),4953+21600)
        self.assertEqual(Path(record['entries'][0]['supervisor_path']).name,'kaggle_supervisor.json')

    def test_isolated_transport_intent_checks_bytes_before_authentication(self):
        self.prepare(); client=FakeAccount(self.output,stage_unknown=True)
        with self.assertRaises(RuntimeError): runner.stage_baseline(self.root,self.output,client)
        path=self.output/'baseline_stage_intent.json'; payload={'intent_path':str(path),'intent_sha256':runner.digest(path)}
        self.assertEqual(transport.intent(payload,'baseline_stage')['dataset_ref'],json.loads((self.output/'kaggle_plan.json').read_text())['baseline_dataset_id'])
        (self.output/'baseline_dataset/best.pt').write_bytes(b'changed after intent')
        with patch.object(transport.importlib.metadata,'version',side_effect=AssertionError('No SDK/version/auth access before validation')):
            with self.assertRaisesRegex(ValueError,'Bound source/payload changed'): transport.execute('stage',payload)

    def test_transport_reference_rejects_latest_wrong_owner_and_unpinned_versions(self):
        self.assertEqual(transport.ref('kevin1220/saber-private-job/1',True),['kevin1220','saber-private-job','1'])
        for value in ('kevin1220/saber-private-job/latest','kevin1220/saber-private-job/0','other/saber-private-job/1','kevin1220/saber-private-job'):
            with self.assertRaises(ValueError): transport.ref(value,True)

    def test_transport_unknown_first_interpreter_cannot_replay_mutation(self):
        self.prepare(); client=FakeAccount(self.output,stage_unknown=True)
        with self.assertRaises(RuntimeError): runner.stage_baseline(self.root,self.output,client)
        path=self.output/'baseline_stage_intent.json'; payload={'intent_path':str(path),'intent_sha256':runner.digest(path)}
        with patch.object(transport.importlib.metadata,'version',side_effect=RuntimeError('simulated interpreter failure before SDK')) as checked:
            with self.assertRaises(RuntimeError): transport.execute('stage',payload)
            self.assertTrue((self.output/'baseline_transport_claim.json').exists())
            with self.assertRaises(FileExistsError): transport.execute('stage',payload)
            self.assertEqual(checked.call_count,1)

    def test_wrong_remote_version_never_downloads_or_releases_lease(self):
        self.prepare(); client=FakeAccount(self.output); runner.stage_baseline(self.root,self.output,client); ledger=self.init_audit()
        original=client.call
        def call(action,payload=None):
            if action=='status': return {'ref':json.loads((self.output/'kaggle_plan.json').read_text())['kernel_id'],'version':2,'status':'COMPLETE'}
            return original(action,payload)
        client.call=call
        with self.assertRaisesRegex(ValueError,'Pinned kernel version changed'): runner.launch(self.output,self.root,client,ledger,self.output/'baseline_receipt.json')
        self.assertTrue((self.root/'results/research/cloud_lease.json').exists()); self.assertNotIn('output',client.calls)

    def test_terminal_reserved_lag_never_clears_lease_or_refunds_charge(self):
        self.prepare(); client=FakeAccount(self.output); runner.stage_baseline(self.root,self.output,client); ledger=self.init_audit()
        original=client.call; queries=[0]
        def call(action,payload=None):
            result=original(action,payload)
            if action=='quota':
                queries[0]+=1
                if queries[0]>1: result['reserved_seconds']=.001
            return result
        client.call=call; clock=[datetime.now(timezone.utc).timestamp()]
        with patch.object(runner.time,'time',side_effect=lambda:clock[0]),patch.object(runner.time,'sleep',side_effect=lambda n:clock.__setitem__(0,clock[0]+n)):
            with self.assertRaisesRegex(ValueError,'Another GPU reservation'): runner.launch(self.output,self.root,client,ledger,self.output/'baseline_receipt.json')
        self.assertTrue((self.root/'results/research/cloud_lease.json').exists())
        receipt=json.loads((self.output/'kaggle_supervisor.json').read_text()); self.assertEqual(receipt['status'],'cleanup_unverified'); self.assertFalse(receipt['runtime_stop_verified'])
        self.assertEqual(json.loads(ledger.read_text())['entries'][0]['reserved_seconds'],21600)

    def test_ledger_is_reread_after_exclusive_lease_before_push(self):
        self.prepare(); client=FakeAccount(self.output); runner.stage_baseline(self.root,self.output,client); ledger=self.init_audit()
        original=runner.bound_intent
        def acquire(*args,**kwargs):
            result=original(*args,**kwargs); value=json.loads(ledger.read_text())
            value['entries']=[{'run':'prior'+str(i),'reserved_seconds':21600,'status':'terminal'} for i in range(3)]
            ledger.write_text(json.dumps(value)); return result
        with patch.object(runner,'bound_intent',side_effect=acquire):
            with self.assertRaisesRegex(ValueError,'25-GPU-hour'): runner.launch(self.output,self.root,client,ledger,self.output/'baseline_receipt.json')
        self.assertNotIn('push',client.calls); self.assertEqual(len(json.loads(ledger.read_text())['entries']),3)
        self.assertFalse((self.root/'results/research/cloud_lease.json').exists())

    def test_fresh_preparations_bind_unique_owned_refs_and_omit_403_existence_guess(self):
        first=self.prepare(); second=runner.prepare(self.root,self.root/'results/research/fresh_second','offline_augmix')
        self.assertNotEqual(first['baseline_dataset_id'],second['baseline_dataset_id'])
        self.assertNotEqual(first['kernel_id'],second['kernel_id'])
        self.assertTrue(first['baseline_dataset_id'].endswith('-'+first['object_nonce']))
        client=FakeAccount(self.output); original=client.call
        def call(action,payload=None):
            if action=='dataset-exists': raise AssertionError('403 must never be treated as evidence of absence')
            return original(action,payload)
        client.call=call
        self.assertEqual(runner.stage_baseline(self.root,self.output,client)['status'],'complete')
        document=json.loads((self.output/'baseline_stage_intent.json').read_text())
        self.assertEqual(document['creation_policy'],'create_new_only_no_update')
        self.assertEqual(document['object_nonce'],first['object_nonce'])

    def test_readonly_403_and_other_errors_never_mean_object_absent(self):
        class HTTPError(Exception): pass
        for status in (403,500):
            error=HTTPError(); error.response=SimpleNamespace(status_code=status)
            def view(*_,error=error): raise error
            with self.assertRaises(HTTPError): transport.object_exists(None,'kevin1220/saber-synthetic',view)
        missing=HTTPError(); missing.response=SimpleNamespace(status_code=404)
        def missing_view(*_): raise missing
        self.assertEqual(transport.object_exists(None,'kevin1220/saber-synthetic',missing_view),{'exists':False})

    def test_kernel_inventory_complete_pagination_owner_anchor_and_target(self):
        anchor='kevin1220/saber-official-data-export-20261030'; target='kevin1220/saber-synthetic-target'
        control=SimpleNamespace(ref=anchor,is_private=True,current_version_number=1,id=123)
        rows=[SimpleNamespace(ref=anchor),SimpleNamespace(ref='kevin1220/saber-another-one')]
        calls=[]
        class API:
            def kernels_list_with_response(self,**kwargs):
                calls.append(kwargs)
                return SimpleNamespace(kernels=rows[:1],next_page_token='next') if kwargs['page_token'] is None else SimpleNamespace(kernels=rows[1:],next_page_token='')
        with patch.object(transport,'kernel_response',return_value=SimpleNamespace(metadata=control)):
            result=transport.kernel_inventory(API(),target)
        self.assertFalse(result['exists']); self.assertTrue(result['inventory_complete']); self.assertEqual(len(calls),2)
        self.assertTrue(all(c['mine'] is True and c['user'] is None and 'search' not in c for c in calls))
        rows[1]=SimpleNamespace(ref=target)
        with patch.object(transport,'kernel_response',return_value=SimpleNamespace(metadata=control)):
            self.assertTrue(transport.kernel_inventory(API(),target)['exists'])

    def test_kernel_inventory_rejects_wrong_account_partial_or_repeating_cursor(self):
        anchor='kevin1220/saber-official-data-export-20261030'; control=SimpleNamespace(ref=anchor,is_private=True,current_version_number=1,id=123)
        class API:
            def kernels_list_with_response(self,**kwargs): return SimpleNamespace(kernels=[SimpleNamespace(ref='another/saber-not-owned')],next_page_token='')
        with patch.object(transport,'kernel_response',return_value=SimpleNamespace(metadata=control)):
            with self.assertRaises(transport.KernelIdentityAmbiguous): transport.kernel_inventory(API(),'kevin1220/saber-synthetic-target')
        class Loop:
            def kernels_list_with_response(self,**kwargs): return SimpleNamespace(kernels=[SimpleNamespace(ref=anchor)],next_page_token='repeat')
        with patch.object(transport,'kernel_response',return_value=SimpleNamespace(metadata=control)):
            with self.assertRaises(transport.KernelIdentityAmbiguous): transport.kernel_inventory(Loop(),'kevin1220/saber-synthetic-target')

    def test_skeleton_403_is_identity_ambiguity_not_absence_or_stage_permission(self):
        anchor='kevin1220/saber-official-data-export-20261030'; control=SimpleNamespace(ref=anchor,is_private=True,current_version_number=1,id=123)
        row=SimpleNamespace(ref='',title='Reference free title',last_run_time='time',id=0,slug='',author='',to_dict=lambda:{'title':'Reference free title','lastRunTime':'time'})
        class API:
            def kernels_list_with_response(self,**kwargs): return SimpleNamespace(kernels=[SimpleNamespace(ref=anchor),row],next_page_token='')
        with patch.object(transport,'kernel_response',side_effect=[SimpleNamespace(metadata=control),ValueError('403')]):
            with self.assertRaisesRegex(transport.KernelIdentityAmbiguous,'positively resolve'): transport.kernel_inventory(API(),'kevin1220/saber-synthetic-target',slugifier=lambda _: 'reference-free-title')
        self.prepare(); client=FakeAccount(self.output); original=client.call
        def call(action,payload=None):
            if action=='kernel-inventory': raise RuntimeError('visible_identity_ambiguous')
            return original(action,payload)
        client.call=call
        with self.assertRaisesRegex(RuntimeError,'visible_identity_ambiguous'): runner.stage_baseline(self.root,self.output,client)
        self.assertNotIn('stage',client.calls); self.assertFalse((self.output/'baseline_stage_intent.json').exists()); self.assertFalse((self.output/'baseline_transport_claim.json').exists())
        self.assertFalse((self.root/'results/research/cloud_lease.json').exists())

    def test_skeleton_must_resolve_positive_ref_id_title_and_time(self):
        anchor='kevin1220/saber-official-data-export-20261030'; control=SimpleNamespace(ref=anchor,is_private=True,current_version_number=1,id=123)
        row=SimpleNamespace(ref='',title='A title',last_run_time='time',id=0,slug='',author='',to_dict=lambda:{'title':'A title','lastRunTime':'time'})
        actual=SimpleNamespace(ref='kevin1220/a-title',id=456,title='A title',last_run_time='time')
        class API:
            def kernels_list_with_response(self,**kwargs): return SimpleNamespace(kernels=[SimpleNamespace(ref=anchor),row],next_page_token='')
        with patch.object(transport,'kernel_response',side_effect=[SimpleNamespace(metadata=control),SimpleNamespace(metadata=actual)]):
            result=transport.kernel_inventory(API(),'kevin1220/saber-synthetic-target',slugifier=lambda _: 'a-title')
        self.assertFalse(result['exists']); self.assertEqual(result['positive_resolutions'][0]['id'],456)
        actual.title='Changed title'
        with patch.object(transport,'kernel_response',side_effect=[SimpleNamespace(metadata=control),SimpleNamespace(metadata=actual)]):
            with self.assertRaises(transport.KernelIdentityAmbiguous): transport.kernel_inventory(API(),'kevin1220/saber-synthetic-target',slugifier=lambda _: 'a-title')

    def test_explicit_reuse_parent_archive_binds_fixed_identity_partial_reconciliation_and_all_outputs(self):
        archive=self.parent_archive(); plan=self.reuse_prepare(archive)
        self.assertEqual(plan['kernel_mode'],'reuse_owned_failed_v1'); self.assertEqual(plan['kernel_id'],transport.REUSE_REF)
        self.assertEqual(len(plan['reuse_parent']['v1_outputs_sha256']),12); self.assertTrue(plan['reuse_parent']['no_atomic_version_precondition'])
        metadata=json.loads((self.output/'kernel-metadata.json').read_text())
        self.assertEqual(metadata['id_no'],transport.REUSE_ID); self.assertEqual(metadata['title'],transport.REUSE_TITLE)
        self.assertEqual(runner.validate_prepared(self.output),plan)
        before={str(p.relative_to(archive)):runner.digest(p) for p in archive.rglob('*') if p.is_file()}
        client=FakeAccount(self.output); runner.stage_baseline(self.root,self.output,client); ledger=self.init_audit()
        clock=[datetime.now(timezone.utc).timestamp()]
        with patch.object(runner.time,'time',side_effect=lambda:clock[0]),patch.object(runner.time,'sleep',side_effect=lambda n:clock.__setitem__(0,clock[0]+n)):
            result=runner.launch(self.output,self.root,client,ledger,self.output/'baseline_receipt.json')
        self.assertEqual(result['accepted']['version'],2); self.assertEqual(result['accepted']['kernel_id'],transport.REUSE_ID)
        self.assertTrue(result['no_atomic_version_precondition']); self.assertNotIn('kernel-inventory',client.calls)
        self.assertEqual(before,{str(p.relative_to(archive)):runner.digest(p) for p in archive.rglob('*') if p.is_file()})

    def test_parent_archive_rejects_changed_raw_bytes_or_relabelled_version_source_and_training(self):
        archive=self.parent_archive(); original=(archive/'current.metadata.json').read_bytes()
        for field,value in [('id',transport.REUSE_ID+1),('ref','kevin1220/saber-official-data-export-20261030'),('currentVersionNumber',2),('isPrivate',False),('enableInternet',True)]:
            metadata=json.loads(original); metadata[field]=value; (archive/'current.metadata.json').write_text(json.dumps(metadata)); self.reseal_parent(archive)
            with self.assertRaisesRegex(ValueError,'identity/source'): runner.validate_reuse_archive(self.root,archive)
        (archive/'current.metadata.json').write_bytes(original); self.reseal_parent(archive)
        (archive/'pinned_v1.source.py').write_bytes(b'changed parent source'); self.reseal_parent(archive)
        with self.assertRaisesRegex(ValueError,'identity/source'): runner.validate_reuse_archive(self.root,archive)

    def test_reuse_cannot_upgrade_partial_cache_assertion_or_train_success_into_eligible_parent(self):
        archive=self.parent_archive(); path=archive/'parent_verification_reconciled.json'; original=path.read_bytes()
        for field,value in [('training_started',True),('pinned_v1_status','COMPLETE'),('fresh_complete_v1_output_count',3)]:
            record=json.loads(original); record[field]=value; path.write_text(json.dumps(record)); self.reseal_parent(archive)
            with self.assertRaises(ValueError): runner.validate_reuse_archive(self.root,archive)
        path.write_bytes(original)
        assessment=archive/'local_complete_archive_assessment.json'; value=json.loads(assessment.read_text()); value['old_cache_was_partial']=False; assessment.write_text(json.dumps(value))
        record=json.loads(original); record['local_assessment_sha256']=runner.digest(assessment); path.write_text(json.dumps(record)); self.reseal_parent(archive)
        with self.assertRaisesRegex(ValueError,'reconciliation'): runner.validate_reuse_archive(self.root,archive)

    def test_parent_drift_blocks_baseline_before_intent_claim_lease_or_upload(self):
        archive=self.parent_archive(); self.reuse_prepare(archive); client=FakeAccount(self.output,parent_drift=True)
        with self.assertRaisesRegex(RuntimeError,'reuse_parent_drift'): runner.stage_baseline(self.root,self.output,client)
        self.assertNotIn('stage',client.calls); self.assertFalse((self.output/'baseline_stage_intent.json').exists())
        self.assertFalse((self.root/'results/research/cloud_lease.json').exists())

    def test_unknown_postpush_parent_drift_keeps_reservation_lease_and_v1_archive(self):
        archive=self.parent_archive(); self.reuse_prepare(archive); client=FakeAccount(self.output,post_parent_drift=True)
        runner.stage_baseline(self.root,self.output,client); ledger=self.init_audit(); before=runner.digest(archive/'complete_reconciled_hash_manifest.json')
        with self.assertRaisesRegex(RuntimeError,'parent v1 output drift'): runner.launch(self.output,self.root,client,ledger,self.output/'baseline_receipt.json')
        self.assertTrue((self.root/'results/research/cloud_lease.json').exists()); self.assertEqual(client.calls.count('push'),1)
        self.assertEqual(json.loads(ledger.read_text())['entries'][0]['reserved_seconds'],21600)
        self.assertEqual(runner.digest(archive/'complete_reconciled_hash_manifest.json'),before)
        with self.assertRaises(ValueError): runner.launch(self.output,self.root,client,ledger,self.output/'baseline_receipt.json')
        self.assertEqual(client.calls.count('push'),1)

    def test_server_timeout_is_shortened_at_actual_push_and_cannot_exceed_absolute_deadline(self):
        document={'server_timeout_seconds':10800,'absolute_deadline_unix':10000.9}
        self.assertEqual(transport.bounded_server_timeout(document,now=1000),9000)
        with self.assertRaises(ValueError): transport.bounded_server_timeout(document,now=9800)

    def parent_api(self,archive,*,current_version=1,current_change=None,pinned_change=None,bad_source=False,changed_output=False,reserved=0,terminal='ERROR'):
        data=json.loads((archive/'current.metadata.json').read_text()); source=(archive/'current.source.py').read_text()
        def metadata(version,changes):
            doc={**data,'currentVersionNumber':version,**(changes or {})}
            return SimpleNamespace(id=doc['id'],ref=doc['ref'],current_version_number=doc['currentVersionNumber'],is_private=doc['isPrivate'],
                enable_internet=doc['enableInternet'],title=doc['title'],to_dict=lambda **_:doc)
        def response(api,value,pinned=False):
            return SimpleNamespace(metadata=metadata(1 if pinned else current_version,pinned_change if pinned else current_change),
                blob=SimpleNamespace(source='unexpected v1 source' if bad_source and pinned else source))
        class API:
            def kernels_status(self,value):
                assert value==transport.REUSE_REF+'/1'
                return SimpleNamespace(status=SimpleNamespace(name=terminal),to_dict=lambda **_:{'status':terminal})
            def quota_view(self):
                return SimpleNamespace(gpu_quota=SimpleNamespace(time_used=timedelta(seconds=4952.029),time_reserved=timedelta(seconds=reserved),
                    total_time_allowed=timedelta(hours=30),minimum_time_allowed=timedelta(hours=30),is_pay_to_scale_enabled=False))
            def kernels_output(self,value,path,**kwargs):
                assert value==transport.REUSE_REF+'/1' and kwargs=={'force':False,'quiet':True}
                target=Path(path)
                for file in (archive/'fresh_pinned_v1_outputs').rglob('*'):
                    if file.is_file():
                        copied=target/file.relative_to(archive/'fresh_pinned_v1_outputs'); copied.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(file,copied)
                if changed_output: (target/'repeat-seed2027/repeat_status.json').write_text('changed server bytes')
        return API(),response

    def test_transport_reuse_fresh_current_and_pin_v1_positive_checks_use_full_outputs(self):
        archive=self.parent_archive(); contract=runner.validate_reuse_archive(self.root,archive); api,response=self.parent_api(archive)
        with patch.object(transport,'kernel_response',side_effect=response):
            proof=transport.verify_reuse_parent(api,contract,self.root/'fresh_parent_check')
        self.assertEqual(proof['status'],'verified'); self.assertTrue(proof['parent_complete_outputs_unchanged'])
        api,response=self.parent_api(archive,current_version=2,reserved=600)
        with patch.object(transport,'kernel_response',side_effect=response):
            proof=transport.verify_reuse_parent(api,contract,self.root/'post_parent_check',after_version=2)
        self.assertEqual(proof['latest_version'],2); self.assertTrue(proof['no_atomic_version_precondition'])

    def test_transport_reuse_rejects_parent_version_shape_privacy_source_or_terminal_drift(self):
        archive=self.parent_archive(); contract=runner.validate_reuse_archive(self.root,archive)
        cases=[{'current_version':2},{'current_change':{'id':transport.REUSE_ID+1}},{'pinned_change':{'isPrivate':False}},
            {'pinned_change':{'machineShape':'unexpected hardware metadata'}},{'bad_source':True},{'terminal':'COMPLETE'},{'reserved':.001}]
        for index,changes in enumerate(cases):
            api,response=self.parent_api(archive,**changes)
            with self.subTest(changes=changes),patch.object(transport,'kernel_response',side_effect=response):
                with self.assertRaises(transport.ReuseParentMismatch): transport.verify_reuse_parent(api,contract,self.root/('drift_'+str(index)))

    def test_transport_postpush_v1_output_change_is_preserved_failure_and_never_alters_original(self):
        archive=self.parent_archive(); contract=runner.validate_reuse_archive(self.root,archive); original=runner.digest(archive/'complete_reconciled_hash_manifest.json')
        api,response=self.parent_api(archive,current_version=2,changed_output=True,reserved=600)
        with patch.object(transport,'kernel_response',side_effect=response):
            with self.assertRaisesRegex(transport.ReuseParentMismatch,'complete outputs drifted'):
                transport.verify_reuse_parent(api,contract,self.root/'post_drift',after_version=2)
        self.assertEqual(json.loads((self.root/'post_drift/proof.json').read_text())['status'],'failed')
        self.assertEqual(runner.digest(archive/'complete_reconciled_hash_manifest.json'),original)
        partial={**contract,'v1_outputs_sha256':dict(list(contract['v1_outputs_sha256'].items())[:3])}
        with self.assertRaisesRegex(transport.ReuseParentMismatch,'Complete fixed'): transport.verify_reuse_parent(api,partial,self.root/'not_created')
        self.assertFalse((self.root/'not_created').exists())


class LocalBudgetNamespaceTests(unittest.TestCase):
    """Execute the actual frozen scanner via AST with standard-library globals.

    This class can also be AST-loaded alone while a real GPU job is running;
    neither the scanner nor this contract requires importing project/Torch code.
    """
    def scanner_and_cloud_name(self):
        import ast
        root=Path(__file__).resolve().parents[1]
        tree=ast.parse((root/'research/run_local.py').read_text())
        function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='used_seconds')
        namespace={'json':json,'datetime':datetime}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'frozen_used_seconds','exec'),namespace)
        controller=ast.parse((root/'research/run_kaggle_augmix.py').read_text())
        launch=next(node for node in controller.body if isinstance(node,ast.FunctionDef) and node.name=='launch')
        assignment=next(node for node in ast.walk(launch) if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='receipt_path' for t in node.targets))
        cloud_name=assignment.value.right.value
        self.assertEqual(cloud_name,'kaggle_supervisor.json')
        return namespace['used_seconds'],cloud_name

    def test_actual_local_scanner_excludes_running_and_finished_cloud_receipts(self):
        scanner,cloud_name=self.scanner_and_cloud_name()
        day=datetime(2026,11,1,tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary)
            running=base/'cloud_running'; running.mkdir()
            (running/cloud_name).write_text(json.dumps({'backend':'kaggle','status':'running','started_at':day.isoformat()}))
            complete=base/'cloud_complete'; complete.mkdir()
            (complete/cloud_name).write_text(json.dumps({'backend':'kaggle','status':'complete','started_at':day.isoformat(),
                'finished_at':(day+timedelta(hours=3)).isoformat()}))
            self.assertEqual(scanner(base,day,day+timedelta(days=1)),0)
            self.assertFalse(list(base.glob('*/supervisor.json')))

    def test_actual_local_scanner_still_charges_local_completion_and_rejects_active_local(self):
        scanner,cloud_name=self.scanner_and_cloud_name()
        day=datetime(2026,11,1,tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary); cloud=base/'cloud'; cloud.mkdir()
            (cloud/cloud_name).write_text(json.dumps({'backend':'kaggle','status':'running','started_at':day.isoformat()}))
            complete=base/'local_complete'; complete.mkdir()
            (complete/'supervisor.json').write_text(json.dumps({'backend':'local','status':'complete',
                'started_at':(day-timedelta(minutes=30)).isoformat(),'finished_at':(day+timedelta(minutes=30)).isoformat()}))
            self.assertEqual(scanner(base,day,day+timedelta(days=1)),1800)
            active=base/'local_active'; active.mkdir()
            (active/'supervisor.json').write_text(json.dumps({'backend':'local','status':'training','started_at':day.isoformat()}))
            with self.assertRaisesRegex(RuntimeError,'unfinished supervisor'): scanner(base,day,day+timedelta(days=1))


class ReuseMetadataContextTests(unittest.TestCase):
    """Pure stdlib transport fakes; AST-loadable without project/Torch imports."""
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name); self.archive=self.root/'original'; self.archive.mkdir()
        self.source='synthetic immutable parent source\n'
        self.source_pin=patch.object(transport,'REUSE_SOURCE_SHA',hashlib.sha256(self.source.encode()).hexdigest())
        self.source_pin.start(); self.addCleanup(self.source_pin.stop)
        self.original={'id':transport.REUSE_ID,'ref':transport.REUSE_REF,'currentVersionNumber':1,'isPrivate':True,
            'enableInternet':False,'title':transport.REUSE_TITLE,'lastRunTime':'2026-10-31T02:40:00Z','enableGpu':True,'machineShape':'NvidiaL4','totalVotes':0}
        for label in ('current','pinned_v1'): (self.archive/(label+'.metadata.json')).write_text(json.dumps(self.original))
        self.blobs={name:('immutable historical output '+name).encode() for name in transport.REUSE_OUTPUTS}
        self.contract={'ref':transport.REUSE_REF,'kernel_id':transport.REUSE_ID,'parent_version':1,
            'source_sha256':transport.REUSE_SOURCE_SHA,'archive_path':str(self.archive),
            'current_metadata_sha256':transport.canonical_hash(self.original),'pinned_metadata_sha256':transport.canonical_hash(self.original),
            'v1_outputs_sha256':{name:hashlib.sha256(value).hexdigest() for name,value in self.blobs.items()}}

    def check(self,name,*,after=2,pinned_changes=None,current_changes=None,bad_source=False,bad_output=False):
        original=self.original; source=self.source; blobs=self.blobs
        def response(api,value,pinned=False):
            self.assertEqual(value,transport.REUSE_REF+'/1' if pinned else transport.REUSE_REF)
            doc={**original,'currentVersionNumber':1 if pinned else after or 1,**((pinned_changes if pinned else current_changes) or {})}
            metadata=SimpleNamespace(id=doc['id'],ref=doc['ref'],current_version_number=doc['currentVersionNumber'],is_private=doc['isPrivate'],
                enable_internet=doc['enableInternet'],title=doc['title'],to_dict=lambda **_:doc)
            return SimpleNamespace(metadata=metadata,blob=SimpleNamespace(source='changed old source' if bad_source and pinned else source))
        class API:
            def kernels_status(self,value):
                assert value==transport.REUSE_REF+'/1'
                return SimpleNamespace(status=SimpleNamespace(name='ERROR'),to_dict=lambda **_:{'status':'ERROR'})
            def quota_view(self):
                return SimpleNamespace(gpu_quota=SimpleNamespace(time_used=timedelta(seconds=4952),time_reserved=timedelta(seconds=600 if after else 0),
                    total_time_allowed=timedelta(hours=30),minimum_time_allowed=timedelta(hours=30),is_pay_to_scale_enabled=False))
            def kernels_output(self,value,path,**kwargs):
                assert value==transport.REUSE_REF+'/1'
                for relative,body in blobs.items():
                    destination=Path(path)/relative; destination.parent.mkdir(parents=True,exist_ok=True); destination.write_bytes(body)
                if bad_output: (Path(path)/'repeat-seed2027/repeat_status.json').write_bytes(b'changed old output')
        with patch.object(transport,'kernel_response',side_effect=response):
            return transport.verify_reuse_parent(API(),self.contract,self.root/name,after_version=after)

    def test_post_metadata_context_changes_are_recorded_without_claiming_immutable_semantics(self):
        changes={'currentVersionNumber':2,'lastRunTime':'2026-11-01T01:00:00Z','enableGpu':False,'machineShape':'NvidiaTeslaT4','totalVotes':7}
        before=(self.archive/'pinned_v1.metadata.json').read_bytes()
        proof=self.check('post_context',pinned_changes=changes)
        self.assertEqual(proof['status'],'verified'); self.assertFalse(proof['parent_metadata_unchanged'])
        self.assertEqual(proof['pinned_v1_requested_version'],1); self.assertEqual(proof['pinned_v1_metadata_reported_version'],2)
        self.assertFalse(proof['pinned_metadata_version_semantics_asserted'])
        comparison=proof['metadata_comparisons']['pinned_v1']
        self.assertEqual(set(comparison['changed_fields']),set(changes)); self.assertNotEqual(comparison['old_canonical_sha256'],comparison['new_canonical_sha256'])
        self.assertTrue(proof['parent_source_unchanged']); self.assertTrue(proof['parent_complete_outputs_unchanged'])
        self.assertEqual((self.archive/'pinned_v1.metadata.json').read_bytes(),before)
        self.assertEqual(json.loads((self.root/'post_context/pinned_v1.metadata.json').read_text()),{**self.original,**changes})

    def test_post_metadata_can_report_one_or_two_but_not_three(self):
        for version in (1,2):
            proof=self.check('version_'+str(version),pinned_changes={'currentVersionNumber':version})
            self.assertEqual(proof['pinned_v1_metadata_reported_version'],version)
            self.assertEqual(proof['parent_metadata_unchanged'],version==1)
        with self.assertRaises(transport.ReuseParentMismatch): self.check('version_3',pinned_changes={'currentVersionNumber':3})
        with self.assertRaises(transport.ReuseParentMismatch): self.check('invalid_transition',after=3)

    def test_post_context_never_relaxes_identity_title_privacy_or_internet(self):
        for index,changes in enumerate(({'id':transport.REUSE_ID+1},{'ref':'kevin1220/saber-official-data-export-20261030'},
                {'title':'Another title'},{'isPrivate':False},{'enableInternet':True})):
            with self.subTest(changes=changes):
                with self.assertRaises(transport.ReuseParentMismatch): self.check('protected_'+str(index),pinned_changes=changes)

    def test_old_source_or_output_changes_still_fail_after_metadata_context_changes(self):
        with self.assertRaisesRegex(transport.ReuseParentMismatch,'source drifted'):
            self.check('source_drift',pinned_changes={'machineShape':'NvidiaTeslaT4'},bad_source=True)
        with self.assertRaisesRegex(transport.ReuseParentMismatch,'outputs drifted'):
            self.check('output_drift',pinned_changes={'machineShape':'NvidiaTeslaT4'},bad_output=True)

    def test_before_update_full_metadata_snapshot_still_must_match(self):
        with self.assertRaisesRegex(transport.ReuseParentMismatch,'metadata drifted'):
            self.check('before_drift',after=None,current_changes={'machineShape':'NvidiaTeslaT4'})
        proof=self.check('before_unchanged',after=None)
        self.assertTrue(proof['parent_metadata_unchanged']); self.assertTrue(proof['metadata_comparisons']['current']['unchanged'])


if __name__=='__main__': unittest.main()
