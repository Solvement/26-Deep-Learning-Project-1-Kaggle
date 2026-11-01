"""Offline public-source/synthetic-baseline fixtures; no account, data or GPU."""
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
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
    def __init__(self,folder,stage_unknown=False,push_unknown=False):
        self.folder=folder; self.calls=[]; self.stage_unknown=stage_unknown; self.push_unknown=push_unknown
    def call(self,action,payload=None):
        self.calls.append(action); plan=json.loads((self.folder/'kaggle_plan.json').read_text())
        if action in ('dataset-exists','kernel-exists'): return {'exists':False}
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
            if self.push_unknown: raise RuntimeError('ambiguous push response')
            return {'ref':plan['kernel_id'],'version':1,'private_requested':True,'private_verified':True}
        if action=='status': return {'ref':plan['kernel_id'],'version':1,'status':'ERROR'}
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
        receipt=json.loads((self.output/'supervisor.json').read_text()); self.assertEqual(receipt['status'],'cleanup_unverified'); self.assertFalse(receipt['runtime_stop_verified'])
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


if __name__=='__main__': unittest.main()
