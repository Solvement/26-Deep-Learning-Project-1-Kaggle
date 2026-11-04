"""CPU-only BN diagnostic contracts; no GPU, account API or official arrays."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
from datetime import datetime, timedelta, timezone
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research import benchmark_augmix_bn as diagnostic


class AugMixBNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import runtime
        import torch
        cls.torch = torch
        torch.set_num_threads(1)

    def tiny(self):
        t = self.torch
        return t.nn.Sequential(t.nn.Conv2d(3,4,3,padding=1), t.nn.BatchNorm2d(4),
            t.nn.ReLU(),t.nn.AdaptiveAvgPool2d(1),t.nn.Flatten(),t.nn.Linear(4,10))

    def test_conversion_precedes_optimizer_preserves_all_initial_bytes_and_parameter_count(self):
        from model import build_model, count_parameters
        model = build_model({"depth":22,"widen_factor":4,"dropout":0.,"num_classes":10})
        original = diagnostic.state_digest(model.state_dict())
        old_ids = {id(p) for p in model.parameters()}
        self.assertEqual(diagnostic.convert_bn(model),19)
        self.assertEqual(diagnostic.state_digest(model.state_dict()),original)
        self.assertEqual(count_parameters(model),4298970)
        optimizer = self.torch.optim.SGD(model.parameters(),lr=.1,momentum=.9,weight_decay=.0005,nesterov=True)
        self.assertEqual({id(p) for p in optimizer.param_groups[0]['params']},{id(p) for p in model.parameters()})
        self.assertNotEqual(old_ids,{id(p) for p in model.parameters()})

    def test_conversion_preserves_mode_eps_momentum_and_requires_grad(self):
        t = self.torch
        model = t.nn.Sequential(t.nn.BatchNorm2d(3,eps=.02,momentum=None)).eval()
        model[0].weight.requires_grad_(False)
        initial = diagnostic.state_digest(model.state_dict())
        diagnostic.convert_bn(model)
        self.assertFalse(model[0].training)
        self.assertFalse(model[0].weight.requires_grad)
        self.assertEqual(model[0].eps,.02)
        self.assertIsNone(model[0].momentum)
        self.assertEqual(diagnostic.state_digest(model.state_dict()),initial)

    def test_local_context_disables_only_bn_and_restores_after_exception(self):
        t = self.torch
        model = self.tiny(); diagnostic.convert_bn(model)
        seen = []
        original = t.nn.functional.batch_norm
        def batch_norm(*args,**kwargs):
            seen.append(t.backends.cudnn.enabled)
            return original(*args,**kwargs)
        hook = model[0].register_forward_hook(lambda *_:self.assertTrue(t.backends.cudnn.enabled))
        try:
            with t.backends.cudnn.flags(enabled=True):
                with patch.object(t.nn.functional,'batch_norm',side_effect=batch_norm):
                    model(t.randn(6,3,32,32)).sum().backward()
                self.assertEqual(seen,[False])
                self.assertTrue(t.backends.cudnn.enabled)
                with patch.object(t.nn.functional,'batch_norm',side_effect=RuntimeError('synthetic BN failure')):
                    with self.assertRaisesRegex(RuntimeError,'synthetic BN failure'):
                        model(t.randn(6,3,32,32))
                self.assertTrue(t.backends.cudnn.enabled)
        finally: hook.remove()

    def test_CPU_arms_match_output_gradient_and_running_statistics_without_GPU_claim(self):
        from research.train_shadow import capture_rng
        t = self.torch; t.manual_seed(2026)
        model = self.tiny()
        prepared = (t.randn(6,3,32,32),t.tensor([0,1],dtype=t.int64))
        rng = capture_rng(t.device('cpu'))
        first, probes_a = diagnostic.run_arm(model,prepared,t.device('cpu'),t.float32,'cudnn',rng)
        second, probes_b = diagnostic.run_arm(model,prepared,t.device('cpu'),t.float32,'native_bn_only',rng)
        self.assertEqual(first['initial_state_sha256'],second['initial_state_sha256'])
        for arm in (first,second):
            self.assertEqual(arm['precision'],'cpu_fp32_test_only')
            self.assertEqual(arm['successful_updates'],23)
            self.assertEqual(arm['finite_positive_gradient_steps'],23)
            self.assertEqual(arm['bn_single_update_verified_steps'],23)
            self.assertEqual(arm['actual_parameter_changed_steps'],23)
            self.assertEqual(len(arm['timed_step_wall_seconds']),20)
            self.assertEqual(arm['timed_step_cuda_ms'],[])
        for key in ('gradients','bn_buffers'):
            self.assertEqual(diagnostic.scalar_difference(probes_a[key],probes_b[key])['max_absolute'],0.)
        self.assertTrue(t.equal(probes_a['output'],probes_b['output']))

    def test_difference_is_scalar_only_and_rejects_nonfinite_or_wrong_topology(self):
        t=self.torch
        result=diagnostic.scalar_difference({'x':t.tensor([1.,2.])},{'x':t.tensor([1.,3.])})
        self.assertEqual(result['max_absolute'],1.)
        self.assertFalse(result['bitwise_equivalence_claimed'])
        self.assertNotIn('tensor',json.dumps(result))
        for other in ({'y':t.ones(2)},{'x':t.ones(3)},{'x':t.tensor([float('nan'),2.])}):
            with self.assertRaises(ValueError): diagnostic.scalar_difference({'x':t.ones(2)},other)

    def test_no_arguments_is_help_only_and_direct_worker_is_rejected(self):
        with patch.object(diagnostic,'worker') as worker,patch.object(diagnostic,'parent') as parent,redirect_stdout(io.StringIO()):
            diagnostic.main([])
            worker.assert_not_called();parent.assert_not_called()
        with self.assertRaises(ValueError): diagnostic.main(['--_worker','.'])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);(path/'diagnostic_inputs.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'manifest differs'):
                diagnostic.worker(path,'0'*64)

    def policy(self):
        return {'concurrent_local_gpu_jobs':1,'local_gpu_seconds_per_utc_day':21600,
            'local_job_timeout_seconds':10800,'stops_at':'2026-11-05T05:00:00+00:00'}

    def test_budget_stays_shared_180_day_and_absolute_hard_stop(self):
        now=datetime(2026,11,1,tzinfo=timezone.utc);p=self.policy()
        self.assertEqual(diagnostic.budget_seconds(p,now,0),180)
        self.assertEqual(diagnostic.budget_seconds(p,now,21590),10)
        stop=datetime.fromisoformat(p['stops_at'])
        self.assertEqual(diagnostic.budget_seconds(p,stop-timedelta(seconds=2),0),2)
        for used in (21600,-1,True,float('nan')):
            with self.assertRaises(ValueError):diagnostic.budget_seconds(p,now,used)
        with self.assertRaises(ValueError):diagnostic.budget_seconds(p,stop,0)

    def contract(self,root):
        folder=root/'results/research/toy';folder.mkdir(parents=True)
        source={'source':'same'}
        manifest={'source_sha256':source,'config_sha256':diagnostic.shared.CONFIG_SHA,'reference_profile_sha256':diagnostic.REFERENCE_SHA}
        (folder/'diagnostic_inputs.json').write_text(json.dumps(manifest))
        digest=diagnostic.sha256(folder/'diagnostic_inputs.json')
        now=datetime.now(timezone.utc)
        worker=SimpleNamespace(pid=101,create_time=lambda:10.)
        parent=SimpleNamespace(pid=202,create_time=lambda:9.)
        receipt={'schema':diagnostic.SCHEMA,'status':'profiling','input_manifest_sha256':digest,
            'started_at':now.isoformat(),'reserved_seconds':120.,'deadline_unix':now.timestamp()+120,
            'pid':202,'process_created_at':9.,'child_pid':101,'child_created_at':10.}
        (folder/'supervisor.json').write_text(json.dumps(receipt))
        lease_path=root/'results/research/local_lease.json';lease_path.write_text(json.dumps({'pid':202,'process_created_at':9.}))
        path=folder/('process_tree_'+'a'*32);path.mkdir()
        spec={'schema':'owned_python_process_tree_v1','nonce':'a'*32,'host_pid':202,'host_created_at':9.,
            'cwd':str(root),'script':str(Path(diagnostic.__file__).resolve()),'script_sha256':diagnostic.sha256(diagnostic.__file__),
            'guard_source_sha256':diagnostic.shared.PINNED_SOURCES['research/process_tree_guard.py'],
            'arguments':['--_worker',str(folder),'--_manifest-sha',digest],'deadline_unix':receipt['deadline_unix'],
            'job_name':'Local\\SaberAugMix_'+'a'*32 if diagnostic.os.name=='nt' else None}
        ready={'schema':spec['schema'],'nonce':spec['nonce'],'pid':101,'created_at':10.,
            'script_sha256':spec['script_sha256'],'method':'windows_job_v1' if diagnostic.os.name=='nt' else 'posix_session_v1'}
        (path/'spec.json').write_text(json.dumps(spec));(path/'ready.json').write_text(json.dumps(ready))
        worker.cmdline=lambda:[diagnostic.sys.executable,str(root/'research/process_tree_guard.py'),'--bootstrap',str(path/'spec.json')]
        return folder,digest,worker,parent,lease_path,path,spec

    def test_guard_rejects_missing_foreign_unknown_lease_and_wrong_nonce_before_native_query(self):
        for change in ('missing','foreign','unknown','nonce'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
                root=Path(temp);folder,digest,worker,parent,lease_path,path,spec=self.contract(root)
                stack.enter_context(patch.object(diagnostic,'ROOT',root))
                stack.enter_context(patch.object(diagnostic.psutil,'Process',side_effect=lambda pid=None:worker if pid is None else parent))
                stack.enter_context(patch.object(diagnostic.os,'getppid',return_value=202))
                native=stack.enter_context(patch.object(diagnostic.shared,'verify_os_boundary'))
                if change=='missing':lease_path.unlink()
                elif change=='foreign':lease_path.write_text(json.dumps({'pid':999,'process_created_at':9.}))
                elif change=='unknown':lease_path.write_text(json.dumps({'pid':202,'process_created_at':9.,'cleanup_requires_reconciliation':True}))
                else:
                    ready=json.loads((path/'ready.json').read_text());ready['nonce']='b'*32;(path/'ready.json').write_text(json.dumps(ready))
                with self.assertRaises((ValueError,FileNotFoundError)):diagnostic.validate_worker(folder,digest)
                native.assert_not_called()

    def active_contract(self, stack, root):
        folder,digest,worker,parent,lease_path,path,spec=self.contract(root)
        original=diagnostic.sha256
        def fingerprint(target):
            target=Path(target)
            if target == root/'research/process_tree_guard.py':
                return diagnostic.shared.PINNED_SOURCES['research/process_tree_guard.py']
            if target == root/diagnostic.REFERENCE: return diagnostic.REFERENCE_SHA
            return original(target)
        stack.enter_context(patch.object(diagnostic,'ROOT',root))
        stack.enter_context(patch.object(diagnostic,'source_snapshot',return_value={'source':'same'}))
        stack.enter_context(patch.object(diagnostic,'sha256',side_effect=fingerprint))
        stack.enter_context(patch.object(diagnostic.psutil,'Process',side_effect=lambda pid=None:worker if pid is None else parent))
        stack.enter_context(patch.object(diagnostic.os,'getppid',return_value=202))
        native=stack.enter_context(patch.object(diagnostic.shared,'verify_os_boundary'))
        return folder,digest,worker,path,spec,native

    def test_complete_script_specific_guard_contract_passes_read_only_native_query(self):
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            folder,digest,worker,path,spec,native=self.active_contract(stack,Path(temp))
            result=diagnostic.validate_worker(folder,digest)
            self.assertEqual(result['config_sha256'],diagnostic.shared.CONFIG_SHA)
            native.assert_called_once()

    def test_bare_bootstrap_wrong_script_long_budget_and_missing_ready_fail_before_native_query(self):
        for change in ('command','script','budget','ready'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
                folder,digest,worker,path,spec,native=self.active_contract(stack,Path(temp))
                if change=='command':worker.cmdline=lambda:[diagnostic.sys.executable,str(diagnostic.__file__),'--_worker',str(folder)]
                elif change=='script':
                    spec['script_sha256']='0'*64;(path/'spec.json').write_text(json.dumps(spec))
                elif change=='budget':
                    receipt=json.loads((folder/'supervisor.json').read_text());receipt['reserved_seconds']=10800
                    (folder/'supervisor.json').write_text(json.dumps(receipt))
                else:(path/'ready.json').unlink()
                with self.assertRaises(ValueError):diagnostic.validate_worker(folder,digest)
                native.assert_not_called()

    def test_failed_actual_OS_membership_cannot_admit_worker(self):
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            folder,digest,worker,path,spec,native=self.active_contract(stack,Path(temp))
            native.side_effect=ValueError('no actual membership')
            with self.assertRaisesRegex(ValueError,'no actual membership'):
                diagnostic.validate_worker(folder,digest)

    def parent_fixture(self,stack,root,guarded):
        from research import run_local, process_tree_guard
        (root/'research').mkdir();(root/'results/research').mkdir(parents=True)
        policy={**self.policy(),'baseline_benchmark':'unused.json'}
        (root/'research/campaign.json').write_text(json.dumps(policy))
        original=diagnostic.sha256
        stack.enter_context(patch.object(diagnostic,'ROOT',root))
        stack.enter_context(patch.object(diagnostic,'source_snapshot',return_value={'same':'source'}))
        stack.enter_context(patch.object(diagnostic,'sha256',side_effect=lambda p:diagnostic.REFERENCE_SHA if Path(p)==root/diagnostic.REFERENCE else original(p)))
        stack.enter_context(patch.object(run_local,'used_seconds',return_value=100))
        return stack.enter_context(patch.object(process_tree_guard,'run_guarded',side_effect=guarded))

    def test_parent_uses_shared_guard_at_most_180_and_releases_verified_lease(self):
        def guarded(command,log,seconds,receipt_path,receipt,cwd):
            self.assertIn('--_worker',command);self.assertLessEqual(seconds,180)
            receipt['process_tree_cleanup_verified']=True
            (receipt_path.parent/'bn_summary.json').write_text(json.dumps({'schema':diagnostic.RESULT_SCHEMA,
                'status':'complete','formal_admission_evidence':False,'accuracy_measured':False}))
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            root=Path(temp);guard=self.parent_fixture(stack,root,guarded)
            diagnostic.parent('cpu_toy_diagnostic')
            guard.assert_called_once()
            self.assertFalse((root/'results/research/local_lease.json').exists())
            receipt=json.loads((root/'results/research/cpu_toy_diagnostic/supervisor.json').read_text())
            self.assertEqual(receipt['status'],'complete');self.assertFalse(receipt['submitted'])

    def test_parent_unknown_tree_retains_and_locks_lease(self):
        def guarded(command,log,seconds,receipt_path,receipt,cwd):
            receipt['process_tree_cleanup_verified']=False
            raise RuntimeError('unknown owned tree')
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            root=Path(temp);self.parent_fixture(stack,root,guarded)
            with self.assertRaisesRegex(RuntimeError,'unknown owned tree'):
                diagnostic.parent('cpu_toy_failure')
            lease=json.loads((root/'results/research/local_lease.json').read_text())
            self.assertTrue(lease['cleanup_requires_reconciliation'])
            self.assertFalse(lease['process_tree_cleanup_verified'])

    def test_existing_lease_or_directory_refuses_without_starting_child(self):
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            root=Path(temp);guard=self.parent_fixture(stack,root,lambda *args,**kwargs:None)
            lease=root/'results/research/local_lease.json';lease.write_text('{"pid":999}')
            with self.assertRaisesRegex(ValueError,'free local lease'):diagnostic.parent('cpu_never_started')
            guard.assert_not_called();self.assertTrue(lease.exists())


if __name__=='__main__':unittest.main()
