"""Pure Pillow contracts plus lazy CPU Torch math tests for LinuxCI only.

Local work during a GPU job may run PureTAContracts alone. Torch imports occur
only in the separately scheduled TorchTAMathTests.setUpClass.
"""
import ast
import copy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np
from research import trivialaugment as ta
from research import incumbent_compare as incumbent

ROOT=Path(__file__).resolve().parents[1]
TRAINER=ROOT/'research/train_trivialaugment.py'


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pure_api():
    """Compile only actual metadata/PIL functions, without executing imports."""
    tree=ast.parse(TRAINER.read_text())
    defaults=next(ast.literal_eval(n.value) for n in ast.parse((ROOT/'train.py').read_text()).body
                  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DEFAULT_CONFIG' for t in n.targets))
    names={'require','json_sha','fixed_config','validate_config','input_config','research_provenance','ta_rng','prepare_ta_pixels','validate_history','best_selection'}
    namespace=dict(ROOT=ROOT,DEFAULT_CONFIG=defaults,ta=ta,np=np,random=random,math=math,json=json,hashlib=hashlib,copy=copy,Path=Path,sha256=file_sha)
    for node in tree.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in {'SCHEMA','PARAMETERS','BASE_CONFIG','BASE_CONFIG_SHA','TA_SOURCE_SHA','SOURCE_KEYS'}:
            namespace[node.targets[0].id]=ast.literal_eval(node.value)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(TRAINER),'exec'),namespace)
    return SimpleNamespace(**namespace)


def history_row(epoch=1):
    return dict(epoch=epoch,successful_updates=313,skipped_updates=0,forward_passes=313,source_examples_seen=40000,model_views_seen=40000,
                ta_operation_counts={name:40000 if name=='identity' else 0 for name in ta.OPERATIONS},ta_rng_state_sha256='a'*64,
                grad_scaler_scale=1.,train_loss=1.,validation_loss=1.,ema_validation_loss=1.1,lr=.0001,
                epoch_seconds=1.,elapsed_seconds=float(epoch),train_accuracy=.5,validation_accuracy=.8,ema_validation_accuracy=.7,selected_variant='model')


class PureTAContracts(unittest.TestCase):
    def test_two_configs_are_exact_parent_addition_and_seed_only_repeat(self):
        before=set(sys.modules);api=pure_api()
        primary=json.loads((ROOT/'research/configs/wrn34_3_trivialaugment.json').read_text())
        repeat=json.loads((ROOT/'research/configs/wrn34_3_trivialaugment_seed2027.json').read_text())
        parent=json.loads((ROOT/'research/configs/wrn34_3.json').read_text())
        self.assertEqual(primary,{**parent,'ta_space':'fixed_standard','ta_strengths':31})
        self.assertEqual(repeat,{**primary,'seed':2027})
        for value in (primary,repeat):api.validate_config({**api.DEFAULT_CONFIG,**value})
        self.assertEqual(api.PARAMETERS,4166298)
        self.assertFalse({'torch','runtime'}&(set(sys.modules)-before))

    def test_exact_config_method_source_and_no_silent_recipe_change(self):
        api=pure_api();config=api.fixed_config();raw=api.input_config(config)
        marker=api.research_provenance('bf16',config,hashlib.sha256(raw).hexdigest())
        self.assertEqual(marker['schema'],'research_trivialaugment_training_v1')
        self.assertEqual(marker['ta_source_sha256'],file_sha(ROOT/'research/trivialaugment.py'))
        self.assertEqual(marker['source_sha256'],{name:file_sha(ROOT/name) for name in api.SOURCE_KEYS})
        self.assertIn('TA_raw_RGB32_then_original_crop_flip',marker['augmentation_order'])
        for key,value in (('epochs',300),('depth',22),('ta_space','wide_standard'),('ta_strengths',30),('mixup_alpha',0.),('seed',2028)):
            with self.assertRaises(ValueError):api.validate_config({**config,key:value})

    def test_CPU_TA_prefix_is_input_preserving_counts_one_view_and_parent_RNG_isolated(self):
        api=pure_api();images=np.random.default_rng(91).integers(0,256,(7,32,32,3),dtype=np.uint8)
        before=images.tobytes();python_state=random.getstate();numpy_state=np.random.get_state()
        pixels,counts=api.prepare_ta_pixels(images,api.ta_rng(2026))
        self.assertEqual((pixels.shape,pixels.dtype),((7,32,32,3),np.uint8))
        self.assertEqual(set(counts),set(ta.OPERATIONS));self.assertEqual(sum(counts.values()),7)
        self.assertEqual(images.tobytes(),before);self.assertEqual(random.getstate(),python_state)
        after=np.random.get_state();self.assertEqual(numpy_state[0],after[0]);self.assertTrue(np.array_equal(numpy_state[1],after[1]));self.assertEqual(numpy_state[2:],after[2:])

    def test_independent_PCG64_is_explicit_checkpointable_and_invalid_state_rejected(self):
        api=pure_api();images=np.random.default_rng(9).integers(0,256,(6,32,32,3),dtype=np.uint8)
        rng=api.ta_rng(2026);api.prepare_ta_pixels(images,rng);saved=copy.deepcopy(rng.bit_generator.state)
        a,ca=api.prepare_ta_pixels(images,rng);b,cb=api.prepare_ta_pixels(images,api.ta_rng(2026,saved))
        self.assertTrue(np.array_equal(a,b));self.assertEqual(ca,cb)
        for state in ({**saved,'bit_generator':'MT19937'},{**saved,'has_uint32':True},{**saved,'uinteger':-1},
                      {**saved,'state':{'state':0,'inc':0}},api.ta_rng(2027).bit_generator.state,{}):
            with self.assertRaises(ValueError):api.ta_rng(2026,state)
        with self.assertRaises(ValueError):api.prepare_ta_pixels(images,np.random.default_rng(np.random.MT19937(2)))

    def test_history_has_exact_single_view_counts_and_cannot_hide_missing_ops(self):
        api=pure_api();row=history_row();counts=api.validate_history([row],1)
        self.assertEqual(sum(counts.values()),40000)
        for key,value in (('forward_passes',626),('source_examples_seen',80000),('model_views_seen',80000),('successful_updates',312),('skipped_updates',1),('grad_scaler_scale',True)):
            with self.assertRaises(ValueError):api.validate_history([{**row,key:value}],1)
        changed=copy.deepcopy(row);changed['ta_operation_counts']['identity']-=1
        with self.assertRaises(ValueError):api.validate_history([changed],1)
        changed=copy.deepcopy(row);changed['ta_operation_counts']['wide_only_op']=0
        with self.assertRaises(ValueError):api.validate_history([changed],1)

    def test_original_selection_and_TA_prefix_order_are_explicit(self):
        api=pure_api();rows=[history_row(i) for i in (1,2)]
        self.assertEqual(api.best_selection(rows),((1,'model'),(.8,-1.)))
        rows[1].update(ema_validation_accuracy=.81,ema_validation_loss=.9,selected_variant='ema')
        self.assertEqual(api.best_selection(rows),((2,'ema'),(.81,-.9)))
        tree=ast.parse(TRAINER.read_text());step=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_step')
        text=ast.unparse(step)
        self.assertLess(text.index('prepare_ta_pixels('),text.index('core_augment('))
        self.assertLess(text.index('core_augment('),text.index('np.random.beta('))
        self.assertEqual(sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='model' for n in ast.walk(step)),1)
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('sam_step','jsd_three_view') for n in ast.walk(step)))

    def gate_api(self):
        core_path=ROOT/'research/benchmark.py';core_tree=ast.parse(core_path.read_text());namespace={}
        gate=next(n for n in core_tree.body if isinstance(n,ast.FunctionDef) and n.name=='promotion_gate')
        exec(compile(ast.Module(body=[gate],type_ignores=[]),str(core_path),'exec'),namespace)
        core=SimpleNamespace(SCHEMA='official_development_stress_v3',POLICIES=incumbent.POLICIES,CASES=incumbent.CASES,
                             sha256=file_sha,promotion_gate=namespace['promotion_gate'])
        path=ROOT/'research/benchmark_trivialaugment.py';tree=ast.parse(path.read_text());scope=dict(ROOT=ROOT,core=core,incumbent=incumbent,math=math,require=pure_api().require)
        for n in tree.body:
            if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('POLICY','PROSPECTIVE_PLAN','PROSPECTIVE_PLAN_SHA'):scope[n.targets[0].id]=ast.literal_eval(n.value)
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prospective_gate')
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),scope)
        return scope['prospective_gate']

    def gate_fixture(self):
        cases={name:dict(accuracy=.96 if name=='clean' else .8,loss=.15,examples=5000,input_pixel_sha256='a'*64) for name in incumbent.CASES}
        report=dict(clean_accuracy=.96,clean_loss=.15,stress_accuracy=.8,cases=cases)
        reference=dict(schema='official_development_stress_v3',status='complete',examples=5000,training_examples=40000,
                       checkpoint_sha256=incumbent.INCUMBENT_CHECKPOINT_SHA,data_signature=incumbent.DATA,evaluation_context=incumbent.CONTEXT,
                       batch_size=128,bn_recalibrated=False,benchmark_source_sha256=file_sha(ROOT/'research/benchmark.py'),stress_cases=list(incumbent.CASES),
                       policies={name:copy.deepcopy(report) for name in incumbent.POLICIES})
        reports=copy.deepcopy(reference['policies'])
        for row in reports.values():row['clean_accuracy']=row['cases']['clean']['accuracy']=.962
        return reports,reference

    def test_prospective_same_weighted_case_gate_cannot_be_rescued_by_mean_or_other_policy(self):
        reports,reference=self.gate_fixture();gate=self.gate_api()
        result=gate(reports,reference,incumbent.DATA,incumbent.CONTEXT)
        self.assertTrue(result['passed']);self.assertFalse(result['automatic_delivery_authorization'])
        weighted=reports['center80_shift20'];weighted['cases']['noise_004']['accuracy']=.78;weighted['cases']['boxblur3']['accuracy']=.84
        weighted['stress_accuracy']=sum(weighted['cases'][c]['accuracy'] for c in incumbent.CASES[1:])/5
        result=gate(reports,reference,incumbent.DATA,incumbent.CONTEXT)
        self.assertTrue(result['mean_gate']['passed']);self.assertFalse(result['passed']);self.assertEqual(result['failed_cases'],['noise_004'])
        self.assertAlmostEqual(result['noise_accuracy_delta'],-.02)
        reference['evaluation_context']={**incumbent.CONTEXT,'amp_dtype':'torch.float16'}
        with self.assertRaises(ValueError):gate(reports,reference,incumbent.DATA,incumbent.CONTEXT)

    def test_importing_and_running_pure_contracts_does_not_import_Torch(self):
        before=set(sys.modules);api=pure_api();api.validate_history([history_row()],1);self.gate_api()
        self.assertFalse({'torch','runtime','model','train'}&(set(sys.modules)-before))


class TorchTAMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # LinuxCI/root-coordinated CPU window only; no module-level Torch import.
        import torch
        from research import train_trivialaugment as trainer
        from research import benchmark_trivialaugment as benchmark
        cls.torch,cls.trainer,cls.benchmark=torch,trainer,benchmark
        torch.set_num_threads(min(2,torch.get_num_threads()))

    def assert_state_equal(self,a,b):
        if isinstance(a,self.torch.Tensor):self.assertTrue(self.torch.equal(a,b))
        elif isinstance(a,np.ndarray):self.assertTrue(np.array_equal(a,b))
        elif isinstance(a,dict):
            self.assertEqual(set(a),set(b))
            for k in a:self.assert_state_equal(a[k],b[k])
        elif isinstance(a,(tuple,list)):
            self.assertEqual(len(a),len(b))
            for x,y in zip(a,b):self.assert_state_equal(x,y)
        else:self.assertEqual(a,b)

    def objects(self,model=None):
        t=self.torch;tr=self.trainer
        model=copy.deepcopy(model) if model is not None else t.nn.Sequential(t.nn.Flatten(),t.nn.Linear(3072,10))
        ema=copy.deepcopy(model).eval().requires_grad_(False)
        optimizer=t.optim.SGD(model.parameters(),lr=.1,momentum=.9,weight_decay=.0005,nesterov=True)
        return model,ema,optimizer,t.amp.GradScaler('cuda',enabled=False)

    def test_forced_identity_matches_original_parent_loss_SGD_EMA_and_every_parent_RNG(self):
        t=self.torch;tr=self.trainer;config=tr.fixed_config();device=t.device('cpu')
        first=self.objects();second=self.objects(first[0]);images=np.random.default_rng(23).integers(0,256,(4,32,32,3),dtype=np.uint8);y=t.tensor([0,1,2,3])
        tr.seed_all(3);initial=tr.capture_rng(device)
        model,ema,opt,scaler=first;x=tr.core_augment(tr.tensor_images(images,device),16)
        mixed=random.random()<.5;self.assertTrue(mixed);lam=float(np.random.beta(.2,.2));permutation=t.randperm(4);x=x*lam+x[permutation]*(1-lam)
        opt.zero_grad(set_to_none=True);logits=model(x);loss=tr.F.cross_entropy(logits,y)*lam+tr.F.cross_entropy(logits,y[permutation])*(1-lam)
        self.assertTrue(tr.optimizer_step(loss,opt,scaler))
        with t.no_grad():t._foreach_lerp_(list(ema.parameters()),list(model.parameters()),1-min(.995,2/11))
        control_rng=tr.capture_rng(device)
        tr.restore_rng(initial,device)
        with patch.object(tr.ta,'sample_operation',return_value=ta.Operation('identity',0)):
            actual=tr.train_step(*second,images,y,config,tr.ta_rng(2026),0,device,t.float32)
        self.assertTrue(t.equal(actual['loss'],loss.detach()));self.assertEqual(actual['lam'],lam)
        for a,b in zip(first[:3],second[:3]):self.assert_state_equal(a.state_dict(),b.state_dict())
        self.assert_state_equal(control_rng,tr.capture_rng(device))
        self.assertEqual(actual['ta_operation_counts']['identity'],4)

    def test_real_TA_SGD_replay_restores_independent_and_parent_streams(self):
        t=self.torch;tr=self.trainer;config=tr.fixed_config();device=t.device('cpu');objects=self.objects();rng=tr.ta_rng(2026)
        images=np.random.default_rng(2).integers(0,256,(4,32,32,3),dtype=np.uint8);y=t.tensor([2,4,6,8]);tr.seed_all(9)
        first=tr.train_step(*objects,images,y,config,rng,0,device,t.float32)
        saved=[copy.deepcopy(obj.state_dict()) for obj in objects[:3]];parent=tr.capture_rng(device);ta_state=copy.deepcopy(rng.bit_generator.state)
        expected=tr.train_step(*objects,images,y,config,rng,1,device,t.float32);expected_states=[copy.deepcopy(obj.state_dict()) for obj in objects[:3]];after=tr.capture_rng(device)
        recovered=self.objects()
        for obj,state in zip(recovered[:3],saved):obj.load_state_dict(state)
        tr.restore_rng(parent,device)
        actual=tr.train_step(*recovered,images,y,config,tr.ta_rng(2026,ta_state),1,device,t.float32)
        self.assertTrue(t.equal(expected['loss'],actual['loss']));self.assertEqual(expected['ta_operation_counts'],actual['ta_operation_counts'])
        for obj,state in zip(recovered[:3],expected_states):self.assert_state_equal(obj.state_dict(),state)
        self.assert_state_equal(after,tr.capture_rng(device));self.assertTrue(first['updated'])

    def formal_fixture(self,epochs=160):
        """Synthetic typed state fixture, not an executed official training run."""
        t=self.torch;tr=self.trainer;config=tr.fixed_config();raw=tr.input_config(config);marker=tr.research_provenance('fp32',config,hashlib.sha256(raw).hexdigest())
        provenance=tr.code_provenance();provenance['research_training']=marker;signature=dict(archive_sha256='a'*64,split_sha256='b'*64)
        model=tr.build_model({k:config[k] for k in ('depth','widen_factor','dropout','num_classes')})
        for name,buffer in model.named_buffers():
            if name.endswith('num_batches_tracked'):buffer.fill_(epochs*313)
        optimizer=t.optim.SGD(model.parameters(),lr=.0001,momentum=.9,weight_decay=.0005,nesterov=True)
        for parameter in model.parameters():optimizer.state[parameter]['momentum_buffer']=t.zeros_like(parameter)
        rng=tr.ta_rng(2026);state=copy.deepcopy(rng.bit_generator.state);state_sha=tr.json_sha(state)
        history=[{**history_row(i),'ta_rng_state_sha256':state_sha} for i in range(1,epochs+1)]
        counts={name:epochs*40000 if name=='identity' else 0 for name in ta.OPERATIONS}
        last=dict(schema=tr.SCHEMA,formal_training_state=True,preflight_state=False,model=copy.deepcopy(model.state_dict()),ema=copy.deepcopy(model.state_dict()),
                  optimizer=optimizer.state_dict(),grad_scaler={},epoch=epochs,updates=epochs*313,skipped_updates=0,forward_passes=epochs*313,
                  config=config,model_config=model.export_config(),parameter_count=tr.PARAMETERS,data_signature=signature,provenance=provenance,history=history,
                  best_accuracy=.8,best_loss=1.,elapsed_seconds=float(epochs),input_config_sha256=hashlib.sha256(raw).hexdigest(),
                  source_examples_seen=epochs*40000,model_views_seen=epochs*40000,ta_operation_counts=counts,ta_rng_state=state,ta_rng_state_sha256=state_sha,**tr.capture_rng(t.device('cpu')))
        selected=copy.deepcopy(model.state_dict())
        for key,value in selected.items():
            if key.endswith('num_batches_tracked'):value.fill_(313)
        checkpoint=dict(schema=tr.SCHEMA,model=selected,model_config=model.export_config(),config=config,epoch=1,variant='model',validation_accuracy=.8,validation_loss=1.,
                        parameter_count=tr.PARAMETERS,data_signature=signature,provenance=provenance,selected_ta_rng_state_sha256=state_sha)
        metrics=dict(schema=tr.SCHEMA,status='complete' if epochs==160 else 'running',epochs_completed=epochs,best_validation_accuracy=.8,best_validation_loss=1.,parameter_count=tr.PARAMETERS,
                     elapsed_seconds=float(epochs),training_examples=40000,validation_examples=5000,config=config,research_training=marker,
                     successful_updates=epochs*313,skipped_updates=0,forward_passes=epochs*313,source_examples_seen=epochs*40000,model_views_seen=epochs*40000,
                     ta_operation_counts=counts,ta_rng_state_sha256=state_sha)
        p=dict(schema=tr.SCHEMA,code=provenance,official_data={},data_signature=signature,train_examples=40000,validation_examples=5000,parameter_count=tr.PARAMETERS,
               normalization=dict(mean=list(tr.MEAN),std=list(tr.STD)))
        env=dict(schema=tr.SCHEMA,device='cpu',amp_dtype='torch.float32',trivialaugment_training_precision='fp32',grad_scaler_enabled=False,research_training=marker)
        return checkpoint,metrics,history,p,env,last,raw

    def test_complete_typed_tensor_optimizer_TA_RNG_counts_and_selection_audit(self):
        fixture=self.formal_fixture();integrity=self.benchmark.validate_training(*fixture)
        self.assertEqual(integrity['successful_updates'],50080);self.assertEqual(integrity['model_views_seen'],6400000)
        self.assertEqual(sum(integrity['ta_operation_counts'].values()),6400000)
        self.assertTrue(integrity['formal_last_state_tensor_optimizer_parent_and_TA_RNG_audited'])
        changed=copy.deepcopy(fixture);changed[5]['optimizer']['state'][0]['momentum_buffer'].view(-1)[0]=float('nan')
        with self.assertRaises(ValueError):self.benchmark.validate_training(*changed)
        changed=copy.deepcopy(fixture);changed[0]['model'][next(k for k in changed[0]['model'] if k.endswith('num_batches_tracked'))].fill_(999)
        with self.assertRaises(ValueError):self.benchmark.validate_training(*changed)

    def test_resume_rejects_preflight_wrong_source_split_raw_and_independent_RNG(self):
        checkpoint,metrics,history,p,env,last,raw=self.formal_fixture(1);tr=self.trainer
        tr.validate_resume(last,last['config'],last['data_signature'],last['provenance'],raw)
        for key,value in (('preflight_state',True),('formal_training_state',False),('ta_rng_state_sha256','f'*64),('source_examples_seen',39999)):
            changed=copy.deepcopy(last);changed[key]=value
            with self.assertRaises(ValueError):tr.validate_resume(changed,last['config'],last['data_signature'],last['provenance'],raw)
        with self.assertRaises(ValueError):tr.validate_resume(last,last['config'],{**last['data_signature'],'split_sha256':'c'*64},last['provenance'],raw)
        changed=copy.deepcopy(last);changed['provenance']['research_training']['source_sha256']['research/trivialaugment.py']='e'*64
        with self.assertRaises(ValueError):tr.validate_resume(changed,last['config'],last['data_signature'],last['provenance'],raw)
        with self.assertRaises(ValueError):tr.validate_resume(last,last['config'],last['data_signature'],last['provenance'],raw+b' ')
        changed=copy.deepcopy(last);changed['ta_rng_state']['bit_generator']='MT19937';changed['ta_rng_state_sha256']=tr.json_sha(changed['ta_rng_state']);changed['history'][-1]['ta_rng_state_sha256']=changed['ta_rng_state_sha256']
        with self.assertRaises(ValueError):tr.validate_resume(changed,last['config'],last['data_signature'],last['provenance'],raw)


if __name__=='__main__':unittest.main()
