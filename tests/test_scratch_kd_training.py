"""Pure config/metadata locally; real learning/state tests are Linux-CI only."""
import ast
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import unittest
from unittest.mock import patch
from research import train_scratch_kd as trainer
from research import benchmark_scratch_kd as benchmark


class PureContracts(unittest.TestCase):
    def test_import_and_help_have_no_numerical_side_effects(self):
        code="import sys;from research import train_scratch_kd as t,benchmark_scratch_kd as b;t.main([]);b.main([]);assert not {'torch','numpy','PIL','runtime'}&set(sys.modules)"
        subprocess.check_output([sys.executable,'-c',code],cwd=trainer.ROOT,text=True)
        with patch.object(trainer,'train') as train,redirect_stdout(io.StringIO()):trainer.main([])
        train.assert_not_called()

    def test_only_alpha_and_repeat_seed_differ_from_frozen_parent(self):
        names=('wrn16_scratch_kd_control.json','wrn16_scratch_kd.json','wrn16_scratch_kd_seed2027.json')
        values=[]
        for name in names:
            raw=(trainer.ROOT/'research/configs'/name).read_bytes();d=json.loads(raw)
            config=trainer.fixed_config(d['kd_alpha'],d['seed']);trainer.input_config(config,raw);values.append(d)
        self.assertEqual(values[1],{**values[0],'kd_alpha':.5});self.assertEqual(values[2],{**values[1],'seed':2027})
        self.assertEqual({k:v for k,v in values[0].items() if k not in ('kd_alpha','kd_temperature')},trainer.read(trainer.ROOT/trainer.BASE_CONFIG))
        for key,value in [('kd_alpha',.25),('kd_temperature',2.),('seed',True),('epochs',200),('cutout',0)]:
            bad={**trainer.fixed_config(),key:value}
            with self.subTest(key=key),self.assertRaises(ValueError):trainer.validate_config(bad)
        raw=json.loads((trainer.ROOT/'research/configs'/names[1]).read_bytes());del raw['kd_temperature']
        with self.assertRaises(ValueError):trainer.input_config(trainer.fixed_config(),json.dumps(raw).encode())

    def history(self,alpha):
        return [{'epoch':i,'successful_updates':313,'skipped_updates':0,'student_forward_passes':313,'teacher_forward_passes':313 if alpha else 0,
            'source_examples_seen':40000,'student_inputs_seen':40000,'teacher_inputs_seen':40000 if alpha else 0,'train_loss':.2,
            'train_accuracy':.8,'validation_loss':.2,'validation_accuracy':.9,'ema_validation_loss':.1,'ema_validation_accuracy':.91,
            'selected_variant':'ema','lr':.01,'epoch_seconds':10.,'elapsed_seconds':i*10.} for i in range(1,161)]

    def test_full_student_and_teacher_counters_and_no_skip_or_partial(self):
        for alpha in (0.,.5):
            config=trainer.fixed_config(alpha);h=self.history(alpha);total=trainer.validate_history(h,config)
            self.assertEqual(total['successful_updates'],50080);self.assertEqual(total['teacher_forward_passes'],50080 if alpha else 0)
            self.assertEqual(trainer.best_selection(h)[0],(1,'ema'))
            for key,value in [('successful_updates',312),('teacher_forward_passes',0 if alpha else 313),('skipped_updates',1),('teacher_inputs_seen',-1)]:
                bad=copy.deepcopy(h);bad[5][key]=value
                with self.subTest(alpha=alpha,key=key),self.assertRaises(ValueError):trainer.validate_history(bad,config)
            with self.assertRaises(ValueError):trainer.validate_history(h[:-1],config)

    def test_formal_and_pref_share_setup_epoch_and_step_no_resume_or_hidden_loader(self):
        tree=ast.parse(Path(trainer.__file__).read_text());functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        calls=[getattr(n.func,'id',None) for n in ast.walk(functions['train']) if isinstance(n,ast.Call)]
        self.assertIn('setup',calls);self.assertIn('start_epoch',calls);self.assertIn('train_step',calls)
        imports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
        self.assertFalse(any((n.module if isinstance(n,ast.ImportFrom) else n.names[0].name) in ('torch','numpy','runtime') for n in imports))
        text=Path(trainer.__file__).read_text();self.assertNotIn('load_kaggle_test',text);self.assertNotIn('clip_grad',text)
        self.assertFalse(any(isinstance(n,ast.Call) and getattr(n.func,'id',None) in ('exec','eval') for n in ast.walk(tree)))
        self.assertIn('validate_saved_state',functions);self.assertNotIn('resume',functions)


class TorchMathContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m=trainer._deps();cls.t=cls.m.t;cls.t.set_num_threads(1)

    def tiny(self):
        t=self.t
        return t.nn.Sequential(t.nn.Conv2d(3,4,3,padding=1,bias=False),t.nn.BatchNorm2d(4),t.nn.ReLU(),
            t.nn.AdaptiveAvgPool2d(1),t.nn.Flatten(),t.nn.Linear(4,10))

    def states(self,model,ema,opt):
        return (copy.deepcopy(model.state_dict()),copy.deepcopy(ema.state_dict()),copy.deepcopy(opt.state_dict()),self.m.kd.capture_parent_rng())

    def assert_states(self,left,right):
        for index in (0,1):
            for key in left[index]:self.assertTrue(self.t.equal(left[index][key],right[index][key]),key)
        self.assertEqual(left[2]['param_groups'],right[2]['param_groups']);self.assertEqual(set(left[2]['state']),set(right[2]['state']))
        for key in left[2]['state']:
            for field in left[2]['state'][key]:self.assertTrue(self.t.equal(left[2]['state'][key][field],right[2]['state'][key][field]))
        self.m.kd.assert_parent_rng_equal(left[3],right[3])

    def test_alpha0_matches_original_frozen_batch_AST_exactly(self):
        m=self.m;t=self.t;config={**trainer.fixed_config(0.),'batch_size':4,'mixup_probability':1.}
        tree=ast.parse((trainer.ROOT/'train.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
        epoch=next(n for n in fn.body if isinstance(n,ast.For) and getattr(n.target,'id',None)=='epoch')
        batch=next(n for n in epoch.body if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple));reference=compile(ast.Module(body=batch.body,type_ignores=[]),'frozen_reference','exec')
        m.runtime.seed_all(42);initial=copy.deepcopy(self.tiny().state_dict());results=[]
        for original in (True,False):
            m.runtime.seed_all(2026);model=self.tiny();model.load_state_dict(initial);ema=copy.deepcopy(model).eval().requires_grad_(False)
            opt=t.optim.SGD(model.parameters(),lr=.1,momentum=.9,weight_decay=.0005,nesterov=True)
            images=t.randint(0,256,(16,32,32,3),dtype=t.uint8);labels=t.arange(16)%10;order=t.randperm(16)
            ns=dict(torch=t,F=t.nn.functional,np=m.np,random=random,config=config,device=t.device('cpu'),model=model,ema=ema,opt=opt,
                model_params=list(model.parameters()),ema_params=list(ema.parameters()),train_x=images,train_y=labels,order=order,epoch=1,
                steps_per_epoch=313,total_steps=160*313,updates=0,correct=t.zeros(()),loss_sum=t.zeros(()),seen=0,schedule=m.core.schedule,augment=m.core.augment)
            loss=None;logits=None
            for index in range(4):
                if original:
                    ns.update(batch=index,start=index*4);exec(reference,ns);loss=ns['loss'].detach().clone();logits=ns['logits'].detach().clone()
                else:
                    result=trainer.train_step(model,ema,opt,images,labels,order,index,config,index,epoch=1,dtype=t.float32)
                    loss=result['loss'];logits=result['logits']
            results.append((self.states(model,ema,opt),loss,logits))
        self.assert_states(results[0][0],results[1][0]);self.assertTrue(t.equal(results[0][1],results[1][1]));self.assertTrue(t.equal(results[0][2],results[1][2]))

    def make_step(self,alpha=.5):
        m=self.m;t=self.t;m.runtime.seed_all(2026);model=self.tiny();ema=copy.deepcopy(model).eval().requires_grad_(False)
        opt=t.optim.SGD(model.parameters(),lr=.1,momentum=.9,weight_decay=.0005,nesterov=True)
        with m.kd.isolated_teacher_rng():teacher=self.tiny().eval().requires_grad_(False)
        snapshot=m.kd.teacher_state_snapshot(teacher,expected_state_tensors=len(teacher.state_dict()),expected_bn_modules=1,
            expected_parameters=sum(p.numel() for p in teacher.parameters()))
        images=t.randint(0,256,(8,32,32,3),dtype=t.uint8);labels=t.arange(8)%10;order=t.randperm(8)
        config={**trainer.fixed_config(alpha),'batch_size':4,'mixup_probability':1.}
        return model,ema,opt,teacher,snapshot,images,labels,order,config

    def test_KD_uses_exact_student_tensor_and_teacher_is_frozen(self):
        t=self.t;model,ema,opt,teacher,snapshot,images,labels,order,config=self.make_step()
        inputs=[];handles=[model.register_forward_pre_hook(lambda obj,args:inputs.append(args[0].detach().clone())),
            teacher.register_forward_pre_hook(lambda obj,args:inputs.append(args[0].detach().clone()))]
        result=trainer.train_step(model,ema,opt,images,labels,order,0,config,0,teacher=teacher,teacher_snapshot=snapshot,dtype=t.float32)
        for handle in handles:handle.remove()
        self.assertEqual(result['teacher_forward_passes'],1);self.assertTrue(t.equal(inputs[0],inputs[1]));self.assertIsNotNone(result['soft_loss'])
        self.m.kd.assert_teacher_immutable(teacher,snapshot);self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_nonfinite_after_SGD_rolls_back_BN_optimizer_EMA_gradients_and_allRNG(self):
        t=self.t;model,ema,opt,teacher,snapshot,images,labels,order,config=self.make_step(0.)
        # Establish real incoming momentum/grad state before testing an abort.
        trainer.train_step(model,ema,opt,images,labels,order,0,config,0,dtype=t.float32)
        before=self.states(model,ema,opt);before_grads=[p.grad.clone() for p in model.parameters()];original=opt.step
        def corrupt():
            value=original()
            with t.no_grad():next(model.parameters()).fill_(float('inf'))
            return value
        with patch.object(opt,'step',side_effect=corrupt),self.assertRaises(ValueError):
            trainer.train_step(model,ema,opt,images,labels,order,1,config,1,dtype=t.float32)
        self.assert_states(before,self.states(model,ema,opt))
        self.assertTrue(all(t.equal(p.grad,g) for p,g in zip(model.parameters(),before_grads)))

    def test_nonfinite_loss_fails_before_update_and_replays_RNG(self):
        t=self.t;model,ema,opt,teacher,snapshot,images,labels,order,config=self.make_step()
        before=self.states(model,ema,opt)
        with patch.object(self.m.kd,'distillation_components',return_value={'loss':t.tensor(float('nan'),requires_grad=True),'kd_loss':t.tensor(float('nan'))}),self.assertRaises(ValueError):
            trainer.train_step(model,ema,opt,images,labels,order,0,config,0,teacher=teacher,teacher_snapshot=snapshot,dtype=t.float32)
        self.assert_states(before,self.states(model,ema,opt))

    def test_teacher_fault_at_final_proof_restores_teacher_and_student(self):
        t=self.t;model,ema,opt,teacher,snapshot,images,labels,order,config=self.make_step(0.)
        before=self.states(model,ema,opt);original=self.m.kd.assert_teacher_immutable;calls=0
        def fault(obj,snap):
            nonlocal calls
            calls+=1
            if calls==2:
                with t.no_grad():next(obj.parameters()).add_(1.)
            return original(obj,snap)
        with patch.object(self.m.kd,'assert_teacher_immutable',side_effect=fault),self.assertRaises(ValueError):
            trainer.train_step(model,ema,opt,images,labels,order,0,config,0,teacher=teacher,teacher_snapshot=snapshot,dtype=t.float32)
        self.assert_states(before,self.states(model,ema,opt));original(teacher,snapshot)

    def test_typed_selected_last_optimizer_RNG_audit_with_explicit_synthetic_history(self):
        # Tensor boundary fixture only: it does not execute/claim160 epochs.
        m=self.m;t=self.t;config=trainer.fixed_config(0.);raw=(trainer.ROOT/'research/configs/wrn16_scratch_kd_control.json').read_bytes()
        teacher_ref={'schema':'synthetic_unit_teacher_reference_only','checkpoint_sha256':trainer.TEACHER_CHECKPOINT}
        with patch.object(trainer,'validate_teacher_reference',return_value=teacher_ref):
            marker=trainer.research_provenance(config,raw,teacher_ref,'fp32');source={**m.core.code_provenance(),'research_training':marker}
            model=m.build_model({k:config[k] for k in ('depth','widen_factor','dropout','num_classes')});ema=copy.deepcopy(model)
            opt=t.optim.SGD(model.parameters(),lr=.01,momentum=.9,weight_decay=.0005,nesterov=True)
            for parameter in model.parameters():parameter.grad=t.zeros_like(parameter)
            opt.step();history=PureContracts().history(0.);totals=trainer.validate_history(history,config)
            for obj in (model,ema):
                for layer in obj.modules():
                    if isinstance(layer,t.nn.BatchNorm2d):layer.num_batches_tracked.fill_(50080)
            selected=copy.deepcopy(model.state_dict())
            for key,value in selected.items():
                if key.endswith('num_batches_tracked'):value.fill_(313)
            common={'schema':trainer.SCHEMA,'config':config,'parameter_count':trainer.PARAMETERS,'role':'development_control','teacher_reference_sha256':trainer.json_sha(teacher_ref)}
            constructor={k:config[k] for k in ('depth','widen_factor','dropout','num_classes')}
            checkpoint={**common,'model':selected,'model_config':constructor,'epoch':1,'variant':'ema','validation_accuracy':.91,'validation_loss':.1,
                'provenance':source,'data_signature':trainer.DATA}
            metrics={**common,**totals,'status':'complete','epochs_completed':160,'training_examples':40000,'validation_examples':5000,'research_training':marker,
                'best_validation_accuracy':.91,'best_validation_loss':.1,'elapsed_seconds':1600.}
            rng=m.kd.capture_parent_rng()
            last={**common,**totals,'formal_training_state':True,'preflight_state':False,'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':opt.state_dict(),
                'model_config':constructor,'epoch':160,'updates':50080,'provenance':source,'history':history,'best_accuracy':.91,'best_loss':.1,'elapsed_seconds':1600.,
                'input_config_sha256':trainer.sha(trainer.ROOT/'research/configs/wrn16_scratch_kd_control.json'),'data_signature':trainer.DATA,
                'python_rng':rng['python'],'numpy_rng':rng['numpy'],'torch_rng':rng['torch_cpu'],'cuda_rng':rng['torch_cuda']}
            provenance={'schema':trainer.SCHEMA,'role':'development_control','code':source,'parameter_count':trainer.PARAMETERS,'data_signature':trainer.DATA,
                'normalization':{'mean':list(m.core.MEAN),'std':list(m.core.STD)}}
            env={'schema':trainer.SCHEMA,'role':'development_control','research_training':marker}
            result=benchmark.validate_training(checkpoint,metrics,history,provenance,env,last,raw,teacher_ref)
            self.assertTrue(result['actual_selected_last_optimizer_parent_RNG_tensors_audited'])
            for field in ('model','ema','optimizer','torch_rng'):
                broken=copy.deepcopy(last)
                if field in ('model','ema'):next(value for value in broken[field].values() if value.is_floating_point()).fill_(float('nan'))
                elif field=='optimizer':broken['optimizer']['state'].pop(0)
                else:broken['torch_rng']=t.tensor([1.])
                with self.subTest(field=field),self.assertRaises((ValueError,RuntimeError)):
                    benchmark.validate_training(checkpoint,metrics,history,provenance,env,broken,raw,teacher_ref)


if __name__=='__main__':unittest.main()
