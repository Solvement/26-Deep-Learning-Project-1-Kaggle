"""Pure local API tests; small real CPU tensor/RNG contracts run in Linux CI."""
from contextlib import redirect_stdout
import io
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import unittest
from unittest.mock import patch

from research import scratch_kd as kd


class PureContracts(unittest.TestCase):
    def test_import_and_alpha_zero_do_not_import_numerical_packages(self):
        code=("import sys;from research import scratch_kd as k;hard=object();"
            "assert k.distillation_loss(object(),object(),hard,4.,0.) is hard;"
            "c=k.distillation_components(object(),object(),hard,4.,0.);"
            "assert c['loss'] is hard and c['hard_loss'] is hard and c['kd_loss'] is None;"
            "assert not {'torch','numpy','PIL','runtime'}&set(sys.modules)")
        subprocess.check_output([sys.executable,"-c",code],cwd=Path(kd.__file__).resolve().parents[1],text=True)

    def test_alpha_zero_is_exact_hard_object_and_never_inspects_teacher(self):
        class Forbidden:
            def __getattribute__(self,name):raise AssertionError("No tensor/teacher inspection on alpha0")
        hard=object()
        self.assertIs(kd.distillation_loss(Forbidden(),Forbidden(),hard,4,0),hard)
        components=kd.distillation_components(Forbidden(),Forbidden(),hard,4,0)
        self.assertIs(components["loss"],hard);self.assertIs(components["hard_loss"],hard)
        self.assertIsNone(components["kd_loss"])

    def test_scalar_boundaries_reject_bool_nan_inf_and_invalid_ranges_without_torch(self):
        for temperature,alpha in ((True,.5),(0,.5),(-1,.5),(float("nan"),.5),(float("inf"),.5),
            (4,True),(4,-.1),(4,1.1),(4,float("nan")),(4,float("inf"))):
            with self.subTest(temperature=temperature,alpha=alpha),self.assertRaises(ValueError):
                kd.distillation_loss(None,None,object(),temperature,alpha)
        for lam in (True,-.1,1.1,float("nan")):
            with self.subTest(lam=lam),self.assertRaises(ValueError):kd.mixed_hard_loss(None,None,None,lam)

    def test_known_teacher_structure_contract_is_explicit(self):
        self.assertEqual((kd.TEACHER_PARAMETERS,kd.TEACHER_STATE_TENSORS,kd.TEACHER_BN_MODULES),(4166298,191,31))


@unittest.skipUnless(os.environ.get("CI")=="true","Real small CPU tensor/RNG tests run only in existing Linux CI")
class TorchContracts(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import torch
        self.t,self.np=torch,np
        self.available=patch.object(torch.cuda,"is_available",return_value=False)
        self.available.start();self.addCleanup(self.available.stop)
        self.threads=torch.get_num_threads();torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads,self.threads)
        self.prior=kd.capture_parent_rng();self.addCleanup(kd.restore_parent_rng,self.prior)

    def fixture(self,batch=3):
        t=self.t
        values=t.arange(batch*10,dtype=t.float32).reshape(batch,10)
        student=(values.mul(.19).sin()*3.1).requires_grad_(True)
        teacher=(values.mul(.37).cos()*2.3+.4).requires_grad_(True)
        labels=t.arange(batch)%10
        hard=t.nn.functional.cross_entropy(student,labels)
        return student,teacher,labels,hard

    def test_fp64_analytic_forward_kl_loss_and_gradients_for_batch1_and64(self):
        t=self.t
        for batch in (1,64):
            student,teacher,labels,hard=self.fixture(batch)
            temperature,alpha=4.,.5
            parts=kd.distillation_components(student,teacher,hard,temperature,alpha)
            actual_grad=t.autograd.grad(parts["loss"],student)[0]
            s,p=student.detach().double(),teacher.detach().double()
            log_s=t.log_softmax(s/temperature,1);log_p=t.log_softmax(p/temperature,1)
            reference_kl=(log_p.exp()*(log_p-log_s)).sum()/batch*temperature**2
            reference_loss=(1-alpha)*hard.detach().double()+alpha*reference_kl
            one_hot=t.nn.functional.one_hot(labels,10).double()
            reference_grad=(1-alpha)*(s.softmax(1)-one_hot)/batch+alpha*temperature*(log_s.exp()-log_p.exp())/batch
            self.assertEqual(parts["kd_loss"].dtype,t.float32)
            t.testing.assert_close(parts["loss"].double(),reference_loss,rtol=2e-6,atol=2e-6)
            t.testing.assert_close(actual_grad.double(),reference_grad,rtol=2e-5,atol=2e-7)
            self.assertIsNone(teacher.grad)

    def test_direction_temperature_square_and_batchmean_not_classmean(self):
        t=self.t
        student=t.tensor([[9.,-1.,-2.,0.,1.,-3.,.5,.8,-.8,2.],[0.,3.,-.5,1.,2.,-3.,.1,.2,.3,.4]],requires_grad=True)
        teacher=t.tensor([[-2.,3.,1.,0.,-.5,.8,1.,-1.,2.,.5],[4.,-1.,2.,0.,-.5,1.,-.5,3.,1.,2.]])
        temp=2.;hard=t.tensor(.8,requires_grad=True)
        actual=kd.distillation_components(student,teacher,hard,temp,1.)["kd_loss"]
        ls=t.log_softmax(student.double()/temp,1);lp=t.log_softmax(teacher.double()/temp,1)
        forward=(lp.exp()*(lp-ls)).sum()/len(student)*temp**2
        reverse=(ls.exp()*(ls-lp)).sum()/len(student)*temp**2
        t.testing.assert_close(actual.double(),forward,rtol=2e-6,atol=2e-6)
        self.assertGreater(abs(actual.item()-reverse.item()),.01)
        self.assertGreater(abs(actual.item()-forward.item()/10),.1)
        self.assertGreater(abs(actual.item()-forward.item()/temp**2),.1)

    def test_shift_invariance_and_identical_logits(self):
        t=self.t;student,teacher,_,hard=self.fixture(4)
        base=kd.distillation_components(student,teacher,hard,4.,.5)["kd_loss"]
        offsets=t.arange(4,dtype=t.float32).reshape(-1,1)*13.25
        shifted=kd.distillation_components(student+offsets,teacher-offsets,hard,4.,.5)["kd_loss"]
        t.testing.assert_close(base,shifted,rtol=2e-5,atol=2e-6)
        equal=kd.distillation_components(student,student.detach().clone(),hard,4.,.5)
        self.assertEqual(equal["kd_loss"].item(),0.)
        grad=t.autograd.grad(equal["kd_loss"],student)[0]
        t.testing.assert_close(grad,t.zeros_like(grad),rtol=0,atol=1e-7)

    def test_original_mixup_hard_loss_order_and_alpha0_gradients_are_exact(self):
        t=self.t
        for mixed in (False,True):
            a,_,labels,_=self.fixture(5);b=a.detach().clone().requires_grad_(True)
            paired=labels.flip(0) if mixed else None;lam=.37 if mixed else 1.
            original=t.nn.functional.cross_entropy(a,labels,label_smoothing=0.)*lam
            if mixed:original=original+t.nn.functional.cross_entropy(a,paired,label_smoothing=0.)*(1-lam)
            hard=kd.mixed_hard_loss(b,labels,paired,lam,label_smoothing=0.)
            actual=kd.distillation_loss(b,None,hard,4.,0.)
            self.assertIs(actual,hard);self.assertTrue(t.equal(original,actual))
            ga=t.autograd.grad(original,a)[0];gb=t.autograd.grad(actual,b)[0]
            self.assertTrue(t.equal(ga,gb))

    def test_teacher_targets_detach_and_hard_loss_object_is_preserved(self):
        t=self.t;student,teacher,_,hard=self.fixture(2)
        result=kd.distillation_components(student,teacher,hard,4.,.5)
        self.assertIs(result["hard_loss"],hard);result["loss"].backward()
        self.assertIsNotNone(student.grad);self.assertIsNone(teacher.grad)

    def test_shapes_nan_inf_and_unrepresentable_fp32_values_fail_closed(self):
        t=self.t;student,teacher,_,hard=self.fixture(2)
        for a,b,c in ((student[:,:9],teacher[:,:9],hard),(student[:0],teacher[:0],hard),
            (student,teacher[:1],hard),(student.long(),teacher,hard),(student,teacher,hard.reshape(1)),
            (student,teacher,t.tensor(float("nan"))),(student,teacher,t.tensor(float("inf")))):
            with self.assertRaises(ValueError):kd.distillation_loss(a,b,c,4.,1.)
        for value in (float("nan"),float("inf"),-float("inf")):
            a=student.detach().clone();a[0,0]=value
            with self.assertRaises(ValueError):kd.distillation_loss(a,teacher,hard,4.,.5)
            b=teacher.detach().clone();b[0,0]=value
            with self.assertRaises(ValueError):kd.distillation_loss(student,b,hard,4.,.5)
        huge=t.full((2,10),1e300,dtype=t.float64)
        with self.assertRaises(ValueError):kd.distillation_loss(huge,teacher,hard,4.,.5)
        overflow=t.zeros((2,10));overflow[:,0]=3e38;overflow[:,1]=-3e38
        with self.assertRaises(ValueError):kd.distillation_loss(overflow,teacher,hard,1.,.5)
        for temperature in (1e-300,1e300):
            with self.assertRaises(ValueError):kd.distillation_loss(student,teacher,hard,temperature,.5)

    def test_large_but_representable_finite_logits_remain_finite(self):
        t=self.t
        s=t.arange(20,dtype=t.float32).reshape(2,10).mul(1e9).requires_grad_(True)
        teacher=s.detach().flip(1)
        loss=kd.distillation_loss(s,teacher,t.tensor(.5),4.,.5)
        self.assertTrue(t.isfinite(loss));self.assertTrue(t.isfinite(t.autograd.grad(loss,s)[0]).all())

    def test_teacher_construction_rng_isolation_and_exception_restore(self):
        t,np=self.t,self.np
        for raises in (False,True):
            before=kd.capture_parent_rng()
            try:
                with kd.isolated_teacher_rng():
                    random.random();np.random.rand(3);t.rand(4);t.nn.Linear(3,10)
                    if raises:raise RuntimeError("teacher load fixture failure")
            except RuntimeError:
                self.assertTrue(raises)
            kd.assert_parent_rng_equal(before,kd.capture_parent_rng())

    def tiny_teacher(self,behaviour=None):
        t,np=self.t,self.np
        class Teacher(t.nn.Module):
            def __init__(self):
                super().__init__();self.bn=t.nn.BatchNorm2d(3);self.head=t.nn.Linear(3,10);self.seen_input=None
            def forward(self,value):
                self.seen_input=id(value)
                random.random();np.random.rand();t.rand(2)
                output=self.head(self.bn(value).mean((2,3)))
                if behaviour=="mutate":self.bn.num_batches_tracked.add_(1)
                if behaviour=="nan":self.head.weight[0,0]=float("nan")
                if behaviour=="raise":raise RuntimeError("teacher forward fixture failure")
                return output
        teacher=Teacher().eval().requires_grad_(False)
        snapshot=kd.teacher_state_snapshot(teacher,expected_state_tensors=7,expected_bn_modules=1,expected_parameters=46)
        return teacher,snapshot

    def test_teacher_forward_is_same_input_nograd_full_bn_immutable_and_rng_isolated(self):
        t=self.t;teacher,snapshot=self.tiny_teacher()
        value=t.arange(2*3*4*4,dtype=t.float32).reshape(2,3,4,4).requires_grad_(True)
        before=kd.capture_parent_rng();output=kd.frozen_teacher_forward(teacher,value,snapshot)
        self.assertEqual(teacher.seen_input,id(value));self.assertEqual(output.shape,(2,10));self.assertFalse(output.requires_grad)
        self.assertIsNone(value.grad);self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        kd.assert_parent_rng_equal(before,kd.capture_parent_rng());kd.assert_teacher_immutable(teacher,snapshot)
        self.assertEqual(teacher.bn.num_batches_tracked.item(),0)

    def test_teacher_failure_restores_all_state_modes_grad_flags_and_rng(self):
        t=self.t
        for behaviour in ("mutate","nan","raise"):
            teacher,snapshot=self.tiny_teacher(behaviour);before=kd.capture_parent_rng()
            value=t.zeros((1,3,4,4))
            with self.subTest(behaviour=behaviour),self.assertRaises((ValueError,RuntimeError)):
                kd.frozen_teacher_forward(teacher,value,snapshot)
            kd.assert_teacher_immutable(teacher,snapshot);kd.assert_parent_rng_equal(before,kd.capture_parent_rng())
        teacher,snapshot=self.tiny_teacher();teacher.train();next(teacher.parameters()).requires_grad_(True)
        with self.assertRaises(ValueError):kd.frozen_teacher_forward(teacher,t.zeros((1,3,4,4)),snapshot)
        kd.assert_teacher_immutable(teacher,snapshot)

    def test_all_cuda_generators_restore_with_mocked_cpu_state_tensors(self):
        t=self.t
        states=[t.tensor([1,2,3],dtype=t.uint8),t.tensor([4,5,6],dtype=t.uint8)]
        def get():return [state.clone() for state in states]
        def put(values):states[:]=[state.clone() for state in values]
        with patch.object(t.cuda,"is_available",return_value=True),patch.object(t.cuda,"get_rng_state_all",side_effect=get),patch.object(t.cuda,"set_rng_state_all",side_effect=put):
            before=kd.capture_parent_rng()
            with self.assertRaises(RuntimeError):
                with kd.isolated_teacher_rng():
                    states[0].add_(1);states[1].add_(2);t.rand(2);random.random();self.np.random.rand()
                    raise RuntimeError("mocked CUDA restore fixture")
            kd.assert_parent_rng_equal(before,kd.capture_parent_rng())


if __name__=="__main__":unittest.main()
