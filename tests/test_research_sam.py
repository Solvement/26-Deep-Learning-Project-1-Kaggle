"""Analytic and failure-path CPU tests for the stateless two-pass SAM helper."""
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import runtime
import torch
from research.sam import sam_step


class ScalarModel(torch.nn.Module):
    def __init__(self, value=1.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(float(value)))

    def forward(self, batch):
        return self.weight*batch


def assert_state_equal(test, left, right):
    if isinstance(left, torch.Tensor):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            assert_state_equal(test, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_state_equal(test, a, b)
    else:
        test.assertEqual(left, right)


class ResearchSAMTests(unittest.TestCase):
    def test_analytic_quadratic_update_uses_perturbed_gradient_at_original_weight(self):
        model = ScalarModel(1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        model.weight.grad = torch.tensor(99.)  # stale gradients must be cleared
        batch = torch.tensor([1.0])
        identities, weights = [], []
        def closure():
            identities.append(id(batch))
            weights.append(model.weight.item())
            return .5*model(batch).square().sum()
        evidence = sam_step(model, optimizer, closure, rho=.05)
        self.assertAlmostEqual(model.weight.item(), .895, places=6)
        self.assertEqual(identities, [id(batch), id(batch)])
        self.assertAlmostEqual(weights[0], 1.0)
        self.assertAlmostEqual(weights[1], 1.05, places=6)
        self.assertAlmostEqual(evidence["first_loss"], .5)
        self.assertAlmostEqual(evidence["second_loss"], .5*1.05**2, places=6)
        self.assertAlmostEqual(evidence["grad_norm"], 1.0)
        self.assertAlmostEqual(evidence["perturbation_norm"], .05, places=6)
        self.assertTrue(evidence["updated"])
        self.assertEqual(evidence["forward_backward_passes"], 2)
        self.assertIsNone(model.weight.grad)

    def test_perturbation_uses_one_global_l2_norm_across_parameters(self):
        model = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor([3.0])), torch.nn.Parameter(torch.tensor([4.0]))])
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        seen = []
        def closure():
            seen.append([parameter.item() for parameter in model])
            return sum(.5*parameter.square().sum() for parameter in model)
        evidence = sam_step(model, optimizer, closure, rho=.5)
        self.assertAlmostEqual(evidence["grad_norm"], 5.0)
        self.assertAlmostEqual(seen[1][0], 3.3, places=6)
        self.assertAlmostEqual(seen[1][1], 4.4, places=6)
        self.assertAlmostEqual(model[0].item(), 2.67, places=6)
        self.assertAlmostEqual(model[1].item(), 3.56, places=6)

    def test_batchnorm_updates_once_and_second_pass_still_uses_batch_statistics(self):
        for momentum in (.1, None):
            with self.subTest(momentum=momentum):
                model = torch.nn.Sequential(torch.nn.BatchNorm1d(2, momentum=momentum, affine=False), torch.nn.Linear(2, 1))
                model.train()
                optimizer = torch.optim.SGD(model.parameters(), lr=.01)
                batch = torch.tensor([[1., 4.], [2., 5.], [7., 11.], [8., 14.]])
                before = copy.deepcopy(model[0])
                before(batch)
                outputs, flags = [], []
                def hook(module, inputs, output):
                    outputs.append(output.detach().clone())
                    flags.append(module.training)
                handle = model[0].register_forward_hook(hook)
                try:
                    sam_step(model, optimizer, lambda: model(batch).square().mean())
                finally:
                    handle.remove()
                self.assertEqual(flags, [True, True])
                self.assertEqual(model[0].momentum, momentum)
                self.assertTrue(model[0].training)
                self.assertEqual(model[0].num_batches_tracked.item(), 1)
                self.assertTrue(torch.equal(model[0].running_mean, before.running_mean))
                self.assertTrue(torch.equal(model[0].running_var, before.running_var))
                # With batch statistics both passes center each feature near0;
                # running-stat inference on this nonzero-mean batch would not.
                for output in outputs:
                    self.assertTrue(torch.allclose(output.mean(0), torch.zeros(2), atol=1e-6))

    def test_eval_and_untracked_batchnorm_flags_are_preserved(self):
        for tracked in (True, False):
            with self.subTest(tracked=tracked):
                normalization = torch.nn.BatchNorm1d(2, momentum=None, track_running_stats=tracked)
                model = torch.nn.Sequential(normalization, torch.nn.Linear(2, 1))
                model.train()
                if tracked:
                    normalization.eval()  # respect a caller's explicitly frozen BN
                optimizer = torch.optim.SGD(model.parameters(), lr=.01)
                before_flag = normalization.training
                sam_step(model, optimizer, lambda: model(torch.tensor([[1., 2.], [3., 7.], [5., 6.]])).square().mean())
                self.assertEqual(normalization.training, before_flag)
                self.assertEqual(normalization.track_running_stats, tracked)
                self.assertIsNone(normalization.momentum)
                if tracked:
                    self.assertEqual(normalization.num_batches_tracked.item(), 0)
                else:
                    self.assertIsNone(normalization.running_mean)
                    self.assertIsNone(normalization.running_var)
                    self.assertIsNone(normalization.num_batches_tracked)

    def test_success_restores_original_weights_without_subtraction_roundoff_before_sgd(self):
        model = ScalarModel(.1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.)
        original = model.weight.detach().clone()
        sam_step(model, optimizer, lambda: model.weight.square(), rho=.2)
        self.assertTrue(torch.equal(model.weight, original))

    def test_second_pass_exception_restores_exact_weights_bn_and_optimizer_state(self):
        model = torch.nn.Sequential(torch.nn.BatchNorm1d(2), torch.nn.Linear(2, 1))
        optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
        batch = torch.tensor([[1., 2.], [3., 7.], [5., 6.]])
        model(batch).square().mean().backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        before_model, before_optimizer = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
        calls = 0
        def closure():
            nonlocal calls
            calls += 1
            value = model(batch).square().mean()
            if calls == 2:
                raise RuntimeError("injected second-pass error")
            return value
        with self.assertRaisesRegex(RuntimeError, "injected"):
            sam_step(model, optimizer, closure)
        assert_state_equal(self, model.state_dict(), before_model)
        assert_state_equal(self, optimizer.state_dict(), before_optimizer)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertEqual(model[0].momentum, .1)
        self.assertTrue(model[0].training)

    def test_nonfinite_losses_and_gradients_never_call_sgd(self):
        for failure in ("first_loss", "first_grad", "second_loss", "second_grad"):
            with self.subTest(failure=failure):
                model = ScalarModel(1)
                optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
                optimizer.state[model.weight]["momentum_buffer"] = torch.tensor(2.0)
                before = copy.deepcopy(optimizer.state_dict())
                calls = 0
                def closure():
                    nonlocal calls
                    calls += 1
                    stage = "first" if calls == 1 else "second"
                    if failure == stage+"_loss":
                        return model.weight*float("inf")
                    if failure == stage+"_grad":
                        # Finite value with an infinite local derivative.
                        return (model.weight-model.weight.detach()).sqrt()
                    return model.weight.square()
                with self.assertRaises(FloatingPointError):
                    sam_step(model, optimizer, closure)
                self.assertEqual(model.weight.item(), 1.0)
                self.assertIsNone(model.weight.grad)
                assert_state_equal(self, optimizer.state_dict(), before)

    def test_zero_gradients_and_unused_parameters_have_explicit_semantics(self):
        model = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor([1.])), torch.nn.Parameter(torch.tensor([2.]))])
        optimizer = torch.optim.SGD(model.parameters(), lr=.1, weight_decay=.1)
        evidence = sam_step(model, optimizer, lambda: (model[0]*0).sum())
        self.assertEqual(evidence["grad_norm"], 0)
        self.assertEqual(evidence["perturbation_norm"], 0)
        self.assertAlmostEqual(model[0].item(), .99, places=6)  # SGD weight decay
        self.assertEqual(model[1].item(), 2)  # None gradient: no decay/update
        unused_leaf = torch.tensor(1., requires_grad=True)
        before = [parameter.clone() for parameter in model]
        with self.assertRaisesRegex(ValueError, "no optimizer parameter"):
            sam_step(model, optimizer, lambda: unused_leaf.square())
        for parameter, saved in zip(model, before):
            self.assertTrue(torch.equal(parameter, saved))

    def test_fp16_scalers_and_wrong_optimizer_or_parameters_are_rejected(self):
        model = ScalarModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        for kwargs in ({"precision": "fp16"}, {"scaler": object()}, {"precision": "bf16"}, {"rho": 0}, {"rho": float("nan")}):
            with self.assertRaises(ValueError):
                sam_step(model, optimizer, lambda: model.weight.square(), **kwargs)
        with self.assertRaises(TypeError):
            sam_step(model, torch.optim.Adam(model.parameters()), lambda: model.weight.square())
        model.half()
        with self.assertRaisesRegex(ValueError, "FP32 master"):
            sam_step(model, optimizer, lambda: model.weight.square())

    def test_parameter_budget_and_parameter_objects_are_unchanged(self):
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        before = [(id(parameter), parameter.numel()) for parameter in model.parameters()]
        sam_step(model, optimizer, lambda: model(torch.ones(4, 3)).square().mean())
        self.assertEqual(before, [(id(parameter), parameter.numel()) for parameter in model.parameters()])


if __name__ == "__main__":
    unittest.main()
