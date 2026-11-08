"""CPU architecture mathematics, residual placement, common initialization and I/O."""
import copy
import io
import unittest
from unittest.mock import patch

import runtime
import torch
from torch import nn
import model as frozen
from research import se_model as se


class ConstantScale(nn.Module):
    def __init__(self, value): super().__init__(); self.value = value
    def forward(self, inputs): return inputs*self.value


class SEModelTests(unittest.TestCase):
    def test_nine_exact_gates_total_trainable_count_and_no_hidden_model_copy(self):
        candidate = se.build_model()
        gates = [module for module in candidate.modules() if isinstance(module, se.SqueezeExcitation)]
        self.assertEqual(len(gates),9)
        self.assertEqual([(gate.channels,gate.hidden_channels) for gate in gates], [(64,4)]*3+[(128,8)]*3+[(256,16)]*3)
        self.assertTrue(all(gate.reduce.bias is not None and gate.expand.bias is not None for gate in gates))
        self.assertEqual(sum(se.count_parameters(gate) for gate in gates),33684)
        self.assertEqual((candidate.count_parameters(),candidate.count_parameters(True)),(4332654,4332654))
        self.assertEqual(len(list(candidate.parameters())),len({id(p) for p in candidate.parameters()}))
        self.assertTrue(all(p.requires_grad for p in candidate.parameters()))

    def test_same_seed_preserves_every_frozen_shared_state_key_and_initial_tensor(self):
        torch.manual_seed(2738)
        baseline = frozen.build_model()
        baseline_rng = torch.get_rng_state().clone()
        torch.manual_seed(2738)
        candidate = se.build_model()
        current = candidate.state_dict()
        for name,value in baseline.state_dict().items():
            self.assertIn(name,current)
            self.assertTrue(torch.equal(value,current[name]),name)
        new_keys = set(current)-set(baseline.state_dict())
        self.assertEqual(len(new_keys),36)  # Four bias/weight tensors for each gate.
        self.assertTrue(all(".se." in name for name in new_keys))
        self.assertFalse(torch.equal(baseline_rng,torch.get_rng_state()))
        torch.manual_seed(2738)
        repeated = se.build_model()
        self.assertTrue(all(torch.equal(value,repeated.state_dict()[key]) for key,value in current.items()))

    def test_gate_matches_explicit_spatial_mean_two_bias_fc_sigmoid_math(self):
        gate = se.SqueezeExcitation(4,2)
        with torch.no_grad():
            gate.reduce.weight.fill_(.1); gate.reduce.bias.fill_(.2)
            gate.expand.weight.fill_(.3); gate.expand.bias.fill_(.4)
        inputs = torch.arange(2*4*3*5,dtype=torch.float32).reshape(2,4,3,5)/30
        means = inputs.mean((2,3))
        hidden = torch.clamp(means@gate.reduce.weight.t()+gate.reduce.bias,min=0)
        weights = torch.sigmoid(hidden@gate.expand.weight.t()+gate.expand.bias)
        self.assertTrue(torch.equal(gate(inputs),inputs*weights[:,:,None,None]))
        self.assertTrue(((weights>0)&(weights<1)).all())
        inputs.requires_grad_(True)
        gate(inputs).sum().backward()
        self.assertTrue(inputs.grad.isfinite().all())
        self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() and p.grad.abs().sum()>0 for p in gate.parameters()))

    def test_gate_is_after_conv2_and_never_gates_identity_or_preactivated_projection(self):
        for in_channels,out_channels,stride in ((16,16,1),(16,32,2)):
            with self.subTest(projection=stride==2):
                original = frozen.PreActWideBlock(in_channels,out_channels,stride,0.).eval()
                candidate = se.SEPreActWideBlock(original).eval()
                inputs = torch.randn(2,in_channels,8,8)
                candidate.se = ConstantScale(1.)
                self.assertTrue(torch.equal(candidate(inputs),original(inputs)))
                candidate.se = ConstantScale(0.)
                shortcut = inputs if original.projection is None else original.projection(original.relu1(original.bn1(inputs)))
                self.assertTrue(torch.equal(candidate(inputs),shortcut))
                candidate.se = ConstantScale(.25)
                activated = original.relu1(original.bn1(inputs))
                residual = original.conv2(original.dropout(original.relu2(original.bn2(original.conv1(activated)))))
                self.assertTrue(torch.equal(candidate(inputs),residual*.25+shortcut))

    def test_cpu_forward_backward_updates_gate_parameters_without_cuda_or_downloads(self):
        with patch.object(torch.cuda,"is_available",side_effect=AssertionError("CPU only")):
            candidate = se.build_model().train()
            inputs,labels = torch.randn(2,3,32,32),torch.tensor([1,7])
            gate_parameters = [p for name,p in candidate.named_parameters() if ".se." in name]
            before = [p.detach().clone() for p in gate_parameters]
            logits = candidate(inputs)
            self.assertEqual(tuple(logits.shape),(2,10))
            loss = nn.functional.cross_entropy(logits,labels)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in candidate.parameters()))
            torch.optim.SGD(candidate.parameters(),lr=.1,momentum=.9).step()
            self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,gate_parameters)))

    def test_typed_config_and_state_round_trip_exactly_without_legacy_builder_fallback(self):
        candidate = se.build_model().eval()
        stream = io.BytesIO()
        torch.save({"model_config":candidate.export_config(),"model":candidate.state_dict()},stream)
        stream.seek(0)
        saved = torch.load(stream,weights_only=True,map_location="cpu")
        restored = se.build_model(saved["model_config"]).eval()
        restored.load_state_dict(saved["model"],strict=True)
        inputs = torch.randn(2,3,32,32)
        with torch.no_grad(): self.assertTrue(torch.equal(candidate(inputs),restored(inputs)))
        self.assertEqual(restored.export_config()["architecture"],se.ARCHITECTURE)
        with self.assertRaises(TypeError): frozen.build_model(saved["model_config"])
        with self.assertRaises(ValueError): se.build_model({k:v for k,v in saved["model_config"].items() if k!="architecture"})
        corrupt = copy.deepcopy(saved["model"])
        corrupt.pop("stage1.0.se.reduce.bias")
        with self.assertRaises(RuntimeError): restored.load_state_dict(corrupt,strict=True)

    def test_unsupported_recipes_and_pretrained_arguments_reject_before_base_construction(self):
        for change in ({"architecture":"wrn"},{"depth":34},{"widen_factor":5},{"num_classes":100},
                {"dropout":.2},{"dropout":True},{"reduction":8},{"reduction":True},{"pretrained":True}):
            with self.subTest(change=change), patch.object(se,"FrozenWideResNet",side_effect=AssertionError("must reject first")):
                with self.assertRaises(ValueError): se.build_model(**change)
        for channels,reduction in ((0,16),(15,16),(64,0),(True,16)):
            with self.assertRaises(ValueError): se.SqueezeExcitation(channels,reduction)

    def test_original_classes_and_builder_are_unchanged_and_new_source_is_explicit(self):
        builder,forward = frozen.build_model,frozen.PreActWideBlock.forward
        _ = se.build_model()
        self.assertIs(frozen.build_model,builder)
        self.assertIs(frozen.PreActWideBlock.forward,forward)
        original = frozen.build_model()
        self.assertTrue(all(type(block) is frozen.PreActWideBlock for stage in (original.stage1,original.stage2,original.stage3) for block in stage))
        provenance = se.source_provenance()
        self.assertEqual(provenance["architecture"],se.ARCHITECTURE)
        self.assertEqual(set(provenance["source_sha256"]),{"model.py","runtime.py","research/se_model.py"})
        self.assertTrue(all(len(value)==64 for value in provenance["source_sha256"].values()))


if __name__ == "__main__": unittest.main()
