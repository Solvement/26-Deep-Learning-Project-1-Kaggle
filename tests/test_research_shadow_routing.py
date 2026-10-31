"""Explicit shadow-cloud receipt routing; no GPU, training, uploads or real proof."""
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from research import run_diagnostic as diagnostic
from research import submit_candidate as submit

SCHEMA = "research_shadow_cloud_repeat_attestation_v1"
OLD_SCHEMA = "research_cloud_repeat_attestation_v1"
MODULE = "research.shadow_repeat_attestation"
ENTRY = "research/shadow_repeat_attestation.py"


class ShadowReceiptRoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/"research").mkdir()
        for name in ("shadow_repeat_attestation.py", "repeat_attestation.py", "submit_candidate.py"):
            (self.root/"research"/name).write_text("mock pinned helper")
        self.frozen = self.root/"frozen.json"
        self.attestation = self.root/"attestation.json"
        self.checkpoint = self.root/"best.pt"
        self.benchmark = self.root/"benchmark.json"
        self.evidence = self.root/"original_cloud_supervisor.json"
        for path in (self.frozen, self.checkpoint, self.benchmark, self.evidence):
            path.write_text("mock immutable evidence")
        self.attestation.write_text(json.dumps({"schema": SCHEMA}))
        self.checked = {"schema": SCHEMA, "attestation_path": str(self.attestation),
            "attestation_sha256": submit.sha256(self.attestation),
            "frozen_manifest_path": str(self.frozen), "frozen_manifest_sha256": submit.sha256(self.frozen),
            "checkpoint_path": str(self.checkpoint), "checkpoint_sha256": submit.sha256(self.checkpoint),
            "benchmark_path": str(self.benchmark), "benchmark_sha256": submit.sha256(self.benchmark),
            "evidence_sha256": {str(self.evidence): submit.sha256(self.evidence)},
            "source_sha256": {ENTRY: submit.sha256(self.root/ENTRY)}}
        self.module = ModuleType(MODULE)
        self.module.validate_attestation = Mock(return_value=self.checked)
        self.args = SimpleNamespace(mode="cloud-shadow-repeat-predict", manifest=self.frozen,
                                    attestation=self.attestation)
        self.delivery = self.root/"delivery.json"
        self.delivery.write_text(json.dumps({"frozen_manifest_path": str(self.frozen),
            "frozen_manifest_sha256": submit.sha256(self.frozen), "repeat": {
                "checkpoint_path": str(self.checkpoint), "benchmark_path": str(self.benchmark),
                "receipt_path": str(self.attestation)}}))

    def test_shadow_prediction_uses_its_own_deep_validator_and_pins_original_evidence(self):
        with patch.object(diagnostic, "ROOT", self.root), patch.dict(sys.modules, {MODULE: self.module}):
            command, output, inputs = diagnostic.build_command(self.args, self.root/"new", self.benchmark)
        self.module.validate_attestation.assert_called_once_with(self.attestation, manifest_path=self.frozen)
        self.assertEqual(command[1], ENTRY)
        self.assertEqual(command[command.index("--attestation")+1], str(self.attestation))
        self.assertEqual(command[command.index("--device")+1], "cuda")
        self.assertEqual(output, self.root/"new/delivery.json")
        self.assertTrue({self.frozen, self.attestation, self.evidence, self.root/ENTRY,
                         self.root/"research/submit_candidate.py"} <= set(inputs))

    def test_different_attestation_types_cannot_cross_prediction_modes(self):
        old = ModuleType("research.repeat_attestation")
        old.validate_attestation = Mock(return_value={**self.checked, "schema": SCHEMA})
        with patch.object(diagnostic, "ROOT", self.root), patch.dict(sys.modules, {
                MODULE: self.module, "research.repeat_attestation": old}):
            self.module.validate_attestation.return_value = {**self.checked, "schema": OLD_SCHEMA}
            with self.assertRaisesRegex(ValueError, "source-bound"):
                diagnostic.build_command(self.args, self.root/"new", self.benchmark)
            self.args.mode = "cloud-repeat-predict"
            with self.assertRaisesRegex(ValueError, "source-bound"):
                diagnostic.build_command(self.args, self.root/"new", self.benchmark)
        self.assertFalse((self.root/"new").exists())

    def test_shadow_prediction_rejects_missing_or_rebound_proof_before_a_command(self):
        with patch.object(diagnostic, "ROOT", self.root), patch.dict(sys.modules, {MODULE: self.module}):
            for field, bad in (("schema", "unknown"), ("attestation_sha256", "0"*64),
                               ("frozen_manifest_sha256", "0"*64), ("evidence_sha256", {})):
                self.module.validate_attestation.return_value = {**self.checked, field: bad}
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "source-bound"):
                    diagnostic.build_command(self.args, self.root/"new", self.benchmark)
            self.args.attestation = None
            with self.assertRaisesRegex(ValueError, "requires"):
                diagnostic.build_command(self.args, self.root/"new", self.benchmark)

    def test_upload_routes_shadow_schema_through_deep_validator_and_pins_its_source(self):
        with patch.dict(sys.modules, {MODULE: self.module}):
            result = submit.validate_v2_repeat_receipt(self.delivery, self.root)
        self.module.validate_attestation.assert_called_once_with(self.attestation, manifest_path=self.frozen)
        self.assertEqual(result["repeat_receipt_schema"], SCHEMA)
        self.assertEqual(result["repeat_receipt_sha256"], submit.sha256(self.attestation))
        self.assertEqual(result["repeat_validator_source_sha256"], submit.sha256(self.root/ENTRY))

    def test_upload_rejects_wrong_schema_or_changed_artifact_bindings(self):
        with patch.dict(sys.modules, {MODULE: self.module}):
            for field, bad in (("schema", OLD_SCHEMA), ("attestation_sha256", "0"*64),
                               ("checkpoint_sha256", "0"*64), ("benchmark_sha256", "0"*64),
                               ("frozen_manifest_sha256", "0"*64),
                               ("checkpoint_path", str(self.root/"renamed.pt"))):
                self.module.validate_attestation.return_value = {**self.checked, field: bad}
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "does not bind"):
                    submit.validate_v2_repeat_receipt(self.delivery, self.root)
            self.attestation.write_text(json.dumps({"schema": "unknown_shadow_attestation"}))
            with self.assertRaisesRegex(ValueError, "original native"):
                submit.validate_v2_repeat_receipt(self.delivery, self.root)


if __name__ == "__main__":
    unittest.main()
