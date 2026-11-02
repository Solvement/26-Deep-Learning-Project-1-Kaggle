"""Stdlib-only matched aggregate comparisons; no model/arrays/GPU/kernel."""
import ast
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research import incumbent_compare as comparison


class IncumbentComparisonTests(unittest.TestCase):
    def fixture(self, *, incumbent=False):
        config = {"depth": 34, "widen_factor": 3, "dropout": 0., "num_classes": 10,
                  "epochs": 300, "seed": 2026, "split_seed": 2026, "train_all": False}
        cases = {case: {"examples": 5000, "accuracy": .96 if case == "clean" else .8,
                        "loss": .12 if case == "clean" else .7,
                        "input_pixel_sha256": format(i+1, "064x")} for i, case in enumerate(comparison.CASES)}
        report = {"cases": cases, "clean_accuracy": .96, "clean_loss": .12, "stress_accuracy": .8}
        return {"schema": comparison.BENCHMARK_SCHEMA, "status": "complete", "examples": 5000,
                "training_examples": 40000, "parameter_count": 4166298, "batch_size": 128,
                "checkpoint_sha256": comparison.INCUMBENT_CHECKPOINT_SHA if incumbent else "a"*64,
                "training_metrics_sha256": "b"*64, "training_history_sha256": "c"*64,
                "checkpoint_epoch": 298, "checkpoint_variant": "model", "config": config,
                "model_config": {k: config[k] for k in ("depth", "widen_factor", "dropout", "num_classes")},
                "bn_recalibrated": False, "confirmation_evaluated": False,
                "hidden_images_loaded": False, "external_data_loaded": False,
                "data_signature": copy.deepcopy(comparison.DATA),
                "evaluation_context": copy.deepcopy(comparison.CONTEXT),
                "benchmark_source_sha256": comparison.BENCHMARK_SOURCE_SHA,
                "source": {"source_sha256": copy.deepcopy(comparison.CORE), "git_commit": "d"*40},
                "stress_cases": list(comparison.CASES), "elapsed_seconds": 31.,
                "policies": {policy: copy.deepcopy(report) for policy in comparison.POLICIES}}

    def compare(self, candidate):
        return comparison.compare_documents(candidate, self.fixture(incumbent=True))

    def test_exact_matched_summary_and_effective_hashes(self):
        result = self.compare(self.fixture())
        self.assertEqual(result["policy"], comparison.POLICY)
        self.assertEqual(result["same_policy_deltas"], dict(clean_accuracy=0., clean_nll=0., mean_stress_accuracy=0.))
        self.assertEqual(result["candidate"]["effective_config_sha256"], comparison.json_digest(self.fixture()["config"]))
        self.assertEqual(set(result["cases"]), set(comparison.CASES))
        self.assertFalse(result["automatic_promotion"])
        self.assertFalse(result["submission_authorized_by_this_report"])
        self.assertFalse(result["tensor_checkpoint_audit_performed"])
        self.assertFalse(result["bootstrap_performed"])
        self.assertNotIn("passed", result)

    def test_same_policy_not_old_flip_and_noise_regression_visible(self):
        candidate = self.fixture()
        report = candidate["policies"][comparison.POLICY]
        report["cases"]["noise_004"]["accuracy"] = .7
        report["cases"]["boxblur3"]["accuracy"] = .95
        report["stress_accuracy"] = sum(report["cases"][case]["accuracy"] for case in comparison.CASES[1:])/5
        incumbent = self.fixture(incumbent=True)
        incumbent["policies"]["flip"]["cases"]["clean"]["accuracy"] = .93
        incumbent["policies"]["flip"]["clean_accuracy"] = .93
        result = comparison.compare_documents(candidate, incumbent)
        self.assertEqual(result["same_policy_deltas"]["clean_accuracy"], 0.)
        self.assertGreater(result["same_policy_deltas"]["mean_stress_accuracy"], 0.)
        self.assertTrue(result["noise_accuracy_regressed"])
        self.assertEqual(result["stress_case_regressions"], ["noise_004"])
        self.assertAlmostEqual(result["cases"]["noise_004"]["accuracy_delta"], -.1)

    def test_partial_repeated_full_or_wrong_parameter_count_rejected(self):
        mutations = [("examples", 128), ("examples", True), ("training_examples", 50000),
                     ("parameter_count", 5000001), ("parameter_count", True), ("batch_size", 256),
                     ("status", "partial"), ("checkpoint_epoch", 301), ("checkpoint_variant", "swa")]
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                candidate = self.fixture(); candidate[key] = value
                with self.assertRaises(ValueError): self.compare(candidate)
        for key, value in (("seed", 2027), ("train_all", True), ("split_seed", 2027)):
            candidate = self.fixture(); candidate["config"][key] = value
            with self.assertRaises(ValueError): self.compare(candidate)

    def test_all_policies_and_cases_are_mandatory(self):
        candidate = self.fixture(); del candidate["policies"]["plain"]
        with self.assertRaises(ValueError): self.compare(candidate)
        for policy in comparison.POLICIES:
            for key, value in (("examples", 4999), ("accuracy", float("nan")), ("loss", -1.),
                               ("loss", float("inf")), ("input_pixel_sha256", "invalid")):
                candidate = self.fixture(); candidate["policies"][policy]["cases"]["clean"][key] = value
                with self.assertRaises(ValueError): self.compare(candidate)
        candidate = self.fixture(); del candidate["policies"]["plain"]["cases"]["boxblur3"]
        with self.assertRaises(ValueError): self.compare(candidate)

    def test_same_context_source_split_and_pixels_required(self):
        for key, value in (("device_type", "cpu"), ("amp_dtype", "torch.float16"), ("torch_version", "other")):
            candidate = self.fixture(); candidate["evaluation_context"][key] = value
            with self.assertRaises(ValueError): self.compare(candidate)
        for key in ("archive_sha256", "split_sha256"):
            candidate = self.fixture(); candidate["data_signature"][key] = "e"*64
            with self.assertRaises(ValueError): self.compare(candidate)
        candidate = self.fixture(); candidate["source"]["source_sha256"]["train.py"] = "e"*64
        with self.assertRaises(ValueError): self.compare(candidate)
        candidate = self.fixture(); candidate["benchmark_source_sha256"] = "e"*64
        with self.assertRaises(ValueError): self.compare(candidate)
        candidate = self.fixture(); candidate["policies"]["plain"]["cases"]["noise_004"]["input_pixel_sha256"] = "e"*64
        with self.assertRaises(ValueError): self.compare(candidate)

    def test_safety_flags_are_exact_booleans_and_scalar_types_are_finite(self):
        for key, value in (("confirmation_evaluated", True), ("hidden_images_loaded", True),
                           ("external_data_loaded", 0), ("bn_recalibrated", 0), ("elapsed_seconds", True)):
            candidate = self.fixture(); candidate[key] = value
            with self.assertRaises(ValueError): self.compare(candidate)
        candidate = self.fixture(); candidate["config"]["lr"] = float("inf")
        with self.assertRaises(ValueError): self.compare(candidate)

    def test_summary_nll_and_stress_must_match_cases(self):
        for key, value in (("clean_accuracy", .97), ("clean_loss", .13), ("stress_accuracy", .81)):
            candidate = self.fixture(); candidate["policies"][comparison.POLICY][key] = value
            with self.assertRaises(ValueError): self.compare(candidate)
        candidate = self.fixture(); candidate["policies"][comparison.POLICY]["cases"]["clean"]["accuracy"] = True
        with self.assertRaises(ValueError): self.compare(candidate)

    def test_incumbent_identity_and_method_sources_are_recorded(self):
        incumbent = self.fixture(incumbent=True); incumbent["checkpoint_sha256"] = "e"*64
        with self.assertRaises(ValueError): comparison.compare_documents(self.fixture(), incumbent)
        candidate = self.fixture()
        candidate["research_training_integrity"] = {"training": {"schema": "test_training_v1", "source_sha256": {"own.py": "f"*64}}}
        result = self.compare(candidate)
        self.assertEqual(result["candidate"]["recorded_method_source"]["source_sha256"], {"own.py": "f"*64})
        candidate["research_training_integrity"]["training"]["source_sha256"]["own.py"] = "invalid"
        with self.assertRaises(ValueError): self.compare(candidate)

    def test_config_hash_is_key_order_invariant_and_not_claimed_raw_file_hash(self):
        self.assertEqual(comparison.json_digest({"a": 1, "b": 2}), comparison.json_digest({"b": 2, "a": 1}))
        self.assertNotEqual(comparison.json_digest({"a": 1}), comparison.digest(b'{ "a": 1 }'))

    def test_duplicate_json_fields_and_nonfinite_constants_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"benchmark.json"
            for raw in ('{"schema": "a", "schema": "b"}', '{"loss": NaN}', '[]'):
                path.write_text(raw)
                with self.assertRaises(ValueError): comparison.read_document(path)

    def test_fresh_only_run_pins_raw_incumbent_and_writes_no_private_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp); candidate = folder/"candidate.json"; incumbent = folder/"incumbent.json"; output = folder/"output.json"
            candidate.write_text(json.dumps(self.fixture())); incumbent.write_text(json.dumps(self.fixture(incumbent=True)))
            checksum = comparison.digest(incumbent.read_bytes())
            with patch.object(comparison, "INCUMBENT_BENCHMARK_SHA", checksum):
                result = comparison.run(candidate, output, incumbent)
                self.assertEqual(result, json.loads(output.read_text()))
                self.assertNotIn(str(folder), output.read_text())
                with self.assertRaises(FileExistsError): comparison.run(candidate, output, incumbent)
            output.unlink()
            with self.assertRaises(ValueError): comparison.run(candidate, output, incumbent)
            self.assertFalse(output.exists())

    def test_source_or_evidence_changes_prevent_output(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp); candidate = folder/"candidate.json"; incumbent = folder/"incumbent.json"; output = folder/"output.json"
            candidate.write_text(json.dumps(self.fixture())); incumbent.write_text(json.dumps(self.fixture(incumbent=True)))
            original = comparison.compare_documents
            def change(a, b):
                result = original(a, b); candidate.write_text("{}"); return result
            with patch.object(comparison, "INCUMBENT_BENCHMARK_SHA", comparison.digest(incumbent.read_bytes())), patch.object(comparison, "compare_documents", side_effect=change):
                with self.assertRaises(RuntimeError): comparison.run(candidate, output, incumbent)
            self.assertFalse(output.exists())

    def test_no_args_is_read_only_and_imports_are_stdlib_only(self):
        with patch.object(comparison, "run") as run, redirect_stdout(io.StringIO()):
            self.assertIsNone(comparison.main([])); run.assert_not_called()
        tree = ast.parse(Path(comparison.__file__).read_text())
        imports = {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        imports |= {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        self.assertFalse(imports & {"torch", "numpy", "runtime", "nbclient", "data", "model", "train"})


if __name__ == "__main__":
    unittest.main()
