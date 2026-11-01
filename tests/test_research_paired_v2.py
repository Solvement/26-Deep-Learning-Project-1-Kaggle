"""V2-only ResizeMix validator dispatch, method integrity and source bindings."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import runtime
import numpy as np
import torch
from model import build_model, count_parameters
from research import paired_compare as legacy
from research import paired_compare_v2 as paired
from research.benchmark_resizemix import validate_resizemix_training
from research.train_resizemix import research_provenance
from train import DEFAULT_CONFIG, code_provenance


class PairedV2Tests(unittest.TestCase):
    def fixture(self, folder, independent=True):
        folder.mkdir(parents=True)
        config = {**DEFAULT_CONFIG, "depth": 10, "widen_factor": 1, "epochs": 2,
                  "warmup_epochs": 0, "batch_size": 20000, "cutout": 0,
                  "mixup_alpha": 0., "mixup_probability": 0., "resizemix_probability": .5,
                  "resizemix_tau_min": .1, "resizemix_tau_max": .8}
        constructor = {key: config[key] for key in ("depth", "widen_factor", "dropout", "num_classes")}
        model = build_model(constructor)
        core = code_provenance()["source_sha256"]
        marker = research_provenance("fp32", config)
        signature = {"archive_sha256": "a"*64, "split_sha256": "b"*64}
        context = {"device_type": "cpu", "amp_dtype": "torch.float32", "torch_version": torch.__version__, "training_core_sha256": core}
        provenance = {"code": {"source_sha256": core, **({"research_training": marker} if independent else {})}}
        checkpoint = {"model": model.state_dict(), "model_config": constructor, "config": config, "epoch": 2,
            "variant": "model", "validation_accuracy": .8, "validation_loss": .5,
            "parameter_count": count_parameters(model), "data_signature": signature, "provenance": provenance["code"]}
        history = [{"epoch": epoch, "validation_accuracy": .8, "validation_loss": .5,
                    "ema_validation_accuracy": .7, "ema_validation_loss": .6,
                    "successful_updates": 2, "skipped_updates": 0, "resizemix_batches": 1, "grad_scaler_scale": 1.}
                   for epoch in (1, 2)]
        metrics = {"status": "complete", "epochs_completed": 2, "config": config,
            "training_examples": 40000, "validation_examples": 5000, "parameter_count": count_parameters(model),
            "best_validation_accuracy": .8, "best_validation_loss": .5,
            "successful_updates": 4, "skipped_updates": 0, "resizemix_batches": 2,
            **({"research_training": marker} if independent else {})}
        environment = {"resizemix_training_precision": "fp32", "grad_scaler_enabled": False}
        for name, value in (("metrics.json", metrics), ("history.json", history), ("provenance.json", provenance), ("environment.json", environment)):
            (folder/name).write_text(json.dumps(value))
        path, evidence = folder/"best.pt", folder/"benchmark.json"
        torch.save(checkpoint, path)
        cases = {case: {"examples": 5000, "input_pixel_sha256": "c"*64, "accuracy": .8} for case in paired.suite.CASES}
        document = {"schema": paired.suite.SCHEMA, "status": "complete", "examples": 5000,
            "training_examples": 40000, "bn_recalibrated": False, "confirmation_evaluated": False,
            "hidden_images_loaded": False, "external_data_loaded": False, "data_signature": signature,
            "evaluation_context": context, "benchmark_source_sha256": paired.suite.sha256(Path(paired.suite.__file__)),
            "source": {"source_sha256": core}, "stress_cases": list(paired.suite.CASES), "batch_size": 128,
            "config": config, "model_config": constructor, "checkpoint_epoch": 2, "checkpoint_variant": "model",
            "parameter_count": count_parameters(model), "policies": {"flip": {"cases": cases, "clean_accuracy": .8, "stress_accuracy": .8}}}
        integrity = validate_resizemix_training(checkpoint, metrics, history, provenance, environment) if independent else None
        self.seal(path, evidence, document, integrity)
        return path, evidence, signature, context, core, document, integrity

    def seal(self, path, evidence, document, integrity):
        document.update(checkpoint_sha256=paired.suite.sha256(path),
                        training_metrics_sha256=paired.suite.sha256(path.parent/"metrics.json"),
                        training_history_sha256=paired.suite.sha256(path.parent/"history.json"))
        core_path = evidence.with_name(evidence.stem+".core.json")
        core_path.write_text(json.dumps(document))
        result = copy.deepcopy(document)
        if integrity is not None:
            result["research_training_integrity"] = {**integrity,
                "validator_source_sha256": paired.suite.sha256(ROOT/"research/benchmark_resizemix.py"),
                "provenance_document_sha256": paired.suite.sha256(path.parent/"provenance.json"),
                "environment_document_sha256": paired.suite.sha256(path.parent/"environment.json"),
                "core_benchmark_sha256": paired.suite.sha256(core_path)}
        evidence.write_text(json.dumps(result))

    def load(self, fixture):
        path, evidence, signature, context, core, *_ = fixture
        return paired.load_experiment(path, evidence, signature, context, core,
                                      np.arange(40000), np.arange(40000, 45000))

    def test_dispatch_adds_actual_resizemix_validator_and_preserves_frozen_helper(self):
        verifier, path = paired.training_validator("research_resizemix_training_v1")
        self.assertIs(verifier, validate_resizemix_training)
        self.assertEqual(path, ROOT/"research/benchmark_resizemix.py")
        self.assertIs(paired.paired_bootstrap, legacy.paired_bootstrap)
        self.assertEqual(paired.PAIRED_HELPER_SOURCE_PATH, Path(legacy.__file__))
        for schema in ("research_shadow_training_v1", "research_sam_training_v1"):
            self.assertEqual(paired.training_validator(schema), legacy.training_validator(schema))
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            paired.training_validator("unknown_new_method")

    def test_real_resize_method_history_scaler_and_wrapped_core_evidence_validate(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = self.fixture(Path(folder)/"candidate")
            checkpoint, document = self.load(fixture)
            self.assertEqual(checkpoint["provenance"]["research_training"]["schema"], "research_resizemix_training_v1")
            self.assertEqual(document["research_training_integrity"]["successful_updates"], 4)

    def test_actual_method_source_step_counts_and_scaler_are_rechecked(self):
        for mutation in ("source", "step", "scaler", "schema"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                fixture = self.fixture(Path(folder)/"candidate")
                path, evidence, _, _, _, document, integrity = fixture
                if mutation in ("source", "schema"):
                    checkpoint = torch.load(path, weights_only=False)
                    marker = checkpoint["provenance"]["research_training"]
                    if mutation == "source": marker["source_sha256"]["research/resizemix.py"] = "d"*64
                    else: marker["schema"] = "unsupported_method"
                    torch.save(checkpoint, path)
                elif mutation == "step":
                    history = json.loads((path.parent/"history.json").read_text())
                    history[0]["successful_updates"] = 3
                    (path.parent/"history.json").write_text(json.dumps(history))
                else:
                    environment = json.loads((path.parent/"environment.json").read_text())
                    environment["grad_scaler_enabled"] = True
                    (path.parent/"environment.json").write_text(json.dumps(environment))
                self.seal(path, evidence, document, integrity)
                with self.assertRaises(ValueError):
                    self.load(fixture)

    def test_wrapper_validator_hash_core_copy_and_incomplete_benchmark_reject(self):
        for mutation in ("verifier", "core", "partial"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                fixture = self.fixture(Path(folder)/"candidate")
                _, evidence, *_ = fixture
                document = json.loads(evidence.read_text())
                if mutation == "verifier": document["research_training_integrity"]["validator_source_sha256"] = "d"*64
                elif mutation == "core": document["research_training_integrity"]["core_benchmark_sha256"] = "d"*64
                else: document["examples"] = 4999
                evidence.write_text(json.dumps(document))
                with self.assertRaises(ValueError):
                    self.load(fixture)

    def run_fixture(self, directory):
        candidate = self.fixture(directory/"candidate")
        baseline = self.fixture(directory/"baseline", independent=False)
        signature = candidate[2]
        (directory/"training_manifest.json").write_text(json.dumps({"split": {"indices_sha256": signature["split_sha256"]}}))
        document = candidate[5]
        document["baseline_sha256"] = paired.suite.sha256(baseline[1])
        self.seal(candidate[0], candidate[1], document, candidate[6])
        args = argparse.Namespace(candidate_checkpoint=candidate[0], candidate_benchmark=candidate[1],
            baseline_checkpoint=baseline[0], baseline_benchmark=baseline[1], output=directory/"paired_v2.json",
            policy="flip", data_dir=directory, device="cpu")
        arrays = (np.zeros((5000, 32, 32, 3), dtype=np.uint8), np.zeros(5000, dtype=np.int64),
                  np.arange(5000, 45000), np.arange(5000))
        return candidate, baseline, args, arrays

    def test_v2_output_binds_both_verifiers_and_real_5000_image_10000_draw_bootstrap(self):
        with tempfile.TemporaryDirectory() as folder:
            candidate, baseline, args, arrays = self.run_fixture(Path(folder))
            vectors = np.zeros((5000, 6), dtype=bool); vectors[:4000] = True
            with patch.object(paired, "load_training_arrays", return_value=arrays), \
                    patch.object(paired, "training_source", return_value={"archive_sha256": candidate[2]["archive_sha256"]}), \
                    patch.object(paired.suite, "BASELINE_CHECKPOINT_SHA", paired.suite.sha256(baseline[0])), \
                    patch.object(paired, "collect_correctness", side_effect=[vectors, vectors]), \
                    patch.object(paired, "seed_all"):
                result = paired.run(args)
            self.assertEqual(result["schema"], "exploratory_paired_image_bootstrap_v2")
            self.assertEqual(result["source_sha256"], paired.suite.sha256(Path(paired.__file__)))
            self.assertEqual(result["paired_helper_source_sha256"], paired.suite.sha256(Path(legacy.__file__)))
            self.assertEqual((result["bootstrap"]["original_images"], result["bootstrap"]["replicates"]), (5000, 10000))
            self.assertEqual(result["bootstrap"]["sampling_unit"], "original_image_with_all_five_stress_views")
            self.assertEqual(result["bootstrap"]["stress"]["percentile_ci95"], [0., 0.])
            self.assertFalse(result["automatic_promotion"])

    def test_change_during_cpu_bootstrap_rejects_before_any_result_write(self):
        with tempfile.TemporaryDirectory() as folder:
            candidate, baseline, args, arrays = self.run_fixture(Path(folder))
            def changed(*values, **kwargs):
                (candidate[0].parent/"metrics.json").write_text("{}"); return {"original_images": 5000}
            with patch.object(paired, "load_training_arrays", return_value=arrays), \
                    patch.object(paired, "training_source", return_value={"archive_sha256": candidate[2]["archive_sha256"]}), \
                    patch.object(paired.suite, "BASELINE_CHECKPOINT_SHA", paired.suite.sha256(baseline[0])), \
                    patch.object(paired, "collect_correctness", return_value=np.zeros((5000, 6), dtype=bool)), \
                    patch.object(paired, "paired_bootstrap", side_effect=changed), patch.object(paired, "seed_all"):
                with self.assertRaisesRegex(RuntimeError, "during paired v2 bootstrap"):
                    paired.run(args)
            self.assertFalse(args.output.exists())


if __name__ == "__main__":
    unittest.main()
