"""Paired-image uncertainty and fail-closed evidence checks; CPU synthetic inputs."""
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
from research import paired_compare as paired


class PairedEvidenceTests(unittest.TestCase):
    def test_unknown_training_extension_rejects_without_falling_back_to_shadow(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            paired.training_validator("unverified_new_training")
        validator, path = paired.training_validator("research_shadow_training_v1")
        self.assertIs(validator, paired.validate_shadow)
        self.assertEqual(path.name, "benchmark_shadow.py")

    def fixture(self):
        config = {"train_all": False, "depth": 22, "widen_factor": 4, "dropout": 0., "num_classes": 10}
        constructor = {key: value for key, value in config.items() if key != "train_all"}
        checkpoint = {"config": config, "model_config": constructor, "epoch": 151,
                      "variant": "model", "parameter_count": 4298970}
        fingerprints = {"checkpoint_sha256": "weights", "training_metrics_sha256": "metrics", "training_history_sha256": "history"}
        signature, context, core_source = {"split": "official"}, {"device_type": "cpu"}, {"train.py": "original"}
        cases = {name: {"examples": 5000, "input_pixel_sha256": "a"*64, "accuracy": .8} for name in paired.suite.CASES}
        document = {"schema": paired.suite.SCHEMA, "status": "complete", "examples": 5000,
                    "training_examples": 40000, "bn_recalibrated": False,
                    "confirmation_evaluated": False, "hidden_images_loaded": False, "external_data_loaded": False,
                    "data_signature": signature, "evaluation_context": context, "benchmark_source_sha256": "current",
                    "source": {"source_sha256": core_source}, "stress_cases": list(paired.suite.CASES),
                    "batch_size": 128, "config": config, "model_config": constructor, "checkpoint_epoch": 151,
                    "checkpoint_variant": "model", "parameter_count": 4298970,
                    "policies": {"flip": {"cases": cases, "clean_accuracy": .8, "stress_accuracy": .8}}, **fingerprints}
        return document, checkpoint, fingerprints, signature, context, core_source, "current"

    def test_identical_checkpoints_have_exact_zero_delta_and_interval(self):
        values = np.random.default_rng(7).integers(0, 2, size=(5000, 6), dtype=np.int8)
        result = paired.paired_bootstrap(values, values)
        for name in ("clean", "stress"):
            self.assertEqual(result[name]["delta_accuracy"], 0)
            self.assertEqual(result[name]["percentile_ci95"], [0, 0])
            self.assertEqual(result[name]["candidate_only_correct"], 0)
            self.assertEqual(result[name]["baseline_only_correct"], 0)
        self.assertEqual(result["replicates"], 10000)
        self.assertEqual(result["seed"], 20261031)

    def test_all_stress_views_follow_the_same_original_image_draw(self):
        # Perfectly correlated views must have the same CI as the single clean
        # view. Sampling 5*N independent stress observations shrinks it falsely.
        candidate = np.repeat(np.array([[1], [1], [1], [0], [0], [0], [1], [0]]), 6, axis=1)
        baseline = 1-candidate
        result = paired.paired_bootstrap(candidate, baseline, replicates=2000, chunk_size=17)
        self.assertEqual(result["clean"]["percentile_ci95"], result["stress"]["percentile_ci95"])
        self.assertEqual(result["stress"]["candidate_only_correct"], 20)
        self.assertEqual(result["stress"]["original_image_net_wins"], 4)
        self.assertLess(result["stress"]["percentile_ci95"][0], 0)
        self.assertGreater(result["stress"]["percentile_ci95"][1], 0)

    def test_matched_pairs_counts_and_seed_are_chunk_invariant(self):
        candidate = np.repeat(np.array([[1], [1], [0], [0]]), 6, axis=1)
        baseline = np.repeat(np.array([[1], [0], [1], [0]]), 6, axis=1)
        a = paired.paired_bootstrap(candidate, baseline, replicates=1000, chunk_size=9)
        b = paired.paired_bootstrap(candidate, baseline, replicates=1000, chunk_size=100)
        self.assertEqual(a, b)
        self.assertEqual(a["clean"]["both_correct"], 1)
        self.assertEqual(a["clean"]["both_wrong"], 1)
        self.assertEqual(a["clean"]["candidate_only_correct"], 1)
        self.assertEqual(a["clean"]["baseline_only_correct"], 1)
        with self.assertRaises(ValueError):
            paired.paired_bootstrap(candidate, baseline[:-1])
        with self.assertRaises(ValueError):
            paired.paired_bootstrap(candidate.astype(float)+.1, baseline)

    def test_source_partial_context_full_and_pixel_metadata_fail_before_model_compute(self):
        fixture = self.fixture()
        paired.validate_document(*fixture)
        for mutation in ({"schema": "official_development_stress_v2"}, {"status": "smoke"},
                         {"benchmark_source_sha256": "changed"}, {"examples": 4999},
                         {"evaluation_context": {"device_type": "cuda"}}, {"checkpoint_sha256": "another"},
                         {"bn_recalibrated": True}):
            changed = ({**fixture[0], **mutation}, *fixture[1:])
            with patch.object(paired, "build_model", side_effect=AssertionError("must reject before model")):
                with self.assertRaises(ValueError):
                    paired.validate_document(*changed)
        changed = copy.deepcopy(fixture)
        changed[1]["config"]["train_all"] = True
        with self.assertRaisesRegex(ValueError, "full-data"):
            paired.validate_document(*changed)
        for mutation in ({"examples": 1}, {"accuracy": float("nan")}, {"input_pixel_sha256": "missing"}):
            changed = copy.deepcopy(fixture)
            changed[0]["policies"]["flip"]["cases"]["noise_004"].update(mutation)
            with self.assertRaises(ValueError):
                paired.validate_document(*changed)

    def test_stability_paths_and_renamed_checkpoint_reject_before_compute(self):
        args = argparse.Namespace(candidate_checkpoint=Path("results/stability_seed2027/best.pt"),
                                  baseline_checkpoint=Path("never-accessed.pt"))
        with patch.object(paired, "load_training_arrays", side_effect=AssertionError("no inputs")):
            with self.assertRaisesRegex(ValueError, "stability"):
                paired.run(args)
        with patch.object(paired.suite, "sha256", return_value=paired.STABILITY_SHA), \
                patch.object(torch, "load", side_effect=AssertionError("no checkpoint deserialization")):
            with self.assertRaisesRegex(ValueError, "renamed"):
                paired.load_experiment(Path("renamed.pt"), Path("none.json"), {}, {}, {}, [], [])

    def test_cli_executor_rejects_full_data_or_changed_source_before_inference(self):
        # Exercise run -> load_experiment -> validation with real CPU file I/O;
        # only the official-input fixture and runtime seeding are replaced.
        for change, message in (("full", "full-data"), ("source", "source/hash/context")):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                directory = Path(folder)
                (directory/"training_manifest.json").write_text(json.dumps({"split": {"indices_sha256": "split"}}))
                document, checkpoint, *_ = self.fixture()
                core = {"train.py": "original"}
                signature = {"archive_sha256": "archive", "split_sha256": "split"}
                checkpoint["data_signature"] = signature
                checkpoint["provenance"] = {"source_sha256": core}
                if change == "full": checkpoint["config"]["train_all"] = True
                path, evidence = directory/"best.pt", directory/"benchmark.json"
                torch.save(checkpoint, path)
                (directory/"metrics.json").write_text("{}")
                (directory/"history.json").write_text("[]")
                document.update(data_signature=signature,
                    evaluation_context={"device_type": "cpu", "amp_dtype": "torch.float32",
                                        "torch_version": torch.__version__, "training_core_sha256": core},
                    benchmark_source_sha256="changed" if change == "source" else paired.suite.sha256(Path(paired.suite.__file__)),
                    checkpoint_sha256=paired.suite.sha256(path),
                    training_metrics_sha256=paired.suite.sha256(directory/"metrics.json"),
                    training_history_sha256=paired.suite.sha256(directory/"history.json"))
                evidence.write_text(json.dumps(document))
                args = argparse.Namespace(candidate_checkpoint=path, candidate_benchmark=evidence,
                    baseline_checkpoint=directory/"never-read.pt", baseline_benchmark=directory/"never-read.json",
                    output=directory/"result.json", policy="flip", data_dir=directory, device="cpu")
                arrays = (None, None, np.arange(40000), np.arange(40000, 45000))
                with patch.object(paired, "seed_all"), patch.object(paired, "load_training_arrays", return_value=arrays), \
                        patch.object(paired, "training_source", return_value={"archive_sha256": "archive"}), \
                        patch.object(paired, "code_provenance", return_value={"source_sha256": core}), \
                        patch.object(paired, "build_model", side_effect=AssertionError("must reject before model")), \
                        patch.object(paired, "collect_correctness", side_effect=AssertionError("no inference")):
                    with self.assertRaisesRegex(ValueError, message): paired.run(args)
                self.assertFalse(args.output.exists())

    def test_collect_reproduces_every_case_on_matched_synthetic_cpu_inputs(self):
        images = torch.zeros((4, 32, 32, 3), dtype=torch.uint8)
        images[1::2] = 255
        labels = torch.tensor([0, 1, 0, 1])
        reference = {"cases": {name: {"accuracy": 1., "input_pixel_sha256": "unused"} for name in paired.suite.CASES}}
        class TinyModel(torch.nn.Module):
            def forward(self, batch):
                mean = batch.float().mean(dim=(1, 2, 3))
                return torch.stack((255-mean, mean), dim=1)
        def logits(model, batch, policy): return model(batch)
        import hashlib
        for name in paired.suite.CASES:
            digest = hashlib.sha256()
            for start in range(0, 4, 3): digest.update(paired.suite.stress(images[start:start+3], name, start).numpy().tobytes())
            reference["cases"][name]["input_pixel_sha256"] = digest.hexdigest()
        with patch.object(paired.suite, "infer", side_effect=logits):
            result = paired.collect_correctness(TinyModel(), images, labels, "flip", 3, torch.device("cpu"), reference)
        self.assertEqual(result.shape, (4, 6))
        self.assertTrue(result.all())
        reference["cases"]["noise_004"]["accuracy"] = .5
        with patch.object(paired.suite, "infer", side_effect=logits):
            with self.assertRaisesRegex(ValueError, "differ"):
                paired.collect_correctness(TinyModel(), images, labels, "flip", 3, torch.device("cpu"), reference)


if __name__ == "__main__":
    unittest.main()
