"""Research safety gates must fail before another GPU job is launched."""
import copy
import ast
import base64
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from cifar_project.research import decisions, load_plan, make_notebook, reservation_available, verify_result


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.plan = load_plan(Path("research/queue.json"))

    def test_reservations_include_prior_and_failed_runs(self):
        state = {"runs": []}
        self.assertTrue(reservation_available(self.plan, state))
        state["runs"].append({"reserved_seconds": 1200, "status": "failed"})
        self.assertTrue(reservation_available(self.plan, state))
        state["runs"].append({"reserved_seconds": 1200, "status": "running"})
        self.assertFalse(reservation_available(self.plan, state))

    def test_plan_rejects_changed_split_seed_and_oversized_model(self):
        original = json.loads(Path("research/queue.json").read_text())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "queue.json"
            for overrides in [{"seed": 43}, {"base_width": 128}, {"base_width": 48, "learning_rate": 0.2}]:
                queue = copy.deepcopy(original)
                queue["candidates"][1]["overrides"] = overrides
                path.write_text(json.dumps(queue))
                with self.assertRaises(ValueError):
                    load_plan(path)

    def test_notebook_rendering_validates_template_before_launch(self):
        notebook = make_notebook("notebooks/train.ipynb", self.plan["experiments"][1], self.plan)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile(cell["source"], "candidate", "exec")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "changed.ipynb"
            path.write_text(json.dumps({"cells": []}))
            with self.assertRaises(ValueError):
                make_notebook(path, self.plan["experiments"][0], self.plan)

    def test_committed_notebook_bundles_the_current_source_only(self):
        notebook = json.loads(Path("notebooks/train.ipynb").read_text())
        payload = None
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            for node in ast.parse(source).body:
                if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "SOURCE_BUNDLE" for target in node.targets):
                    payload = ast.literal_eval(node.value)
        self.assertIsInstance(payload, str)
        expected = {path.as_posix(): path.read_bytes() for path in Path("cifar_project").glob("*.py")}
        expected["configs/baseline.json"] = Path("configs/baseline.json").read_bytes()
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload, validate=True))) as bundle:
            self.assertEqual(set(bundle.namelist()), set(expected))
            for name, content in expected.items():
                self.assertEqual(bundle.read(name), content)

    def test_result_rejects_partial_budget_and_changed_data(self):
        experiment = self.plan["experiments"][0]
        run = {"smoke_test": False, "batch_limits": {"train": None, "validation": None},
               "config": experiment["config"], "trainable_parameters": experiment["parameters"],
               "training_examples": 40000, "validation_examples": 5000, "confirmation_examples": 5000,
               "dataset_sha256": "data", "split_sha256": "split",
               "provenance": {"device": "cuda", "source_sha256": "source", "torch": "version", "gpu": "device"}}
        summary = {"smoke_test": False, "completed_epochs": 10, "validation_accuracy": 0.8}
        history = [{"epoch": i + 1, "seconds": 30,
                    "train": {"accuracy": 0.9, "loss": 0.5, "examples": 40000},
                    "validation": {"accuracy": 0.8, "loss": 0.8, "examples": 5000}} for i in range(10)]
        result = verify_result(summary, run, history, experiment, self.plan)
        self.assertAlmostEqual(result["last3_development_accuracy"], 0.8)
        self.assertFalse(result["confirmation_evaluated"])
        with self.assertRaises(ValueError):
            verify_result(summary, run, history[:-1], experiment, self.plan)
        changed = copy.deepcopy(run)
        changed["split_sha256"] = "different"
        with self.assertRaises(ValueError):
            verify_result(summary, changed, history, experiment, self.plan, run)

    def test_screening_decision_never_declares_final_success(self):
        state = {"runs": [
            {"id": "baseline", "status": "complete", "result": {"last3_development_accuracy": 0.8, "median_epoch_seconds": 30}},
            {"id": "candidate", "status": "complete", "result": {"last3_development_accuracy": 0.82, "median_epoch_seconds": 45}}]}
        result = decisions(state)
        self.assertFalse(result["final_model_selected"])
        self.assertFalse(result["confirmation_evaluated"])
        self.assertIn("replicate", result["comparisons"][0]["decision"])


if __name__ == "__main__":
    unittest.main()
