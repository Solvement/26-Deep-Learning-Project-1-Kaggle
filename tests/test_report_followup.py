"""CPU-only report boundaries; fixtures are synthetic, never scored evidence."""
import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import nbformat

import build_report as report


class FollowupReportTests(unittest.TestCase):
    def scored_fixture(self):
        followup = self.public_fixture()
        template = followup["rows"][0]
        followup["rows"] = [dict(template,id="resizemix",label="Synthetic ResizeMix",checkpoint_sha256="1"*64),
                            dict(template,id="wrn34_300",label="Synthetic WRN300",checkpoint_sha256="2"*64),
                            dict(template,id="sam",label="Synthetic SAM",checkpoint_sha256="3"*64)]
        followup["official_submissions"] = [
            {"run":"candidate","checkpoint_sha256":"1"*64,"policy":"center80_shift20","training_examples":40000,"public_score":.88},
            {"run":"candidate","checkpoint_sha256":"2"*64,"policy":"center80_shift20","training_examples":40000,"public_score":.9},
            {"run":"sam_actual_run","checkpoint_sha256":"3"*64,"policy":"center80_shift20","training_examples":40000,"public_score":.8}]
        return followup

    def test_shared_raw_run_labels_use_unique_checkpoint_policy_scope_without_mutating_snapshot(self):
        followup = self.scored_fixture()
        before = report.followup_snapshot(followup)
        labels = [report.receipt_label(followup, receipt) for receipt in followup["official_submissions"]]
        self.assertEqual(labels,["Synthetic ResizeMix","Synthetic WRN300","Synthetic SAM"])
        self.assertEqual(report.followup_snapshot(followup),before)
        self.assertEqual([r["run"] for r in followup["official_submissions"]],["candidate","candidate","sam_actual_run"])

    def test_unknown_policy_hash_scope_or_ambiguous_row_keeps_actual_run_name(self):
        followup = self.scored_fixture()
        receipt = followup["official_submissions"][0]
        for changes in ({"policy":"flip"},{"checkpoint_sha256":"f"*64},{"checkpoint_sha256":None,"policy":None},
                        {"training_examples":50000},{"training_examples":True}):
            self.assertEqual(report.receipt_label(followup,{**receipt,**changes}),"candidate")
        followup["rows"].append(dict(followup["rows"][0]))
        self.assertEqual(report.receipt_label(followup,receipt),"candidate")

    def test_outcome_uses_scored_matched_recipe_and_actual_sam_negative_result(self):
        followup = self.scored_fixture()
        # A high development accuracy and stale display aggregate cannot pick the winner.
        followup["rows"][0]["clean_accuracy"] = 1.
        followup["best_new_public"] = .99
        text = report.scored_outcome_text(followup)
        self.assertIn("Synthetic WRN300 at 90.00%",text)
        self.assertIn("SAM scored 80.00%",text)
        self.assertIn("did not improve the public best",text)
        followup["official_submissions"][2]["public_score"] = .95
        self.assertNotIn("did not improve",report.scored_outcome_text(followup))

    def test_zero_is_a_real_score_but_missing_or_unmatched_sam_creates_no_conclusion(self):
        followup = self.scored_fixture()
        followup["official_submissions"] = [dict(followup["official_submissions"][0],public_score=0.)]
        text = report.scored_outcome_text(followup)
        self.assertIn("Synthetic ResizeMix at 0.00%",text)
        self.assertNotIn("SAM scored",text)
        for score in (None,False,float("nan"),float("inf")):
            followup["official_submissions"][0]["public_score"] = score
            self.assertEqual(report.scored_outcome_text(followup),"")
        followup = self.scored_fixture()
        followup["official_submissions"][2]["training_examples"] = 50000
        self.assertNotIn("SAM scored",report.scored_outcome_text(followup))

    def test_research_cell_emits_a_real_png_on_the_headless_backend(self):
        evidence = dict(self.primary_fixture(), followup=self.public_fixture())
        with patch.object(nbformat, "write") as write:
            report.create_notebook(evidence, None, self.plots_fixture(), Path("unused-fixture-output"), execute=False)
        cells = [c for c in write.call_args.args[0].cells if c.cell_type == "code"]
        namespace = {"json": json}
        with patch("IPython.display.display") as display, redirect_stdout(io.StringIO()):
            exec(compile(cells[6].source, "research-cell-fixture", "exec"), namespace)
        images = [call.args[0] for call in display.call_args_list if getattr(call.args[0], "format", None) == "png"]
        self.assertEqual(len(images), 1)
        self.assertIsInstance(images[0].data, bytes)
        self.assertTrue(images[0].data.startswith(b"\x89PNG\r\n\x1a\n"))

    def public_fixture(self):
        """An intentionally synthetic public helper contract, not real results."""
        return {
            "schema": "research_report_followup_v1", "recorded_at": "synthetic-fixture-snapshot",
            "repository": "https://github.com/example/synthetic-research",
            "source_summary_sha256": "a" * 64,
            "baseline": {"clean_accuracy": .6, "clean_nll": .8, "stress_accuracy": .5,
                         "policy": "flip", "examples": 5000, "training_examples": 40000},
            "rows": [{"id": "synthetic_swa", "label": "Synthetic SWA rejected", "parameters": 1234,
                      "epochs": 200, "precision": "bf16", "policy": "center80_shift20",
                      "checkpoint_epoch": 200, "checkpoint_variant": "swa", "clean_accuracy": .5,
                      "clean_nll": .9, "stress_accuracy": .4, "gate_passed": False,
                      "clean_delta": -.1, "stress_delta": -.1, "clean_ci95": [-.2, -.05],
                      "stress_ci95": [-.15, -.05], "repeat_status": "not_promoted",
                      "repeat_clean_accuracy": None, "repeat_stress_accuracy": None,
                      "repeat_epochs_completed": None, "repeat_planned_epochs": None,
                      "delivery_status": "not_promoted", "checkpoint_sha256": "b" * 64}],
            "official_submissions": [], "unscored_receipts_count": 1,
            "original_best_public": .874, "original_primary_public": .871,
            "best_new_public": None, "best_observed_public": .874,
            "limitations": ["Synthetic reused-development limitation."],
            "engineering": ["Synthetic engineering-only status."],
        }

    def primary_fixture(self):
        return {"runs": [], "submission": {"run": "final_full", "public_score": .871},
                "confirmation": {"run": "regularized", "accuracy": .9664}, "selection": {},
                "stability": {}, "submissions": [], "exploratory": {}}

    def plots_fixture(self):
        return {"learning_curves": report.ROOT / "outputs" / "synthetic-curves.png",
                "architecture": report.ROOT / "outputs" / "synthetic-architecture.png",
                "followup": report.ROOT / "outputs" / "synthetic-followup.png"}

    def test_default_destination_keeps_original_behavior(self):
        self.assertEqual(report.followup_output_dir(None, False), report.ROOT / "outputs")
        with tempfile.TemporaryDirectory() as folder:
            existing = Path(folder)
            self.assertEqual(report.followup_output_dir(existing, False), existing)

    def test_followup_requires_unused_explicit_destination(self):
        with self.assertRaisesRegex(ValueError, "explicit new"):
            report.followup_output_dir(None, True)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(ValueError, "must not already exist"):
                report.followup_output_dir(root, True)
            target = root / "new-research-edition"
            self.assertEqual(report.followup_output_dir(target, True), target)
            self.assertFalse(target.exists())

    def test_followup_cannot_write_inside_frozen_backup(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(report, "ROOT", Path(folder)):
            frozen = Path(folder) / "outputs" / "primary_delivery_20261031"
            for target in (frozen, frozen / "new-child"):
                with self.subTest(target=target), self.assertRaisesRegex(ValueError, "frozen primary"):
                    report.followup_output_dir(target, True)
                self.assertFalse(target.exists())

    def test_public_followup_is_attached_without_replacing_primary_evidence(self):
        primary = {"runs": [{"name": "final_full"}], "selection": {"run": "regularized"},
                   "confirmation": {"run": "regularized", "accuracy": 0.9664},
                   "submission": {"run": "final_full", "public_score": 0.871}}
        original = copy.deepcopy(primary)
        helper = ModuleType("scripts.research_report_evidence")
        projection = {"schema": "synthetic_public_followup_fixture", "official_submissions": []}
        helper.load_followup = Mock(return_value=projection)
        summary, receipts = Path("public-summary-fixture.json"), Path("receipts-fixture")
        with patch.dict(sys.modules, {"scripts.research_report_evidence": helper}):
            result = report.attach_followup(primary, summary, receipts)
        helper.load_followup.assert_called_once_with(summary, receipts)
        self.assertIs(result, primary)
        self.assertIs(result["followup"], projection)
        self.assertEqual({key: result[key] for key in original}, original)

    def test_rejected_followup_does_not_create_destination_or_render(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "new-research-edition"
            arguments = ["build_report.py", "--research-summary", "synthetic-public-summary.json",
                         "--output-dir", str(target)]
            with patch.object(sys, "argv", arguments), patch.object(report, "collect_evidence", return_value={}), \
                    patch.object(report, "attach_followup", side_effect=ValueError("synthetic rejected evidence")), \
                    patch.object(report, "make_plots") as plots:
                with self.assertRaisesRegex(ValueError, "rejected evidence"):
                    report.main()
            self.assertFalse(target.exists())
            plots.assert_not_called()

    def test_default_notebook_retains_six_code_cells_without_writing_outputs(self):
        evidence = {"runs": [], "submission": {}, "confirmation": {}, "selection": {},
                    "stability": {}, "submissions": [], "exploratory": {}}
        plots = {"learning_curves": report.ROOT / "outputs" / "synthetic-curves.png",
                 "architecture": report.ROOT / "outputs" / "synthetic-architecture.png"}
        with patch.object(nbformat, "write") as write:
            report.create_notebook(evidence, None, plots, Path("unused-fixture-output"), execute=False)
        notebook = write.call_args.args[0]
        self.assertEqual(sum(cell.cell_type == "code" for cell in notebook.cells), 6)
        self.assertTrue(all(cell.execution_count is None for cell in notebook.cells if cell.cell_type == "code"))
        self.assertFalse(any("research_report_evidence" in cell.source for cell in notebook.cells))

    def test_original_code_cells_match_the_frozen_curated_notebook_sources(self):
        frozen = nbformat.read(report.ROOT / "notebooks" / "Project1_Results_Executed.ipynb", as_version=4)
        evidence = self.primary_fixture()
        evidence["results_dir"] = str(report.ROOT / "results")
        plots = {"learning_curves": report.ROOT / "outputs" / "learning_curves.png",
                 "architecture": report.ROOT / "outputs" / "architecture.png",
                 "confirmation": report.ROOT / "outputs" / "confirmation_confusion.png"}
        with patch.object(nbformat, "write") as write:
            report.create_notebook(evidence, {"name": "final_full"}, plots, Path("unused-fixture-output"), execute=False)
        generated = write.call_args.args[0]
        self.assertEqual([cell.source for cell in generated.cells if cell.cell_type == "code"],
                         [cell.source for cell in frozen.cells if cell.cell_type == "code"])

    def test_followup_notebook_preserves_six_original_cells_and_adds_two_public_cells(self):
        primary = self.primary_fixture()
        followup = dict(primary, followup=self.public_fixture())
        with patch.object(nbformat, "write") as write:
            report.create_notebook(primary, None, self.plots_fixture(), Path("unused-fixture-output"), execute=False)
            original = write.call_args.args[0]
            report.create_notebook(followup, None, self.plots_fixture(), Path("unused-fixture-output"), execute=False)
            extended = write.call_args.args[0]
        before = [cell.source for cell in original.cells if cell.cell_type == "code"]
        after = [cell.source for cell in extended.cells if cell.cell_type == "code"]
        self.assertEqual(len(after), 8)
        self.assertEqual(after[:6], before)
        nbformat.validate(extended)
        namespace = {"json": json, "accuracy": report.accuracy}
        with redirect_stdout(io.StringIO()) as output:
            exec(after[6], namespace)
            exec(after[7], namespace)
        text = output.getvalue()
        self.assertIn("No newly scored research submission", text)
        self.assertIn("Best newly scored research public score: Pending", text)
        self.assertIn("Original primary public score: 87.10%", text)
        self.assertIn("Best original public score: 87.40%", text)
        self.assertIn("96.64% confirmation belongs only to regularized trained on 40,000 images", text)
        self.assertEqual(namespace["followup"], followup["followup"])
        self.assertNotIn("results/research/", "\n".join(after[6:]))
        self.assertNotIn("load_followup", "\n".join(after[6:]))

    def test_public_snapshot_hash_and_delta_intervals(self):
        projection = self.public_fixture()
        payload, digest = report.followup_snapshot(projection)
        self.assertEqual(json.loads(payload), projection)
        self.assertEqual(hashlib.sha256(payload.encode()).hexdigest(), digest)
        self.assertEqual(report.delta_interval([-.002, .004]), "[-0.20, +0.40] pp")
        self.assertEqual(report.delta_interval(None), "Not available")
        projection["baseline"]["clean_accuracy"] = float("nan")
        with self.assertRaises(ValueError):
            report.followup_snapshot(projection)

    def test_missing_or_other_confirmation_is_never_replaced_with_a_fixed_accuracy(self):
        for confirmation in ({}, {"run": "other-recipe", "accuracy": .9664}):
            with self.subTest(confirmation=confirmation):
                statement = report.confirmation_scope({"confirmation": confirmation})
                self.assertIn("unavailable", statement)
                self.assertNotIn("96.64%", statement)
        self.assertIn("91.00%", report.confirmation_scope({"confirmation": {"run": "regularized", "accuracy": .91}}))

    def test_written_snapshot_bytes_match_embedded_and_manifest_hash_on_windows(self):
        projection = self.public_fixture()
        payload, expected = report.followup_snapshot(projection)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic-public-snapshot.json"
            measured = report.write_followup_snapshot(path, projection)
            self.assertEqual(path.read_bytes(), payload.encode("utf-8"))
            self.assertNotIn(b"\r\n", path.read_bytes())
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)
            self.assertEqual(measured, expected)

    def test_new_scored_best_never_replaces_original_primary_or_original_best(self):
        projection = self.public_fixture()
        projection.update(best_new_public=.9, best_observed_public=.9)
        projection["official_submissions"] = [{
            "run": "synthetic_scored_fixture", "submission_ref": 123,
            "public_score": .9, "private_score": None, "submission_sha256": "c"*64,
            "checkpoint_sha256": "b"*64, "receipt_sha256": "d"*64}]
        primary = self.primary_fixture()
        with patch.object(nbformat, "write") as write:
            report.create_notebook(dict(primary, followup=projection), None, self.plots_fixture(),
                                   Path("unused-fixture-output"), execute=False)
        cells = [cell for cell in write.call_args.args[0].cells if cell.cell_type == "code"]
        namespace = {"json": json, "accuracy": report.accuracy, "followup": projection}
        with redirect_stdout(io.StringIO()) as output:
            exec(cells[-1].source, namespace)
        self.assertIn("Original primary public score: 87.10%", output.getvalue())
        self.assertIn("Best original public score: 87.40%", output.getvalue())
        self.assertIn("Best newly scored research public score: 90.00%", output.getvalue())
        self.assertIn("Best observed public score (original + research): 90.00%", output.getvalue())
        self.assertEqual(primary["submission"]["public_score"], .871)

    def test_default_pdf_keeps_five_sections_and_original_references(self):
        from reportlab import platypus
        with patch.object(platypus, "SimpleDocTemplate") as doc, patch("pypdf.PdfReader") as reader:
            reader.return_value.pages = [None] * 5
            report.create_pdf(self.primary_fixture(), None, self.plots_fixture(), Path("unused-fixture-output"))
        story = doc.return_value.build.call_args.args[0]
        self.assertEqual(sum(isinstance(item, platypus.PageBreak) for item in story), 4)
        text = "\n".join(getattr(item, "text", "") for item in story)
        self.assertNotIn("Follow-on research:", text)
        self.assertNotIn("Phase boundary:", text)
        self.assertNotIn("Averaging Weights Leads", text)

    def test_followup_plot_uses_points_without_inventing_missing_swa_intervals(self):
        from matplotlib.figure import Figure
        projection = self.public_fixture()
        paired = projection["rows"][0]
        paired.update(label="Synthetic paired candidate", gate_passed=True)
        swa = dict(paired, label="Synthetic SWA without paired evidence", gate_passed=False,
                   clean_delta=-.02, stress_delta=-.03, clean_ci95=None, stress_ci95=None)
        projection["rows"].append(swa)
        with patch.object(Figure, "savefig", autospec=True) as save:
            target = report.make_followup_plot(projection, Path("unused-fixture-output"))
        self.assertEqual(target.name, "research_conditional_deltas.png")
        figure = save.call_args.args[0]
        self.assertEqual(len(figure.axes), 2)
        for axis, metric in zip(figure.axes, ("clean", "stress")):
            self.assertEqual(len(axis.collections), 2)  # One real CI segment plus its caps only.
            failed_points = [line for line in axis.lines if line.get_marker() == "x"]
            self.assertEqual(len(failed_points), 1)
            self.assertEqual(list(failed_points[0].get_xdata()), [100*swa[metric+"_delta"]])
            self.assertIn("[gate failed]", " ".join(label.get_text() for label in figure.axes[0].get_yticklabels()))
            self.assertGreaterEqual(axis.xaxis.label.get_fontsize(), 13)

    def test_followup_plot_is_only_wired_when_public_followup_exists(self):
        from matplotlib.figure import Figure
        primary = self.primary_fixture()
        with patch.object(Figure, "savefig"), patch.object(report, "choose_run", return_value=None), \
                patch.object(report, "make_followup_plot", return_value=Path("synthetic-followup.png")) as followup_plot:
            original = report.make_plots(primary, Path("unused-fixture-output"))
            followup_plot.assert_not_called()
            extended = report.make_plots(dict(primary, followup=self.public_fixture()), Path("unused-fixture-output"))
        self.assertNotIn("followup", original)
        self.assertEqual(extended["followup"], Path("synthetic-followup.png"))
        followup_plot.assert_called_once()

    def test_research_tables_stay_after_primary_and_before_references_without_rendering(self):
        from reportlab import platypus
        evidence = dict(self.primary_fixture(), followup=self.public_fixture())
        with patch.object(platypus, "SimpleDocTemplate") as doc, patch("pypdf.PdfReader") as reader:
            reader.return_value.pages = [None] * 7
            report.create_pdf(evidence, None, self.plots_fixture(), Path("unused-fixture-output"))
        story = doc.return_value.build.call_args.args[0]
        text = "\n".join(getattr(item, "text", "") for item in story)
        self.assertLess(text.index("Measured findings and limitations"), text.index("Follow-on research: development evidence"))
        self.assertLess(text.index("Follow-on research: methods and official outcomes"), text.index("Reproduction, delivery and references"))
        self.assertEqual(sum(isinstance(item, platypus.PageBreak) for item in story), 6)
        self.assertIn("96.64% confirmation belongs only to regularized", text)
        self.assertIn("full-50k refit has no independent development score", text)
        table_text = " ".join(str(getattr(cell, "text", cell)) for item in story if isinstance(item, platypus.Table)
                              for row in item._cellvalues for cell in row)
        self.assertIn("Synthetic SWA rejected", table_text)
        self.assertIn("Not promoted", table_text)
        self.assertIn("No newly scored research submission", table_text)
        self.assertIn("5,000 original-image clusters", text)
        self.assertIn("10,000 bootstrap replicates", text)
        self.assertIn("SWA without paired evidence has a point only", text)


if __name__ == "__main__":
    unittest.main()
