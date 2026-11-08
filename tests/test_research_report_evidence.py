"""Reporting safety: fixed local comparisons and genuine canonical score states."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.research_report_evidence import load_followup

ROOT = Path(__file__).resolve().parents[1]


class ResearchReportEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.path = self.directory/"research-followup-results.json"
        self.summary = json.loads((ROOT/"docs/research-followup-results.json").read_text())
        self.legacy = json.loads((ROOT/"docs/results.json").read_text())
        (self.directory/"results.json").write_text(json.dumps(self.legacy))
        self.receipts = self.directory/"submission_receipts"

    def load(self):
        self.path.write_text(json.dumps(self.summary))
        return load_followup(self.path, self.receipts)

    def receipt(self, **changes):
        value = {"run": "wrn34_3_s2026", "competition": "cifar-10-res-net-architecture-challenge-dl",
            "team": "Saber", "status": "complete", "submission_ref": 900001,
            "public_score": .91, "private_score": None, "submission_sha256": "1"*64,
            "checkpoint_sha256": self.summary["wrn34_3_primary"]["checkpoint_sha256"],
            "manifest_sha256": "3"*64, "frozen_manifest_sha256": "4"*64,
            "parameter_count": 4166298, "policy": "center80_shift20", "seed": 2026,
            "training_examples": 40000, "submitted_at": "2026-10-31T20:00:00+00:00",
            "checked_at": "2026-10-31T20:01:00+00:00", "delivery_schema": "research_prediction_delivery_v1"}
        value.update(changes)
        return value

    def write_receipt(self, record, name="900001.json"):
        self.receipts.mkdir(exist_ok=True)
        path = self.receipts/name
        path.write_text(json.dumps(record))
        return path

    def test_real_public_aggregate_has_exact_local_values_and_separate_original_identity(self):
        actual = self.load()
        rows = {r["id"]: r for r in actual["rows"]}
        self.assertEqual(set(rows), {"wrn34_160", "sam", "resizemix", "wrn34_300", "swa40"})
        self.assertEqual((rows["wrn34_300"]["epochs"], rows["wrn34_300"]["checkpoint_epoch"]), (300, 298))
        self.assertEqual(rows["wrn34_300"]["clean_accuracy"], .9654)
        self.assertEqual(rows["sam"]["clean_nll"], self.summary["SAM_local_primary"]["policies"]["center80_shift20"]["clean_loss"])
        self.assertEqual(rows["wrn34_160"]["repeat_status"], "complete_passed")
        self.assertFalse(rows["swa40"]["gate_passed"])
        self.assertEqual(rows["swa40"]["delivery_status"], "not_promoted")
        self.assertEqual((actual["original_best_public"], actual["original_primary_public"], actual["best_observed_public"]), (.874, .871, .874))
        self.assertIsNone(actual["best_new_public"])
        self.assertEqual(actual["official_submissions"], [])
        serialized = json.dumps(actual)
        for private in ("C:\\", "/content/", "KGAT_", "ya29.", "supervisor.json"):
            self.assertNotIn(private, serialized)

    def test_reader_import_and_real_projection_never_import_torch(self):
        code = ("import sys; from scripts.research_report_evidence import load_followup; "
                "load_followup('docs/research-followup-results.json', 'tmp/nonexistent-report-receipts'); "
                "assert 'torch' not in sys.modules; print('stdlib-only')")
        output = subprocess.check_output([sys.executable, "-c", code], cwd=ROOT, text=True)
        self.assertEqual(output.strip(), "stdlib-only")

    def test_free_text_and_csv_ready_cannot_invent_a_new_score_or_repeat(self):
        self.summary.pop("SAM_local_repeat", None)
        self.summary["SAM_status"] = "complete; public accuracy99.9%"
        self.summary["sam_status"] = "scored"
        self.summary["research_submission"].update(actual_best_public=.999, new_submission_reference=123456)
        result = self.load()
        self.assertIsNone(result["best_new_public"])
        self.assertEqual(result["best_observed_public"], .874)
        self.assertEqual(next(r for r in result["rows"] if r["id"] == "sam")["repeat_status"], "not_verified")

    def test_zero_is_a_genuine_completed_score_and_does_not_replace_old_best(self):
        self.write_receipt(self.receipt(public_score=0.))
        result = self.load()
        self.assertEqual(result["best_new_public"], 0.)
        self.assertEqual(result["best_observed_public"], .874)
        self.assertEqual(len(result["official_submissions"]), 1)
        self.assertEqual(next(r for r in result["rows"] if r["id"] == "wrn34_160")["delivery_status"], "scored")

    def test_new_official_score_is_separate_from_original_primary_and_old_best(self):
        self.write_receipt(self.receipt())
        result = self.load()
        self.assertEqual((result["original_primary_public"], result["original_best_public"], result["best_new_public"], result["best_observed_public"]),
                         (.871, .874, .91, .91))
        self.assertEqual(self.legacy["confirmation"]["accuracy"], .9664)

    def test_only_whitelisted_fields_enter_the_public_snapshot(self):
        self.summary["interpretation"] = "KGAT_fakecredential /content/private C:\\Private"
        record = self.receipt(message="KGAT_fakecredential", csv_path="C:\\Private\\prediction.csv",
                              runtime_url="https://private.invalid/secret", labels=[1, 2, 3])
        self.write_receipt(record)
        result = self.load()
        encoded = json.dumps(result)
        for forbidden in ("fakecredential", "prediction.csv", "private.invalid", "labels", "csv_path"):
            self.assertNotIn(forbidden, encoded)
        self.assertEqual(set(result["official_submissions"][0]), {"run", "submission_ref", "public_score", "private_score",
            "submission_sha256", "checkpoint_sha256", "policy", "receipt_sha256", "training_examples", "validation_examples"})

    def test_scored_row_requires_same_checkpoint_policy_training_scope_and_parameters(self):
        different_recipes = ({"policy": "flip"}, {"policy": "plain"}, {"parameter_count": 4166299},
            {"training_examples": 50000, "validation_examples": 0, "independent_development_evaluated": False,
             "development_accuracy": None, "parent_delivery_sha256": "5"*64,
             "delivery_schema": "research_refit_delivery_v1"})
        for changes in different_recipes:
            with self.subTest(changes=changes):
                self.write_receipt(self.receipt(**changes))
                actual = self.load()
                row = next(r for r in actual["rows"] if r["id"] == "wrn34_160")
                self.assertEqual(row["delivery_status"], "qualified_unscored")
                self.assertEqual((actual["best_new_public"], actual["best_observed_public"]), (.91, .91))
                self.assertEqual(actual["official_submissions"][0]["policy"], changes.get("policy", "center80_shift20"))
                self.assertEqual(actual["official_submissions"][0]["training_examples"], changes.get("training_examples", 40000))
        self.write_receipt(self.receipt())
        row = next(r for r in self.load()["rows"] if r["id"] == "wrn34_160")
        self.assertEqual(row["delivery_status"], "scored")

    def test_pending_upload_states_and_intent_directories_are_never_scores(self):
        for state in ("upload_attempt_started", "submitted", "recovered", "unknown", "scoring_pending", "scoring_error"):
            with self.subTest(state=state):
                self.write_receipt(self.receipt(status=state, public_score=None))
                actual = self.load()
                self.assertEqual((actual["official_submissions"], actual["unscored_receipts_count"]), ([], 1))
                self.assertIsNone(actual["best_new_public"])
        self.path.write_text(json.dumps(self.summary))
        with self.assertRaisesRegex(ValueError, "canonical"):
            load_followup(self.path, self.directory/"submission_intents")

    def test_completed_benchmark_or_composed_proof_cannot_be_relabelled_a_score(self):
        for schema in ("official_development_stress_v3", "research_cloud_repeat_attestation_v1", "research_submit_intent_v1"):
            with self.subTest(schema=schema):
                self.write_receipt(self.receipt(schema=schema))
                with self.assertRaisesRegex(ValueError, "not a scored receipt"):
                    self.load()

    def test_receipts_reject_other_identity_nonfinite_bool_percentage_and_missing_score(self):
        mutations = [("team", "Other"), ("competition", "another-competition"), ("submission_ref", True),
                     ("public_score", True), ("public_score", float("nan")), ("public_score", float("inf")),
                     ("public_score", 91), ("public_score", ".91"), ("public_score", None),
                     ("parameter_count", 5000001), ("seed", 2027), ("manifest_sha256", "0"*63),
                     ("checked_at", "2026-10-31T19:59:00+00:00"), ("checked_at", "2026-10-31T20:01:00")]
        for field, bad in mutations:
            with self.subTest(field=field, bad=str(bad)):
                self.write_receipt(self.receipt(**{field: bad}))
                with self.assertRaises(ValueError):
                    self.load()

    def test_duplicates_deduplicate_only_when_model_and_score_metadata_agree(self):
        first = self.receipt()
        self.write_receipt(first)
        self.write_receipt({**first, "checked_at": "2026-10-31T20:02:00+00:00"}, "copy.json")
        self.assertEqual(len(self.load()["official_submissions"]), 1)
        mutations = [("submission_ref", 900002), ("submission_sha256", "5"*64), ("checkpoint_sha256", "5"*64),
                     ("public_score", .92), ("parameter_count", 4298970), ("frozen_manifest_sha256", "5"*64)]
        for key, value in mutations:
            with self.subTest(key=key):
                self.write_receipt({**first, key: value}, "copy.json")
                with self.assertRaisesRegex(ValueError, "conflict"):
                    self.load()

    def test_complete_receipt_supersedes_an_identical_pending_reference(self):
        self.write_receipt(self.receipt(status="scoring_pending", public_score=None), "a.json")
        self.write_receipt(self.receipt(checked_at="2026-10-31T20:02:00+00:00"), "b.json")
        result = self.load()
        self.assertEqual((len(result["official_submissions"]), result["unscored_receipts_count"]), (1, 0))

    def test_legacy_reference_or_csv_cannot_be_counted_as_new_research(self):
        old = self.legacy["official_submissions"][0]
        for changes in ({"submission_ref": old["submission_ref"]}, {"submission_sha256": old["submission_sha256"]}):
            self.write_receipt(self.receipt(**changes))
            with self.assertRaisesRegex(ValueError, "original submission"):
                self.load()

    def test_full50k_scores_have_no_independent_development_result(self):
        record = self.receipt(run="wrn34_3_full50k_s2026", training_examples=50000,
            checkpoint_sha256="2"*64, validation_examples=0, independent_development_evaluated=False,
            development_accuracy=None, parent_delivery_sha256="5"*64, delivery_schema="research_refit_delivery_v1")
        self.write_receipt(record)
        projected = self.load()["official_submissions"][0]
        self.assertEqual((projected["training_examples"], projected["validation_examples"]), (50000, 0))
        self.assertNotIn("development_accuracy", projected)
        for field, bad in (("validation_examples", 5000), ("validation_examples", False),
                           ("independent_development_evaluated", True), ("development_accuracy", .99),
                           ("parent_delivery_sha256", "bad")):
            self.write_receipt({**record, field: bad})
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.load()

    def test_partial_benchmark_wrong_context_or_data_cannot_enter_comparison_table(self):
        original = copy.deepcopy(self.summary)
        cases = [lambda q: q.update(schema="official_development_stress_v2"),
                 lambda q: q.update(status="running"), lambda q: q.update(training_examples=50000),
                 lambda q: q["evaluation_context"].update(amp_dtype="torch.float16"),
                 lambda q: q["data_signature"].update(split_sha256="0"*64),
                 lambda q: q["policies"]["plain"]["cases"]["clean"].update(examples=4999),
                 lambda q: q["policies"]["plain"].update(clean_loss=float("nan"))]
        for mutation in cases:
            self.summary = copy.deepcopy(original)
            mutation(self.summary["SAM_local_primary"])
            with self.assertRaises(ValueError):
                self.load()

    def test_benchmark_aggregate_and_saved_gate_must_match_measured_cases(self):
        original = copy.deepcopy(self.summary)
        self.summary["SAM_local_primary"]["policies"]["center80_shift20"]["stress_accuracy"] += .001
        with self.assertRaisesRegex(ValueError, "aggregate"):
            self.load()
        self.summary = copy.deepcopy(original)
        self.summary["SAM_local_primary"]["promotion"]["center80_shift20"]["passed"] = False
        with self.assertRaisesRegex(ValueError, "gate"):
            self.load()

    def test_pairing_counts_cluster_unit_hashes_and_intervals_are_bound(self):
        original = copy.deepcopy(self.summary)
        cases = [lambda p: p.update(candidate_checkpoint_sha256="0"*64),
                 lambda p: p.update(candidate_benchmark_sha256="0"*64),
                 lambda p: p["evaluation_context"].update(amp_dtype="torch.float16"),
                 lambda p: p["bootstrap"].update(replicates=9999),
                 lambda p: p["bootstrap"].update(sampling_unit="individual_stress_view"),
                 lambda p: p["bootstrap"]["clean"].update(candidate_only_correct=999),
                 lambda p: p["bootstrap"]["stress"].update(percentile_ci95=[.5, .7])]
        for mutation in cases:
            self.summary = copy.deepcopy(original)
            mutation(self.summary["SAM_paired"])
            with self.assertRaises(ValueError):
                self.load()

    def test_partial_training_never_counts_as_a_complete_independent_repeat(self):
        self.summary.pop("resizemix_local_repeat", None)
        # The public campaign now has a completed repeat. Construct the earlier
        # progress-only state explicitly so this safety test stays independent
        # of the campaign's current progress and completion metadata.
        progress = self.summary["resizemix_repeat_progress"] = {
            "planned_epochs": 160, "training_seed": 2027,
            "configuration_and_sources_verified": True,
            "all_recorded_metrics_finite": True,
        }
        for epoch in (9, 160):
            progress.update(epochs_completed=epoch, successful_updates=epoch*313, skipped_updates=0)
            row = next(r for r in self.load()["rows"] if r["id"] == "resizemix")
            self.assertEqual((row["repeat_status"], row["repeat_epochs_completed"]), ("in_progress", epoch))
            self.assertIsNone(row["repeat_clean_accuracy"])

    def test_independent_repeat_requires_different_checkpoint_seed_and_same_recipe(self):
        original = copy.deepcopy(self.summary)
        mutations = [lambda r: r["config"].update(seed=2026),
                     lambda r: r["config"].update(weight_decay=.0001),
                     lambda r: r.update(checkpoint_sha256=self.summary["wrn34_3_primary"]["checkpoint_sha256"])]
        for mutation in mutations:
            self.summary = copy.deepcopy(original)
            mutation(self.summary["wrn34_3_repeat"])
            with self.assertRaises(ValueError):
                self.load()

    def test_complete_repeat_without_paired_evidence_does_not_qualify(self):
        del self.summary["wrn34_3_paired"]
        row = next(r for r in self.load()["rows"] if r["id"] == "wrn34_160")
        self.assertTrue(row["gate_passed"])
        self.assertEqual(row["repeat_status"], "complete_passed")
        self.assertEqual(row["delivery_status"], "awaiting_pair")
        self.assertIsNone(row["clean_ci95"])
        self.assertIsNone(row["stress_ci95"])

    def test_stress_only_promotion_requires_strictly_positive_paired_stress_lower_bound(self):
        pair = self.summary["wrn34_3_paired"]["bootstrap"]["stress"]
        for lower in (-.001, 0):
            with self.subTest(lower=lower):
                pair["percentile_ci95"][0] = lower
                row = next(r for r in self.load()["rows"] if r["id"] == "wrn34_160")
                self.assertTrue(row["gate_passed"])
                self.assertLess(row["clean_delta"], .001)
                self.assertGreaterEqual(row["stress_delta"], .005)
                self.assertEqual(row["repeat_status"], "complete_passed")
                self.assertEqual(row["delivery_status"], "not_promoted")
        pair["percentile_ci95"][0] = .000001
        row = next(r for r in self.load()["rows"] if r["id"] == "wrn34_160")
        self.assertEqual(row["delivery_status"], "qualified_unscored")

    def test_clean_path_promotion_still_requires_a_pair_but_not_positive_stress_lower_bound(self):
        repeat = copy.deepcopy(self.summary["WRN34_300_local_primary"])
        repeat["checkpoint_sha256"] = "6"*64
        repeat["config"]["seed"] = 2027
        self.summary["WRN34_300_local_repeat"] = repeat
        self.summary["WRN34_300_local_paired"]["bootstrap"]["stress"]["percentile_ci95"][0] = 0
        row = next(r for r in self.load()["rows"] if r["id"] == "wrn34_300")
        self.assertGreaterEqual(row["clean_delta"], .001)
        self.assertEqual(row["delivery_status"], "qualified_unscored")
        del self.summary["WRN34_300_local_paired"]
        row = next(r for r in self.load()["rows"] if r["id"] == "wrn34_300")
        self.assertEqual(row["delivery_status"], "awaiting_pair")

    def test_swa_cannot_swap_variant_or_inherit_parent_advantage_without_increment_gate(self):
        original = copy.deepcopy(self.summary)
        for mutation in (lambda q: q.update(candidate_eligible=True),
                         lambda q: q.update(checkpoint_variant="ema"),
                         lambda q: q["promotion_vs_parent"]["center80_shift20"].update(passed=True)):
            self.summary = copy.deepcopy(original)
            mutation(self.summary["SWA_fixed40_primary"])
            with self.assertRaises(ValueError):
                self.load()

    def test_duplicate_json_keys_cannot_hide_conflicting_score_or_state(self):
        self.write_receipt(self.receipt())
        path = self.receipts/"900001.json"
        text = path.read_text()
        path.write_text(text[:-1]+', "public_score": 0.99}')
        with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
            self.load()


if __name__ == "__main__":
    unittest.main()
