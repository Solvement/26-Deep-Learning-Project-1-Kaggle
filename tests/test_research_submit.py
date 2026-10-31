"""Research upload safety with fake APIs only; no network mutations or GPU use."""
from datetime import datetime, timezone
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from research import submit_candidate as submit


class ResearchSubmissionTests(unittest.TestCase):
    def fixture(self, root):
        (root/"research").mkdir()
        (root/"results").mkdir()
        (root/"research/campaign.json").write_text(json.dumps({"team": "Saber", "stops_at": "2999-01-01T00:00:00Z",
                                                            "submission_limit_per_team_utc_day": 5}))
        (root/"results/kaggle_submissions.json").write_text("[]")
        csv_path, manifest = root/"submission.csv", root/"delivery.json"
        csv_path.write_text("id,label\n"+"".join(f"{i},{i%10}\n" for i in range(2000)))
        manifest.write_text(json.dumps({"schema": "research_prediction_delivery_v1"}))
        details = {"run": "wrn34_3_s2026", "competition": submit.COMPETITION, "csv_path": str(csv_path),
            "submission_sha256": submit.sha256(csv_path), "checkpoint_sha256": "b"*64,
            "parameter_count": 4166298, "policy": "center80_shift20", "seed": 2026, "training_examples": 40000,
            "manifest_sha256": submit.sha256(manifest), "frozen_manifest_sha256": "c"*64}
        api = Mock()
        api.recent.return_value = []
        api.limits.return_value = {"num_today": 0, "num_allowed_now": 5}
        api.submit.return_value = {"ref": 123}
        api.get.return_value = {"ref": 123, "status": "COMPLETE", "public_score": ".9", "team_name": "Saber",
                                "description": "Saber research sha256:"+details["submission_sha256"]}
        validator = Mock(side_effect=lambda path: copy.deepcopy(details))
        return manifest, details, api, validator

    def test_upload_intent_exists_before_one_mutation_and_original_ledger_stays_frozen(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest, details, api, validator = self.fixture(root)
            def uploaded(path):
                intent = submit.read_json(path)
                self.assertEqual(intent["status"], "upload_attempt_started")
                self.assertIsNone(intent["submission_ref"])
                self.assertEqual(submit.sha256(intent["staged_csv"]), details["submission_sha256"])
                return {"ref": 123}
            api.submit.side_effect = uploaded
            receipt = submit.submit_once(manifest, api, root, validator, timeout=0)
            self.assertEqual(receipt["public_score"], .9)
            self.assertIsNone(receipt["private_score"])
            self.assertEqual(validator.call_count, 2)
            api.submit.assert_called_once()
            self.assertEqual((root/"results/kaggle_submissions.json").read_text(), "[]")
            self.assertTrue((root/"results/research/submission_receipts/123.json").is_file())
            self.assertFalse((root/"results/research/submission_upload.lock").exists())
            self.assertEqual(submit.submit_once(manifest, api, root, validator, timeout=0)["submission_ref"], 123)
            api.submit.assert_called_once()

    def test_exhausted_or_inconsistent_quota_creates_no_upload_intent(self):
        for limits in ({"num_today": 5, "num_allowed_now": 0}, {"num_today": 1, "num_allowed_now": 5},
                       {"num_today": True, "num_allowed_now": 4}):
            with self.subTest(limits=limits), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                manifest, _, api, validator = self.fixture(root)
                api.limits.return_value = limits
                with self.assertRaises((RuntimeError, ValueError)):
                    submit.submit_once(manifest, api, root, validator, timeout=0)
                api.submit.assert_not_called()
                self.assertFalse((root/"results/research/submission_intents").exists())

    def test_legacy_csv_duplicate_is_reused_without_any_api_access(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest, details, api, validator = self.fixture(root)
            old = {"submission_sha256": details["submission_sha256"], "submission_ref": 77, "status": "complete", "public_score": .874}
            (root/"results/kaggle_submissions.json").write_text(json.dumps([old]))
            self.assertEqual(submit.submit_once(manifest, api, root, validator), old)
            self.assertEqual(api.mock_calls, [])

    def test_unknown_upload_never_retries_even_on_a_later_utc_day(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest, details, api, validator = self.fixture(root)
            api.submit.side_effect = TimeoutError("unknown response")
            with self.assertRaises(TimeoutError): submit.submit_once(manifest, api, root, validator, timeout=0)
            intent = root/"results/research/submission_intents"/(details["submission_sha256"]+".json")
            self.assertEqual(submit.read_json(intent)["status"], "unknown")
            with patch.object(submit, "now", return_value=datetime(2026, 11, 2, tzinfo=timezone.utc)):
                with self.assertRaisesRegex(RuntimeError, "unknown"):
                    submit.submit_once(manifest, api, root, validator, timeout=0)
            api.submit.assert_called_once()

    def test_missing_reference_can_only_recover_by_reading_unique_official_marker(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest, details, api, validator = self.fixture(root)
            api.submit.return_value = {"success": True}
            with self.assertRaisesRegex(RuntimeError, "unambiguous"):
                submit.submit_once(manifest, api, root, validator, timeout=0)
            api.recent.return_value = [{"ref": 123, "description": "sha256:"+details["submission_sha256"], "team_name": "Saber"}]
            self.assertEqual(submit.submit_once(manifest, api, root, validator, timeout=0)["submission_ref"], 123)
            api.submit.assert_called_once()

    def test_partial_invalid_delivery_and_hard_stop_fail_before_any_api_access(self):
        for invalid in (True, False):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                manifest, _, api, validator = self.fixture(root)
                if invalid: validator.side_effect = ValueError("partial dev or missing independent repeat")
                else:
                    policy = submit.read_json(root/"research/campaign.json")
                    policy["stops_at"] = "2000-01-01T00:00:00Z"
                    (root/"research/campaign.json").write_text(json.dumps(policy))
                with self.assertRaises((ValueError, RuntimeError)):
                    submit.submit_once(manifest, api, root, validator, timeout=0)
                self.assertEqual(api.mock_calls, [])

    def test_second_validation_change_does_not_create_an_ambiguous_intent(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest, details, api, validator = self.fixture(root)
            validator.side_effect = [details, {**details, "checkpoint_sha256": "d"*64}]
            with self.assertRaisesRegex(RuntimeError, "changed"):
                submit.submit_once(manifest, api, root, validator, timeout=0)
            api.submit.assert_not_called()
            self.assertFalse((root/"results/research/submission_intents").exists())

    def test_known_reference_score_check_is_read_only_after_campaign_stop(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest, _, api, validator = self.fixture(root)
            api.get.return_value.update(status="PENDING", public_score=None)
            result = submit.submit_once(manifest, api, root, validator, timeout=0)
            self.assertEqual(result["status"], "scoring_pending")
            (root/"research/campaign.json").write_text("{}")  # Check does not start any new submission.
            api.reset_mock()
            api.get.return_value.update(status="COMPLETE", public_score=".91")
            self.assertEqual(submit.check_known(123, api, root, timeout=0)["public_score"], .91)
            api.get.assert_called_once_with(123)
            api.submit.assert_not_called()
            api.limits.assert_not_called()
            api.recent.assert_not_called()
            with self.assertRaisesRegex(ValueError, "locally known"):
                submit.check_known(456, api, root, timeout=0)

    def test_rescored_reference_preserves_prior_official_score_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest, _, api, validator = self.fixture(root)
            submit.submit_once(manifest, api, root, validator, timeout=0)
            api.get.return_value["public_score"] = ".91"
            self.assertEqual(submit.check_known(123, api, root, timeout=0)["public_score"], .91)
            history = list((root/"results/research/submission_receipt_history").glob("123_*.json"))
            self.assertEqual(len(history), 1)
            self.assertEqual(submit.read_json(history[0])["public_score"], .9)
            api.submit.assert_called_once()

    def test_research_recovery_requires_full_csv_hash_marker(self):
        digest = "a"*64
        self.assertIsNone(submit.matching_recent([{"ref": 1, "description": "sha256:"+digest[:16]}], digest))
        self.assertEqual(submit.matching_recent([{"ref": 1, "description": "sha256:"+digest}], digest)["ref"], 1)

    def test_conflicting_references_and_existing_upload_lock_block_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest, details, api, validator = self.fixture(root)
            old = [{"submission_sha256": details["submission_sha256"], "submission_ref": ref} for ref in (1, 2)]
            (root/"results/kaggle_submissions.json").write_text(json.dumps(old))
            with self.assertRaisesRegex(RuntimeError, "conflict"):
                submit.submit_once(manifest, api, root, validator, timeout=0)
            self.assertEqual(api.mock_calls, [])
            (root/"results/kaggle_submissions.json").write_text("[]")
            submit.durable_json(root/"results/research/submission_upload.lock", {"token": "other"}, exclusive=True)
            with self.assertRaises(FileExistsError): submit.submit_once(manifest, api, root, validator, timeout=0)
            self.assertEqual(api.mock_calls, [])

    def test_multiple_remote_matches_or_unresolved_other_csv_block_new_uploads(self):
        for multiple in (True, False):
            with self.subTest(multiple=multiple), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                manifest, details, api, validator = self.fixture(root)
                if multiple:
                    api.recent.return_value = [{"ref": ref, "description": "sha256:"+details["submission_sha256"]} for ref in (123, 124)]
                else:
                    submit.durable_json(root/"results/research/submission_intents/other.json",
                                        {"submission_sha256": "e"*64, "submission_ref": None, "status": "unknown"})
                with self.assertRaisesRegex(RuntimeError, "Multiple|unresolved"):
                    submit.submit_once(manifest, api, root, validator, timeout=0)
                api.submit.assert_not_called()
                api.limits.assert_not_called()

    def test_final_transport_revalidates_real_artifacts_before_importing_or_authenticating_kaggle(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, details, _, _ = self.fixture(root)
            intent_path = root/"intent.json"
            intent = {**details, "status": "upload_attempt_started", "submission_ref": None,
                      "lock_token": "owner", "validation_python": sys.executable,
                      "manifest_path": "delivery.json", "validated": details}
            submit.durable_json(intent_path, intent)
            submit.durable_json(root/"results/research/submission_upload.lock", {"token": "owner"})
            with patch.object(submit, "ROOT", root), \
                    patch.object(submit.subprocess, "run", return_value=Mock(returncode=1, stdout="", stderr="private failure")) as child, \
                    patch.object(submit.importlib.metadata, "version", side_effect=AssertionError("no Kaggle import/auth")):
                with self.assertRaisesRegex(ValueError, "validation failed"):
                    submit.transport("submit", {"intent_path": str(intent_path), "intent_sha256": submit.sha256(intent_path)})
                self.assertIn("--_validate", child.call_args.args[0])

    def test_prepare_validator_checks_actual_official_csv_and_creates_no_intent(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest, details, _, _ = self.fixture(root)
            (root/"data").mkdir()
            sample = root/"data/sample_submission.csv"
            sample.write_bytes(Path(details["csv_path"]).read_bytes())
            (root/"research/delivery.py").write_text("# strict artifact validator fixture\n")
            delivery = ModuleType("research.delivery")
            delivery.validate_delivery = Mock(return_value=details)
            with patch.dict(sys.modules, {"research.delivery": delivery}), patch.object(submit, "SAMPLE_SHA", submit.sha256(sample)):
                result = submit.validate_manifest(manifest, root)
                self.assertEqual(result["submission_sha256"], details["submission_sha256"])
                delivery.validate_delivery.assert_called_once_with(manifest, require_prediction=True)
                self.assertFalse((root/"results/research/submission_intents").exists())
                Path(details["csv_path"]).write_text("id,label\n0,10\n")
                with self.assertRaises(ValueError): submit.validate_manifest(manifest, root)

    def test_refit_schema_dispatch_requires_its_dedicated_parent_and_zero_dev_validator(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest, details, _, _ = self.fixture(root)
            manifest.write_text(json.dumps({"schema": "research_refit_delivery_v1"}))
            (root/"data").mkdir()
            sample = root/"data/sample_submission.csv"
            sample.write_bytes(Path(details["csv_path"]).read_bytes())
            (root/"research/refit_delivery.py").write_text("# independently audited full-refit validator fixture")
            refit = ModuleType("research.refit_delivery")
            full = {**details, "manifest_sha256": submit.sha256(manifest), "training_examples": 50000,
                    "validation_examples": 0, "development_accuracy": None, "independent_development_evaluated": False,
                    "parent_delivery_sha256": "f"*64}
            refit.validate_refit_delivery = Mock(return_value=full)
            primary = ModuleType("research.delivery")
            primary.validate_delivery = Mock(side_effect=AssertionError("wrong validator"))
            with patch.dict(sys.modules, {"research.refit_delivery": refit, "research.delivery": primary}), \
                    patch.object(submit, "SAMPLE_SHA", submit.sha256(sample)):
                result = submit.validate_manifest(manifest, root)
                self.assertEqual(result["training_examples"], 50000)
                self.assertEqual(result["delivery_schema"], "research_refit_delivery_v1")
                refit.validate_refit_delivery.assert_called_once_with(manifest, require_prediction=True)
                primary.validate_delivery.assert_not_called()
                refit.validate_refit_delivery.return_value = {**full, "validation_examples": 5000}
                with self.assertRaisesRegex(ValueError, "zero independent"):
                    submit.validate_manifest(manifest, root)
                self.assertFalse((root/"results/research/submission_intents").exists())

    def test_training_count_alone_cannot_upgrade_old_or_unknown_delivery_schema(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest, details, _, _ = self.fixture(root)
            primary = ModuleType("research.delivery")
            primary.validate_delivery = Mock(return_value={**details, "training_examples": 50000})
            with patch.dict(sys.modules, {"research.delivery": primary}):
                with self.assertRaisesRegex(ValueError, "audited supported"):
                    submit.validate_manifest(manifest, root)
            manifest.write_text(json.dumps({"schema": "unreviewed_refit_v9", "training_examples": 50000}))
            with self.assertRaisesRegex(ValueError, "Unsupported research delivery schema"):
                submit.validate_manifest(manifest, root)

    def test_v2_schema_dispatch_calls_versioned_validator_and_preserves_old_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest, details, _, _ = self.fixture(root)
            manifest.write_text(json.dumps({"schema": "research_prediction_delivery_v2"}))
            (root/"data").mkdir()
            sample = root/"data/sample_submission.csv"
            sample.write_bytes(Path(details["csv_path"]).read_bytes())
            (root/"research/delivery_v2.py").write_text("# source-bound v2 validator fixture")
            versioned = ModuleType("research.delivery_v2")
            versioned.validate_delivery = Mock(return_value={**details, "manifest_sha256": submit.sha256(manifest)})
            original = ModuleType("research.delivery")
            original.validate_delivery = Mock(side_effect=AssertionError("v1 validator cannot authorize v2"))
            with patch.dict(sys.modules, {"research.delivery_v2": versioned, "research.delivery": original}), \
                    patch.object(submit, "SAMPLE_SHA", submit.sha256(sample)):
                result = submit.validate_manifest(manifest, root)
                self.assertEqual(result["delivery_schema"], "research_prediction_delivery_v2")
                versioned.validate_delivery.assert_called_once_with(manifest, require_prediction=True)
                original.validate_delivery.assert_not_called()
                self.assertFalse((root/"results/research/submission_intents").exists())


if __name__ == "__main__": unittest.main()
