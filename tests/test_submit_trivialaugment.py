"""TA uploader contracts with synthetic CSVs and fake APIs only."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from research import submit_trivialaugment as upload


class TrivialAugmentSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        for folder in ("research", "results", "data"):
            (self.root/folder).mkdir()
        self.write("research/campaign.json", {"team": "Saber", "stops_at": "2999-01-01T00:00:00Z",
            "submission_limit_per_team_utc_day": 5})
        self.write("results/kaggle_submissions.json", [])
        for filename in ("run_trivialaugment_repeat.py", "trivialaugment_delivery.py"):
            (self.root/"research"/filename).write_text("# synthetic source fixture\n")
        self.csv = self.root/"submission.csv"
        self.csv.write_text("id,label\n"+"".join(f"{i},{i%10}\n" for i in range(2000)))
        sample = self.root/"data/sample_submission.csv"
        sample.write_bytes(self.csv.read_bytes())
        self.manifest = self.write("delivery.json", {"schema": upload.SCHEMA})
        self.details = {"run": upload.PRIMARY_RUN, "competition": upload.shared.COMPETITION,
            "csv_path": str(self.csv), "submission_sha256": upload.shared.sha256(self.csv),
            "checkpoint_sha256": upload.PRIMARY_CHECKPOINT, "parameter_count": 4166298,
            "policy": "center80_shift20", "seed": 2026, "training_examples": 40000,
            "manifest_sha256": upload.shared.sha256(self.manifest), "frozen_manifest_sha256": upload.PRIMARY_FREEZE,
            "repeat_receipt_schema": upload.REPEAT_SCHEMA, "repeat_receipt_sha256": "a"*64,
            "repeat_validator_source_sha256": upload.shared.sha256(self.root/"research/run_trivialaugment_repeat.py")}
        self.delivery = ModuleType("research.trivialaugment_delivery")
        self.delivery.validate_delivery = Mock(side_effect=lambda *args, **kwargs: copy.deepcopy(self.details))
        self.enter = contextlib.ExitStack()
        self.addCleanup(self.enter.close)
        self.enter.enter_context(patch.dict(sys.modules, {self.delivery.__name__: self.delivery}))
        self.enter.enter_context(patch.object(upload.shared, "SAMPLE_SHA", upload.shared.sha256(sample)))
        self.api = Mock()
        self.api.recent.return_value = []
        self.api.limits.return_value = {"num_today": 0, "num_allowed_now": 5}
        self.api.submit.return_value = {"ref": 123}
        self.api.get.return_value = {"ref": 123, "status": "COMPLETE", "public_score": ".91",
            "team_name": "Saber", "description": "sha256:"+self.details["submission_sha256"]}

    def write(self, relative, value):
        path = self.root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def validated(self):
        return upload.validate_manifest(self.manifest, self.root)

    def intent(self):
        details = self.validated()
        digest = details["submission_sha256"]
        staged = self.root/"results/research/upload_files"/digest/"submission.csv"
        staged.parent.mkdir(parents=True)
        staged.write_bytes(self.csv.read_bytes())
        self.write("results/research/submission_upload.lock", {"token": "owned"})
        intent = {**details, "team": "Saber", "submission_ref": None, "status": "upload_attempt_started",
            "validated": details, "lock_token": "owned", "staged_csv": str(staged),
            "manifest_path": str(self.manifest), "validation_python": sys.executable,
            "description": "Saber research sha256:"+digest}
        path = self.write("results/research/submission_intents/"+digest+".json", intent)
        return path, intent, {"intent_path": str(path), "intent_sha256": upload.shared.sha256(path)}

    def sdk(self, api):
        module = ModuleType("kaggle.api.kaggle_api_extended")
        module.KaggleApi = Mock(return_value=api)
        return {"kaggle": ModuleType("kaggle"), "kaggle.api": ModuleType("kaggle.api"), module.__name__: module}

    def test_fixed_primary_and_repeat_contract_rejects_wrong_types_or_weights(self):
        result = self.validated()
        self.assertEqual(result["delivery_schema"], upload.SCHEMA)
        self.delivery.validate_delivery.assert_called_with(self.manifest, require_prediction=True)
        for key, value in (("run", "wrn34_trivialaugment_bf16_s2027"), ("seed", 2027),
            ("parameter_count", 4166298.0), ("training_examples", 50000), ("policy", "flip"),
            ("checkpoint_sha256", "b"*64), ("frozen_manifest_sha256", "c"*64),
            ("repeat_receipt_schema", "research_native_repeat_v1"), ("repeat_receipt_sha256", "")):
            with self.subTest(key=key):
                original = self.details[key]
                self.details[key] = value
                with self.assertRaises(ValueError):
                    self.validated()
                self.details[key] = original

    def test_old_schema_is_rejected_before_delivery_dispatch(self):
        self.write("delivery.json", {"schema": "research_prediction_delivery_v1"})
        with self.assertRaisesRegex(ValueError, "explicit TrivialAugment"):
            self.validated()
        self.delivery.validate_delivery.assert_not_called()

    def test_actual_csv_ids_and_content_not_manifest_claims(self):
        original = self.csv.read_bytes()
        for content in ("id,label\n0,0\n", original.decode().replace("1,1\n", "0,1\n", 1),
                        original.decode().replace("1,1\n", "1,10\n", 1)):
            with self.subTest(content=content[:24]):
                self.csv.write_text(content)
                with self.assertRaises(ValueError):
                    self.validated()
        self.csv.write_bytes(original)
        self.details["submission_sha256"] = "b"*64
        with self.assertRaisesRegex(ValueError, "CSV or manifest"):
            self.validated()

    def test_manifest_and_repeat_source_are_rehashed(self):
        self.manifest.write_text(json.dumps({"schema": upload.SCHEMA, "changed": True}))
        with self.assertRaisesRegex(ValueError, "CSV or manifest"):
            self.validated()
        self.details["manifest_sha256"] = upload.shared.sha256(self.manifest)
        (self.root/"research/run_trivialaugment_repeat.py").write_text("# modified\n")
        with self.assertRaisesRegex(ValueError, "repeat validator source"):
            self.validated()

    def test_prepare_never_instantiates_api_or_creates_intent(self):
        with patch.object(upload, "KaggleClient") as client, patch.object(upload, "validate_manifest", return_value=self.validated()), contextlib.redirect_stdout(io.StringIO()) as output:
            upload.main(["--prepare", "--manifest", str(self.manifest)])
        self.assertIs(json.loads(output.getvalue())["upload_intent_created"], False)
        client.assert_not_called()
        self.assertFalse((self.root/"results/research/submission_intents").exists())

    def test_client_uses_only_new_transport_for_every_action_and_redacts_errors(self):
        client = upload.KaggleClient(sys.executable)
        with patch.object(upload.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout='{"ok":true,"result":{}}')) as run:
            for action in ("recent", "limits", "get", "submit"):
                client.call(action, {})
                command = run.call_args.args[0]
                self.assertEqual(Path(command[1]), Path(upload.__file__).resolve())
                self.assertEqual(command[2:], ["--_transport", action])
        with patch.object(upload.subprocess, "run", return_value=SimpleNamespace(returncode=1, stdout='{"ok":false,"status_code":403,"secret":"credential"}')):
            with self.assertRaises(upload.shared.KaggleError) as error:
                client.limits()
        self.assertNotIn("credential", str(error.exception))

    def test_once_only_upload_preserves_typed_receipt_and_durable_intent(self):
        def mutation(path):
            intent = upload.shared.read_json(path)
            self.assertEqual(intent["status"], "upload_attempt_started")
            self.assertEqual(intent["delivery_schema"], upload.SCHEMA)
            self.assertIsNone(intent["submission_ref"])
            return {"ref": 123}
        self.api.submit.side_effect = mutation
        receipt = upload.submit_once(self.manifest, self.api, self.root, timeout=0)
        self.assertEqual(receipt["repeat_receipt_schema"], upload.REPEAT_SCHEMA)
        self.assertEqual(receipt["public_score"], .91)
        self.assertIsNone(receipt["private_score"])
        self.api.submit.assert_called_once()
        self.api.reset_mock()
        existing = upload.submit_once(self.manifest, self.api, self.root, timeout=0)
        self.assertEqual(existing["submission_ref"], receipt["submission_ref"])
        self.assertEqual(existing["submission_sha256"], receipt["submission_sha256"])
        self.assertEqual(self.api.mock_calls, [])
        self.assertEqual((self.root/"results/kaggle_submissions.json").read_text(), "[]")

    def test_invalid_chain_and_whole_team_quota_never_create_intents(self):
        self.details["seed"] = 2027
        with self.assertRaises(ValueError):
            upload.submit_once(self.manifest, self.api, self.root, timeout=0)
        self.assertEqual(self.api.mock_calls, [])
        self.details["seed"] = 2026
        self.api.limits.return_value = {"num_today": 5, "num_allowed_now": 0}
        with self.assertRaisesRegex(RuntimeError, "whole-team"):
            upload.submit_once(self.manifest, self.api, self.root, timeout=0)
        self.api.submit.assert_not_called()
        self.assertFalse((self.root/"results/research/submission_intents").exists())

    def test_unknown_outcome_cannot_resend_and_only_exact_reference_recovers(self):
        self.api.submit.side_effect = TimeoutError("synthetic unknown")
        with self.assertRaises(TimeoutError):
            upload.submit_once(self.manifest, self.api, self.root, timeout=0)
        with self.assertRaisesRegex(RuntimeError, "unknown"):
            upload.submit_once(self.manifest, self.api, self.root, timeout=0)
        self.api.submit.assert_called_once()
        self.api.recent.return_value = [{"ref": 123, "team_name": "Saber",
            "description": "sha256:"+self.details["submission_sha256"]}]
        self.assertEqual(upload.submit_once(self.manifest, self.api, self.root, timeout=0)["submission_ref"], 123)
        self.api.submit.assert_called_once()

    def test_known_reference_check_is_read_only_and_unknown_ref_fails(self):
        upload.submit_once(self.manifest, self.api, self.root, timeout=0)
        self.api.reset_mock()
        self.assertEqual(upload.shared.check_known(123, self.api, self.root, timeout=0)["submission_ref"], 123)
        self.api.get.assert_called_once_with(123)
        self.api.submit.assert_not_called()
        self.api.limits.assert_not_called()
        self.api.recent.assert_not_called()
        with self.assertRaises(ValueError):
            upload.shared.check_known(999, self.api, self.root, timeout=0)

    def test_final_transport_revalidates_own_schema_before_authentication(self):
        path, intent, payload = self.intent()
        response = SimpleNamespace(returncode=0, stdout=json.dumps(intent["validated"]))
        with patch.object(upload.subprocess, "run", return_value=response) as run:
            self.assertEqual(upload.checked_intent(payload, self.root), intent)
            command = run.call_args.args[0]
            self.assertEqual(command[1:3], [str(Path(upload.__file__).resolve()), "--_validate"])
            response.stdout = json.dumps({**intent["validated"], "checkpoint_sha256": "d"*64})
            with self.assertRaisesRegex(ValueError, "artifact or staged"):
                upload.checked_intent(payload, self.root)
        self.write("results/research/submission_upload.lock", {"token": "other"})
        with patch.object(upload.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "owner lock"):
                upload.checked_intent(payload, self.root)
            run.assert_not_called()
        intent["status"] = "unknown"
        self.write(str(path.relative_to(self.root)), intent)
        payload["intent_sha256"] = upload.shared.sha256(path)
        with self.assertRaisesRegex(ValueError, "fresh canonical"):
            upload.checked_intent(payload, self.root)

    def test_official_transport_checks_fresh_quota_and_staged_bytes_again(self):
        _, intent, payload = self.intent()
        api = Mock()
        api.competition_get_submission_limits.return_value = SimpleNamespace(num_today=0, num_allowed_now=5)
        api.competition_submit.return_value = SimpleNamespace(ref=123)
        with patch.dict(sys.modules, self.sdk(api)), patch.object(upload.importlib.metadata, "version", return_value=upload.shared.KAGGLE_VERSION), patch.object(upload.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=json.dumps(intent["validated"]))):
            self.assertEqual(upload.transport("submit", payload, self.root), {"ref": 123})
            api.competition_submit.assert_called_once()
            api.reset_mock()
            api.competition_get_submission_limits.return_value = SimpleNamespace(num_today=5, num_allowed_now=0)
            with self.assertRaisesRegex(RuntimeError, "whole-team"):
                upload.transport("submit", payload, self.root)
            api.competition_submit.assert_not_called()
            def quota_changes_csv(*args):
                Path(intent["staged_csv"]).write_text("changed")
                return SimpleNamespace(num_today=0, num_allowed_now=5)
            api.competition_get_submission_limits.side_effect = quota_changes_csv
            with self.assertRaisesRegex(ValueError, "after final quota"):
                upload.transport("submit", payload, self.root)
            api.competition_submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
