"""New WRN300 refit boundaries, using synthetic CPU files and fake APIs only."""
from contextlib import ExitStack
from datetime import datetime, timezone
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from data import CSV_SHA256
from research import wrn300_refit_delivery as delivery
from research import submit_wrn300_refit as uploader


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class WRN300RefitDeliveryTests(unittest.TestCase):
    def fixture(self, root):
        job = root/"results/research/wrn300_full_s2026"
        prediction = root/"results/research/wrn300_full_prediction"
        data = root/"data"
        for path in (job, prediction, data, root/"research"):
            path.mkdir(parents=True, exist_ok=True)
        for name in ("train.py", "research/run_wrn300_refit.py", "research/process_tree_guard.py", "research/submit_wrn300_refit.py"):
            (root/name).write_text("# independent synthetic source fixture\n", encoding="utf-8")
        config = {"epochs": 300, "seed": 2026, "split_seed": 2026, "train_all": False,
            "final_variant": "ema", "depth": 34, "widen_factor": 3, "dropout": 0.0,
            "num_classes": 10, "batch_size": 128}
        frozen = {"config": config, "weight_variant": "model", "policy": "center80_shift20",
            "model_config": {"depth": 34, "widen_factor": 3, "dropout": 0.0, "num_classes": 10},
            "data_signature": {"archive_sha256": "a"*64, "split_sha256": "b"*64},
            "evaluation_context": {"device_type": "cuda", "amp_dtype": "torch.bfloat16"},
            "data_dir": str(data), "benchmark_path": str(root/"parent_benchmark.json")}
        frozen_path, parent_path, plan = root/"frozen.json", root/"parent_delivery.json", job/"refit_plan.json"
        write(frozen_path, frozen)
        write(parent_path, {"schema": "research_prediction_delivery_v2", "frozen_manifest_path": str(frozen_path)})
        write(plan, {"schema": "synthetic_only_refit_plan"})
        parent = {"manifest_sha256": delivery.shared.sha256(parent_path), "frozen_manifest_sha256": delivery.shared.sha256(frozen_path),
            "checkpoint_sha256": delivery.PARENT_CHECKPOINT_SHA, "training_examples": 40000,
            "seed": 2026, "policy": "center80_shift20", "repeat_receipt_schema": "research_shadow_cloud_repeat_attestation_v1"}
        for name in delivery.ARTIFACT_NAMES:
            write(job/name, {"synthetic_file": name})
        write(job/"supervisor.json", {"finished_at": "2026-11-01T00:01:00+00:00"})
        info = {"run": job.name, "plan_sha256": delivery.shared.sha256(plan),
            "parent_delivery_path": str(parent_path), "parent_delivery_sha256": parent["manifest_sha256"],
            "checkpoint_path": str(job/"best.pt"), "checkpoint_sha256": delivery.shared.sha256(job/"best.pt"),
            "training_examples": 50000, "validation_examples": 0, "epochs_completed": 300,
            "training_precision": "fp16", "weight_variant": "model", "initial_grad_scaler_scale": 65536.0,
            "attempted_updates": 117300, "source_examples_seen": 15000000, "parameter_count": 4166298,
            "policy": "center80_shift20", "config": {**config, "train_all": True, "final_variant": "model"},
            "data_signature": frozen["data_signature"], "evaluation_context": frozen["evaluation_context"],
            "source_git_commit": "d"*40,
            "source_sha256": {"train.py": delivery.shared.sha256(root/"train.py")}}
        info["evidence_paths"] = list(map(str, [plan, parent_path, frozen_path, *(job/name for name in delivery.ARTIFACT_NAMES)]))
        runner = ModuleType("research.run_wrn300_refit")
        runner.SUPERVISOR_SCHEMA = "research_wrn300_refit_local_supervisor_v1"
        runner.validate_refit = Mock(side_effect=lambda path: copy.deepcopy(info))
        runner.validate_worker = Mock(side_effect=RuntimeError("synthetic inactive guard"))
        sample = data/"sample_submission.csv"
        sample.write_text("id,label\n"+"".join(f"{i},{i%10}\n" for i in range(2000)), encoding="utf-8")
        for name in ("test.csv", "official_test.npz", "official_test_source_manifest.json"):
            (data/name).write_text("synthetic source file", encoding="utf-8")
        test_paths = [data/name for name in ("sample_submission.csv", "test.csv", "official_test.npz", "official_test_source_manifest.json")]
        test_manifest = data/"kaggle_test_manifest.json"
        write(test_manifest, {"competition": delivery.shared.COMPETITION, "count": 2000,
            "shape": [2000, 32, 32, 3], "dtype": "uint8", "layout": "NHWC", "order": "sample_submission.csv ID order",
            "labels_loaded": False, "test_csv_sha256": delivery.shared.sha256(data/"test.csv"),
            "sample_submission_sha256": delivery.shared.sha256(sample), "source_provenance": {
                "archive_sha256": delivery.shared.sha256(data/"official_test.npz"),
                "source_manifest_sha256": delivery.shared.sha256(data/"official_test_source_manifest.json")}})
        write(root/"research/campaign.json", {"team": "Saber", "submission_limit_per_team_utc_day": 5, "stops_at": "2999-01-01T00:00:00Z"})
        (root/"results/kaggle_submissions.json").write_text("[]")
        return SimpleNamespace(root=root, job=job, prediction=prediction, data=data, plan=plan, parent=parent,
            frozen=frozen, info=info, runner=runner, test_paths=test_paths, test_manifest=test_manifest)

    def patches(self, f):
        stack = ExitStack()
        stack.enter_context(patch.dict(sys.modules, {"research.run_wrn300_refit": f.runner}))
        stack.enter_context(patch.object(delivery, "ROOT", f.root))
        stack.enter_context(patch.object(delivery.shared, "validate_manifest", return_value=f.parent))
        stack.enter_context(patch.object(delivery, "PARENT_DELIVERY_SHA", f.parent["manifest_sha256"]))
        stack.enter_context(patch.object(delivery, "PARENT_FREEZE_SHA", f.parent["frozen_manifest_sha256"]))
        stack.enter_context(patch.object(delivery, "test_inputs", return_value=f.test_paths))
        stack.enter_context(patch.object(delivery.shared, "SAMPLE_SHA", delivery.shared.sha256(f.data/"sample_submission.csv")))
        stack.enter_context(patch.dict(CSV_SHA256, {"test.csv": delivery.shared.sha256(f.data/"test.csv")}))
        return stack

    def prediction_fixture(self, f):
        _, _, _, paths = delivery.validate_refit_run(f.plan)
        manifest = f.prediction/"delivery.json"
        csv_path = f.prediction/"submission.csv"
        csv_path.write_bytes((f.data/"sample_submission.csv").read_bytes())
        inputs_path = f.prediction/"refit_inputs.json"
        binding = delivery.snapshots([*paths, *f.test_paths])
        write(inputs_path, {"mode": "predict", "plan_path": str(f.plan),
            "plan_sha256": f.info["plan_sha256"], "source_sha256": f.info["source_sha256"], "input_sha256": binding})
        document = {"schema": delivery.SCHEMA, "status": "complete", "created_at": "2026-11-01T00:02:01+00:00",
            **delivery.metadata(f.plan, f.info, f.parent, f.frozen), "refit_evidence_sha256": delivery.snapshots(paths),
            "csv_path": str(csv_path), "submission_sha256": delivery.shared.sha256(csv_path), "rows": 2000,
            "columns": ["id", "label"], "ids_match_official_order": True,
            "test_manifest_path": str(f.test_manifest), "test_files_sha256": delivery.snapshots([*f.test_paths, f.test_manifest]),
            "prediction_input_manifest_sha256": delivery.shared.sha256(inputs_path)}
        write(manifest, document)
        nonce = "1"*32
        guard = f.prediction/("process_tree_"+nonce)
        script_sha = delivery.shared.sha256(f.root/"research/run_wrn300_refit.py")
        guard_sha = delivery.shared.sha256(f.root/"research/process_tree_guard.py")
        identity = {"schema": "owned_python_process_tree_v1", "nonce": nonce, "pid": 222,
            "created_at": 12345.0, "script_sha256": script_sha, "method": "windows_job_v1"}
        write(guard/"ready.json", identity)
        write(guard/"completion.json", {**identity, "exit_code": 0})
        deadline = datetime(2026, 11, 1, 0, 5, tzinfo=timezone.utc).timestamp()
        write(guard/"spec.json", {"schema": identity["schema"], "nonce": nonce, "host_pid": 111,
            "host_created_at": 12340.0, "cwd": str(f.root), "script": str(f.root/"research/run_wrn300_refit.py"),
            "script_sha256": script_sha, "guard_source_sha256": guard_sha, "deadline_unix": deadline,
            "job_name": "Local\\SaberAugMix_"+nonce,
            "arguments": ["--_worker", str(f.prediction), "--_manifest-sha", delivery.shared.sha256(inputs_path)]})
        proof = {"schema": identity["schema"], "nonce": nonce, "method": identity["method"], "guard_directory": str(guard),
            "guard_source_sha256": guard_sha, "cleanup_verified": True, "child_pid": 222, "child_created_at": 12345.0,
            "timed_out": False, "child_returncode": 0, "completion_authenticated": True, "completion_exit_code": 0}
        write(guard/"host_result.json", proof)
        receipt = {"schema": f.runner.SUPERVISOR_SCHEMA, "status": "complete", "backend": "local", "mode": "predict",
            "run": f.prediction.name, "trainer": "wrn300_full_refit", "precision": "parent_frozen_inference",
            "plan_sha256": f.info["plan_sha256"], "submitted": False, "confirmation_evaluated": False,
            "process_tree_cleanup_verified": True, "artifact": "delivery.json", "artifact_sha256": delivery.shared.sha256(manifest),
            "input_manifest_sha256": delivery.shared.sha256(inputs_path), "source_sha256": f.info["source_sha256"],
            "input_sha256": binding, "started_at": "2026-11-01T00:02:00+00:00", "finished_at": "2026-11-01T00:02:02+00:00",
            "elapsed_seconds": 2.0, "reserved_seconds": 180, "deadline_unix": deadline,
            "pid": 111, "process_created_at": 12340.0, "child_pid": 222, "child_created_at": 12345.0,
            "process_tree_guard_results": [proof]}
        write(f.prediction/"supervisor.json", receipt)
        return manifest, document, receipt, guard

    def test_complete_new_delivery_has_zero_dev_and_fixed_final300_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(Path(directory))
            with self.patches(f):
                manifest, _, _, _ = self.prediction_fixture(f)
                result = delivery.validate_refit_delivery(manifest)
            self.assertEqual((result["training_examples"], result["validation_examples"]), (50000, 0))
            self.assertEqual((result["checkpoint_epoch"], result["weight_variant"]), (300, "model"))
            self.assertIsNone(result["development_accuracy"])
            self.assertFalse(result["independent_development_evaluated"])
            self.assertEqual(result["parent_delivery_sha256"], f.parent["manifest_sha256"])
            f.runner.validate_refit.assert_called()

    def test_wrong_recipe_partial_updates_parent_or_repeat_weights_are_rejected(self):
        changes = [{"training_examples": 40000}, {"validation_examples": 5000}, {"epochs_completed": 299},
            {"attempted_updates": 117299}, {"source_examples_seen": 14999999}, {"weight_variant": "ema"},
            {"training_precision": "bf16"}, {"initial_grad_scaler_scale": 256.0}, {"parameter_count": 5000001},
            {"checkpoint_sha256": delivery.PARENT_CHECKPOINT_SHA}, {"parent_delivery_sha256": "d"*64},
            {"config": {"epochs": 300, "train_all": True, "seed": 2027}}, {"validation_examples": False}]
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                f = self.fixture(Path(directory))
                with self.patches(f):
                    f.info.update(change)
                    with self.assertRaises(ValueError):
                        delivery.validate_refit_run(f.plan)

    def test_missing_actual_training_proof_or_changed_source_stops_delivery(self):
        for missing in (True, False):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                f = self.fixture(Path(directory))
                with self.patches(f):
                    if missing:
                        f.info["evidence_paths"].remove(str(f.job/"last.pt"))
                    else:
                        (f.root/"train.py").write_text("changed after training")
                    with self.assertRaises(ValueError):
                        delivery.validate_refit_run(f.plan)

    def test_manifest_cannot_claim_new_dev_or_change_frozen_policy(self):
        for key, value in (("development_accuracy", .99), ("validation_examples", 5000),
                ("independent_development_evaluated", True), ("policy", "flip"),
                ("checkpoint_epoch", 298), ("training_precision", "bf16"), ("bn_recalibrated", True)):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                f = self.fixture(Path(directory))
                with self.patches(f):
                    manifest, document, receipt, _ = self.prediction_fixture(f)
                    document[key] = value
                    write(manifest, document)
                    receipt["artifact_sha256"] = delivery.shared.sha256(manifest)
                    write(f.prediction/"supervisor.json", receipt)
                    with self.assertRaises(ValueError):
                        delivery.validate_refit_delivery(manifest)

    def test_real_csv_order_row_count_and_integer_labels_are_checked(self):
        for kind in ("label", "order", "count"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                f = self.fixture(Path(directory))
                with self.patches(f):
                    manifest, document, receipt, _ = self.prediction_fixture(f)
                    csv_path = Path(document["csv_path"])
                    lines = csv_path.read_text().splitlines()
                    if kind == "label": lines[1] = "0,10"
                    elif kind == "order": lines[1], lines[2] = lines[2], lines[1]
                    else: lines.pop()
                    csv_path.write_text("\n".join(lines)+"\n")
                    document["submission_sha256"] = delivery.shared.sha256(csv_path)
                    write(manifest, document)
                    receipt["artifact_sha256"] = delivery.shared.sha256(manifest)
                    write(f.prediction/"supervisor.json", receipt)
                    with self.assertRaises(ValueError):
                        delivery.validate_refit_delivery(manifest)

    def test_cleanup_unknown_missing_guard_and_forged_completion_are_rejected(self):
        for kind in ("cleanup", "nonce", "exit_code", "guard_source", "input_coverage", "time"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                f = self.fixture(Path(directory))
                with self.patches(f):
                    manifest, document, receipt, guard = self.prediction_fixture(f)
                    if kind == "cleanup": receipt["process_tree_cleanup_verified"] = False
                    elif kind == "time": receipt["finished_at"] = "2026-11-01T00:01:59+00:00"
                    elif kind == "input_coverage":
                        inputs = delivery.read(f.prediction/"refit_inputs.json")
                        inputs["input_sha256"].pop(str(f.job/"last.pt"))
                        write(f.prediction/"refit_inputs.json", inputs)
                        receipt["input_sha256"] = inputs["input_sha256"]
                    else:
                        path = guard/("spec.json" if kind == "guard_source" else "completion.json")
                        proof = delivery.read(path)
                        proof[{"nonce": "nonce", "exit_code": "exit_code", "guard_source": "guard_source_sha256"}[kind]] = {"nonce": "2"*32, "exit_code": 1, "guard_source": "f"*64}[kind]
                        write(path, proof)
                    write(f.prediction/"supervisor.json", receipt)
                    with self.assertRaises(ValueError):
                        delivery.validate_refit_delivery(manifest)

    def test_uncontrolled_prediction_is_rejected_before_hidden_or_torch_work(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(Path(directory))
            with self.patches(f):
                with self.assertRaisesRegex(ValueError, "supervised"):
                    delivery.predict(f.plan, f.prediction)
                f.runner.validate_worker.assert_not_called()
                write(f.prediction/"refit_inputs.json", {})
                with self.assertRaisesRegex(RuntimeError, "inactive guard"):
                    delivery.predict(f.plan, f.prediction, guard_dir=f.prediction)
                f.runner.validate_worker.assert_called_once()


class WRN300SubmissionTests(unittest.TestCase):
    def fixture(self, root):
        for path in (root/"research", root/"results", root/"data"):
            path.mkdir(parents=True, exist_ok=True)
        write(root/"research/campaign.json", {"team": "Saber", "submission_limit_per_team_utc_day": 5, "stops_at": "2999-01-01T00:00:00Z"})
        (root/"results/kaggle_submissions.json").write_text("[]")
        (root/"research/wrn300_refit_delivery.py").write_text("# typed validation fixture")
        sample = root/"data/sample_submission.csv"
        sample.write_text("id,label\n"+"".join(f"{i},{i%10}\n" for i in range(2000)))
        csv_path = root/"submission.csv"
        csv_path.write_bytes(sample.read_bytes())
        manifest = root/"delivery.json"
        write(manifest, {"schema": uploader.SCHEMA})
        details = {"run": "wrn300_full_s2026", "competition": uploader.shared.COMPETITION,
            "csv_path": str(csv_path), "submission_sha256": uploader.shared.sha256(csv_path),
            "checkpoint_sha256": "a"*64, "manifest_sha256": uploader.shared.sha256(manifest),
            "frozen_manifest_sha256": "b"*64, "parent_delivery_sha256": "c"*64,
            "training_examples": 50000, "validation_examples": 0, "development_accuracy": None,
            "independent_development_evaluated": False, "seed": 2026, "parameter_count": 4166298,
            "policy": "center80_shift20", "epochs_completed": 300, "checkpoint_epoch": 300,
            "weight_variant": "model", "training_precision": "fp16", "delivery_schema": uploader.SCHEMA,
            "uploader_source_sha256": uploader.shared.sha256(Path(uploader.__file__))}
        api = Mock()
        api.recent.return_value = []
        api.limits.return_value = {"num_today": 3, "num_allowed_now": 2}
        api.submit.return_value = {"ref": 999001}
        api.get.return_value = {"ref": 999001, "status": "COMPLETE", "public_score": ".89", "team_name": "Saber",
            "description": "sha256:"+details["submission_sha256"]}
        validator = Mock(side_effect=lambda path: copy.deepcopy(details))
        return manifest, details, api, validator

    def test_new_validator_checks_official_csv_and_does_not_dispatch_old_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, details, _, _ = self.fixture(root)
            with patch.object(delivery, "validate_refit_delivery", return_value=details) as typed, \
                    patch.object(uploader.shared, "SAMPLE_SHA", uploader.shared.sha256(root/"data/sample_submission.csv")):
                result = uploader.validate_manifest(manifest, root)
                self.assertEqual(result["delivery_schema"], uploader.SCHEMA)
                typed.assert_called_once_with(manifest.resolve(), require_prediction=True)
                write(manifest, {"schema": "research_refit_delivery_v1"})
                with self.assertRaisesRegex(ValueError, "explicit WRN300"):
                    uploader.validate_manifest(manifest, root)
                self.assertFalse((root/"results/research/submission_intents").exists())

    def test_one_upload_persists_real_ref_and_reuses_shared_frozen_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, details, api, validator = self.fixture(root)
            def upload(path):
                intent = uploader.shared.read_json(path)
                self.assertEqual(intent["status"], "upload_attempt_started")
                self.assertIsNone(intent["submission_ref"])
                self.assertEqual(uploader.shared.sha256(intent["staged_csv"]), details["submission_sha256"])
                return {"ref": 999001}
            api.submit.side_effect = upload
            receipt = uploader.submit_once(manifest, api, root, validator, timeout=0)
            self.assertEqual((receipt["public_score"], receipt["delivery_schema"], receipt["validation_examples"]), (.89, uploader.SCHEMA, 0))
            self.assertIsNone(receipt["private_score"])
            self.assertEqual((root/"results/kaggle_submissions.json").read_text(), "[]")
            api.reset_mock()
            reused = uploader.submit_once(manifest, api, root, validator, timeout=0)
            self.assertEqual((reused["submission_ref"], reused["public_score"]), (999001, .89))
            self.assertEqual(api.mock_calls, [])

    def test_no_quota_or_partial_evidence_creates_no_upload_intent(self):
        for kind in ("quota", "partial", "hardstop"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest, _, api, validator = self.fixture(root)
                if kind == "quota": api.limits.return_value = {"num_today": 5, "num_allowed_now": 0}
                elif kind == "partial": validator.side_effect = ValueError("Incomplete final300 refit")
                else:
                    policy = uploader.shared.read_json(root/"research/campaign.json")
                    write(root/"research/campaign.json", {**policy, "stops_at": "2000-01-01T00:00:00Z"})
                with self.assertRaises((RuntimeError, ValueError)):
                    uploader.submit_once(manifest, api, root, validator, timeout=0)
                api.submit.assert_not_called()
                self.assertFalse((root/"results/research/submission_intents").exists())

    def test_unknown_upload_is_not_retried_and_unique_official_ref_can_recover(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, details, api, validator = self.fixture(root)
            api.submit.side_effect = TimeoutError("Synthetic unknown result")
            with self.assertRaises(TimeoutError): uploader.submit_once(manifest, api, root, validator, timeout=0)
            intent = root/"results/research/submission_intents"/(details["submission_sha256"]+".json")
            self.assertEqual(uploader.shared.read_json(intent)["status"], "unknown")
            with patch.object(uploader.shared, "now", return_value=datetime(2026, 11, 2, tzinfo=timezone.utc)):
                with self.assertRaisesRegex(RuntimeError, "unknown"):
                    uploader.submit_once(manifest, api, root, validator, timeout=0)
            api.recent.return_value = [{"ref": 999001, "team_name": "Saber", "description": "sha256:"+details["submission_sha256"]}]
            self.assertEqual(uploader.submit_once(manifest, api, root, validator, timeout=0)["submission_ref"], 999001)
            api.submit.assert_called_once()

    def test_conflicting_official_reference_stops_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _, api, validator = self.fixture(root)
            api.get.return_value["ref"] = 999002
            with self.assertRaisesRegex(RuntimeError, "reference"):
                uploader.submit_once(manifest, api, root, validator, timeout=0)
            api.submit.assert_called_once()
            self.assertFalse((root/"results/research/submission_receipts/999001.json").exists())

    def transport_fixture(self, root):
        manifest, details, _, _ = self.fixture(root)
        digest = details["submission_sha256"]
        intent_path = root/"results/research/submission_intents"/(digest+".json")
        staged = root/"results/research/upload_files"/digest/"submission.csv"
        staged.parent.mkdir(parents=True)
        staged.write_bytes(Path(details["csv_path"]).read_bytes())
        intent = {**details, "status": "upload_attempt_started", "submission_ref": None,
            "lock_token": "owner", "staged_csv": str(staged), "validation_python": sys.executable,
            "manifest_path": str(manifest), "validated": details, "team": "Saber", "description": "sha256:"+digest}
        write(intent_path, intent)
        write(root/"results/research/submission_upload.lock", {"token": "owner"})
        return intent_path, intent

    def test_real_transport_uses_new_validator_and_one_official_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            intent_path, intent = self.transport_fixture(root)
            sdk = ModuleType("kaggle.api.kaggle_api_extended")
            api = Mock()
            api.competition_get_submission_limits.return_value = SimpleNamespace(num_today=3, num_allowed_now=2)
            api.competition_submit.return_value = SimpleNamespace(ref=999001)
            sdk.KaggleApi = Mock(return_value=api)
            child = Mock(returncode=0, stdout=json.dumps(intent["validated"]))
            with patch.dict(sys.modules, {"kaggle.api.kaggle_api_extended": sdk}), \
                    patch.object(uploader.importlib.metadata, "version", return_value="2.2.5"), \
                    patch.object(uploader.subprocess, "run", return_value=child) as subprocess_call, \
                    patch.object(uploader.shared, "transport", side_effect=AssertionError("old transport must not be used")):
                self.assertEqual(uploader.transport_submit({"intent_path": str(intent_path), "intent_sha256": uploader.shared.sha256(intent_path)}, root), {"ref": 999001})
            command = subprocess_call.call_args.args[0]
            self.assertEqual(Path(command[1]).resolve(), Path(uploader.__file__).resolve())
            self.assertIn("--_validate", command)
            api.authenticate.assert_called_once()
            api.competition_submit.assert_called_once()

    def test_failed_final_validation_stops_before_authentication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            intent_path, _ = self.transport_fixture(root)
            with patch.object(uploader.subprocess, "run", return_value=Mock(returncode=1)) as child, \
                    patch.object(uploader.importlib.metadata, "version", side_effect=AssertionError("no authentication")):
                with self.assertRaisesRegex(ValueError, "validation failed"):
                    uploader.transport_submit({"intent_path": str(intent_path), "intent_sha256": uploader.shared.sha256(intent_path)}, root)
            self.assertEqual(Path(child.call_args.args[0][1]).resolve(), Path(uploader.__file__).resolve())

    def test_old_schema_unknown_intent_or_foreign_lock_never_reaches_validator(self):
        for field, value in (("delivery_schema", "research_refit_delivery_v1"), ("status", "unknown"),
                ("lock_token", "foreign"), ("checkpoint_sha256", "f"*64)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                intent_path, intent = self.transport_fixture(root)
                write(intent_path, {**intent, field: value})
                with patch.object(uploader.subprocess, "run", side_effect=AssertionError("no child or API")):
                    with self.assertRaises(ValueError):
                        uploader.transport_submit({"intent_path": str(intent_path), "intent_sha256": uploader.shared.sha256(intent_path)}, root)

    def test_changed_final_validator_or_staged_csv_never_reaches_auth(self):
        for kind in ("validator", "csv"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                intent_path, intent = self.transport_fixture(root)
                validated = copy.deepcopy(intent["validated"])
                if kind == "validator": validated["checkpoint_sha256"] = "f"*64
                else: Path(intent["staged_csv"]).write_text("changed after staging")
                with patch.object(uploader.subprocess, "run", return_value=Mock(returncode=0, stdout=json.dumps(validated))), \
                        patch.object(uploader.importlib.metadata, "version", side_effect=AssertionError("no authentication")):
                    with self.assertRaisesRegex(ValueError, "changed"):
                        uploader.transport_submit({"intent_path": str(intent_path), "intent_sha256": uploader.shared.sha256(intent_path)}, root)

    def test_transport_import_is_stdlib_only_and_keeps_frozen_uploader_unchanged(self):
        root = Path(delivery.__file__).resolve().parents[1]
        code = "import sys; import research.submit_wrn300_refit; import research.wrn300_refit_delivery; assert 'torch' not in sys.modules; print('stdlib-only')"
        self.assertEqual(subprocess.check_output([sys.executable, "-c", code], cwd=root, text=True).strip(), "stdlib-only")
        self.assertEqual(uploader.shared.sha256(root/"research/submit_candidate.py"), "747db0278f70d0149ebdc7dd19d597881f8d3ab97692a5468cb6b5eba592fdc0")


if __name__ == "__main__":
    unittest.main()
