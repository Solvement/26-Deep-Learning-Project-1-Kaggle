"""Pure report boundaries with explicit synthetic fixtures; no training claims."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from scripts import campaign_report_evidence_v2 as report
from tests import test_campaign_report_evidence as fixtures


class ReportExtensionContracts(unittest.TestCase):
    def test_import_and_help_do_not_import_numerics_or_renderers(self):
        script="import sys; from scripts import campaign_report_evidence_v2 as r; r.main([]); assert not {'torch','numpy','PIL','runtime','nbformat','nbclient','reportlab'}&set(sys.modules)"
        result=subprocess.run([sys.executable,"-c",script],cwd=report.ROOT,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn("--manifest",result.stdout)

    def wrn_fixture(self,root):
        doc,baseline,pixels=fixtures.CampaignReportContracts().doc();doc=copy.deepcopy(doc)
        doc["config"].update(depth=16,widen_factor=5,epochs=200)
        doc.update(parameter_count=4289754,checkpoint_epoch=199,checkpoint_variant="model",checkpoint_sha256=report.WRN_CHECKPOINT)
        incumbent=copy.deepcopy(doc["policies"]["center80_shift20"])
        values={"clean":.9644,"noise_004":.7074,"boxblur3":.4844,"contrast065":.9556,"brightness080":.962,"resize24":.8232}
        other={"clean":.9654,"noise_004":.7768,"boxblur3":.4394,"contrast065":.9594,"brightness080":.9648,"resize24":.8454}
        for row,actual in [(doc["policies"]["center80_shift20"],values),(incumbent,other)]:
            for case,value in actual.items():row["cases"][case]["accuracy"]=value
            row.update(clean_accuracy=actual["clean"],clean_loss=row["cases"]["clean"]["loss"],stress_accuracy=sum(v for k,v in actual.items() if k!="clean")/5)
        baseline["flip"].update(clean_accuracy=.9628,stress_accuracy=.77628)
        folder=root/"results/research/wrn16_5_200_s2026";folder.mkdir(parents=True)
        own={"supervisor":"supervisor.json","metrics":"metrics.json","history":"history.json","provenance":"provenance.json",
             "environment":"environment.json","config":"config.json","benchmark":"benchmark.json","incumbent_comparison":"incumbent_comparison.json","best":"best.pt","last":"last.pt"}
        for name in own.values():(folder/name).write_text("{}")
        (folder/"history.json").write_text(json.dumps([{"epoch":i} for i in range(1,201)]))
        (folder/"metrics.json").write_text(json.dumps({"status":"complete","epochs_completed":200,"config":doc["config"]}))
        raw={key:report.sha(folder/name) for key,name in own.items()}
        doc.update(training_metrics_sha256=raw["metrics"],training_history_sha256=raw["history"])
        public={"schema":"research_wrn16_200_public_results_v1","status":"complete_rejected","run":folder.name,"config":doc["config"],
            "source_sha256":{},"raw_evidence_sha256":raw,"original_baseline_screen":{"passed":True},
            "incumbent_screen":{"all_required_point_and_case_gates_passed":False},
            "decision":{k:False for k in ("paired","frozen","fresh160_control_executed","repeat_executed","hidden_prediction","submission","full50k_refit","schedule_causality_established")},
            "audit_scope":{"last_checkpoint_update_state_independently_loaded":False,"selected_checkpoint_tensor_reloaded_by_independent_audit":False},
            "training":{"successful_updates":62600,"source_examples_seen":8000000}}
        receipt={"run":folder.name,"status":"complete","backend":"local","trainer":"frozen","resumed":False,"precision":"bf16",
            "confirmation_evaluated":False,"submitted":False,"benchmark_sha256":raw["benchmark"],"elapsed_seconds":2389.9}
        files={"wrn16200_summary":public,"wrn16200_benchmark":doc,"wrn16200_supervisor":receipt}
        hashes={"wrn16200_summary":"1"*64,"wrn16200_benchmark":raw["benchmark"],"wrn16200_supervisor":raw["supervisor"]}
        paths={"wrn16200_benchmark":folder/"benchmark.json","wrn16200_supervisor":folder/"supervisor.json"}
        return files,hashes,paths,baseline,pixels,incumbent,raw

    def test_200_original_gate_cannot_hide_floor_failure_or_invent_score_tensor_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            files,hashes,paths,baseline,pixels,incumbent,raw=self.wrn_fixture(Path(tmp))
            with patch.object(report,"ROOT",Path(tmp)),patch.object(report,"WRN_BENCHMARK",raw["benchmark"]),patch.object(report,"WRN_CHECKPOINT",raw["best"]),patch.object(report,"WRN_SUMMARY",hashes["wrn16200_summary"]),patch.object(report.parent,"_source_map"),patch.object(report.parent,"_full_benchmark",return_value=files["wrn16200_benchmark"]["policies"]["center80_shift20"]):
                files["wrn16200_benchmark"]["checkpoint_sha256"]=raw["best"]
                row=report._wrn200(files,hashes,paths,baseline,pixels,incumbent)
                self.assertTrue(row["original_gate_passed"]);self.assertFalse(row["gate_passed"]);self.assertIsNone(row["public_score"])
                self.assertEqual(row["qualification_status"],"rejected_incumbent_mean_noise_resize")
                self.assertIn("not independently reloaded",row["workload_scope"])
                for field in ("submission","fresh160_control_executed","schedule_causality_established"):
                    bad=copy.deepcopy(files);bad["wrn16200_summary"]["decision"][field]=True
                    with self.subTest(field=field),self.assertRaises(ValueError):report._wrn200(bad,hashes,paths,baseline,pixels,incumbent)
                bad=copy.deepcopy(files);bad["wrn16200_summary"]["audit_scope"]["last_checkpoint_update_state_independently_loaded"]=True
                with self.assertRaises(ValueError):report._wrn200(bad,hashes,paths,baseline,pixels,incumbent)
                (Path(tmp)/"results/research/wrn16_5_200_s2026/last.pt").write_text("changed")
                with self.assertRaises(ValueError):report._wrn200(files,hashes,paths,baseline,pixels,incumbent)

    def kd_fixture(self):
        folder=report.ROOT/"results/research/scratch_kd_control_native_preflight_20261103_v1"
        manifest={"mode":"preflight","role":"control26"}
        doc={"median_step_seconds":.22,"conservative_pipeline_seconds":1.5*.22*313*160+300,"teacher_forward_passes":0,"accuracy_evaluated":False}
        receipt={"timing_guard_within_job_cap":False,"formal_training_admitted":False,"elapsed_seconds":21.4}
        public={"schema":"research_scratch_kd_control_preflight_public_results_v1","raw_evidence_sha256":
            {"preflight_report":report.KD_REPORT,"supervisor":report.KD_SUPERVISOR,"inputs":"a"*64}}
        files={"kdcontrol_summary":public,"kdcontrol_report":doc,"kdcontrol_supervisor":receipt,"kdcontrol_inputs":manifest}
        hashes={"kdcontrol_report":report.KD_REPORT,"kdcontrol_supervisor":report.KD_SUPERVISOR,"kdcontrol_inputs":"a"*64}
        paths={"kdcontrol_report":folder/"scratch_kd_preflight.json","kdcontrol_supervisor":folder/"supervisor.json","kdcontrol_inputs":folder/"inputs.json"}
        return files,hashes,paths,doc,manifest,receipt

    def test_control_engineering_is_not_KD_speed_accuracy_or_admission(self):
        from research import run_scratch_kd as controller
        files,hashes,paths,doc,manifest,receipt=self.kd_fixture()
        with patch.object(controller,"completed_envelope",return_value=(doc,manifest,receipt,[])),patch.object(controller,"validate_preflight_report"):
            row=report._kd_control(files,hashes,paths)
            self.assertFalse(row["formal_training_admitted"]);self.assertFalse(row["KD_positive_alpha_preflight_run"])
            self.assertFalse(row["accuracy_evaluated"]);self.assertEqual(row["teacher_forward_passes"],0);self.assertIsNone(row["public_score"])
            for key,value in [("teacher_forward_passes",23),("accuracy_evaluated",True),("conservative_pipeline_seconds",10800)]:
                before=doc[key];doc[key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):report._kd_control(files,hashes,paths)
                doc[key]=before
            receipt["formal_training_admitted"]=True
            with self.assertRaises(ValueError):report._kd_control(files,hashes,paths)

    def test_incomplete_or_unknown_role_groups_fail_before_reading_private_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"manifest.json"
            for roles in [{},{"campaign_manifest":{}},{**{k:{} for k in report.ROLES},"unknown":{}}]:
                path.write_text(json.dumps({"schema":report.INPUT_SCHEMA,"recorded_at":"2026-11-03T04:30:00+00:00","files":roles}))
                with self.assertRaises(ValueError):report.load_campaign(path)


if __name__=="__main__":unittest.main()
