"""Explicit JSON-only report extension for completed WRN16/200 and KD control timing.

The original projection, all scored receipts and their source stay unchanged.
This is a report adapter, never a training, promotion or delivery validator.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
PARENT_SOURCE="d1f069f6bbecb5d3129729a07d1de11caf6a9094f291e80553481dd7a39ad2e1"
def require(condition,message):
    if not condition:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
require(sha(ROOT/"scripts/campaign_report_evidence.py")==PARENT_SOURCE,"Original report adapter changed")
from scripts import campaign_report_evidence as parent

INPUT_SCHEMA="campaign_report_input_manifest_v2"
SCHEMA="audited_campaign_report_projection_v2"
ROLES={"campaign_manifest","wrn16200_summary","wrn16200_benchmark","wrn16200_supervisor",
       "kdcontrol_summary","kdcontrol_report","kdcontrol_supervisor","kdcontrol_inputs"}
WRN_BENCHMARK="07ac967c6875ec2189d158ecb9059ef228e3dae7bf22d583923b508cb87aa1bc"
WRN_CHECKPOINT="3fb57b6d458bfd4310468d8aca8a02e3aa29b5a542e9bfe76ba80483bff59ce7"
WRN_SUMMARY="cb6302e031c0901cc6b00b90e4905674d65ef90016445e367aaed39a2829f327"
KD_REPORT="358da6f9a93d789cf339ed5c754befd069f3231f6532fbfe0b762f88c3b4be2a"
KD_SUPERVISOR="a935dc9ac779f1a20a0d1be21c4abd512c71ad17570237cff9b3b5c6c31b91e1"
KD_CONTROLLER="61ef86a809338d87cdbd4784b2df1917aa232217708151462ce3b2ea148f3ab7"


def _wrn200(files,hashes,paths,baseline,pixels,incumbent):
    public,doc,receipt=(files[k] for k in ("wrn16200_summary","wrn16200_benchmark","wrn16200_supervisor"))
    require(hashes["wrn16200_summary"]==WRN_SUMMARY and hashes["wrn16200_benchmark"]==WRN_BENCHMARK,
            "Only the actual completed200 report snapshot is supported")
    folder=(ROOT/"results/research/wrn16_5_200_s2026").resolve()
    require(paths["wrn16200_benchmark"]==folder/"benchmark.json" and paths["wrn16200_supervisor"]==folder/"supervisor.json",
            "Use the original completed200 raw records")
    require(public.get("schema")=="research_wrn16_200_public_results_v1" and public.get("status")=="complete_rejected"
            and public.get("run")==receipt.get("run")==folder.name and receipt.get("status")=="complete"
            and receipt.get("backend")=="local" and receipt.get("trainer")=="frozen" and receipt.get("resumed") is False
            and receipt.get("precision")=="bf16" and receipt.get("confirmation_evaluated") is False
            and receipt.get("submitted") is False and receipt.get("benchmark_sha256")==WRN_BENCHMARK,
            "Actual completed scratch200 original PID/watchdog scope required")
    report=parent._full_benchmark(doc,baseline,pixels)
    require(doc.get("parameter_count")==4289754 and doc.get("config",{}).get("epochs")==200
            and doc.get("config")==public.get("config") and doc.get("checkpoint_sha256")==WRN_CHECKPOINT
            and (doc.get("checkpoint_epoch"),doc.get("checkpoint_variant"))==(199,"model"),"Exact200 selected identity differs")
    row=parent._row("wrn16_200","WRN16-5 / 200",doc,report,baseline,incumbent,"bf16",200)
    require(row["original_gate_passed"] is True and row["incumbent_mean_passed"] is False and row["case_floor_passed"] is False,
            "Retain the actual200 prospective negative")
    negative={case for case,delta in row["case_accuracy_deltas_vs_incumbent"].items() if case!="clean" and delta<-.01-1e-12}
    require(negative=={"noise_004","resize24"} and public["original_baseline_screen"]["passed"] is True
            and public["incumbent_screen"]["all_required_point_and_case_gates_passed"] is False,
            "Original gate passing cannot hide the noise/resize failure")
    parent._source_map(public["source_sha256"])
    raw=public["raw_evidence_sha256"]
    own={"supervisor":"supervisor.json","metrics":"metrics.json","history":"history.json","provenance":"provenance.json",
         "environment":"environment.json","config":"config.json","benchmark":"benchmark.json",
         "incumbent_comparison":"incumbent_comparison.json","best":"best.pt","last":"last.pt"}
    require(all(sha(folder/name)==raw[key] for key,name in own.items()) and raw["supervisor"]==hashes["wrn16200_supervisor"]
            and raw["benchmark"]==WRN_BENCHMARK and raw["best"]==WRN_CHECKPOINT,"Actual200 selected/last/log JSON bytes changed")
    history=parent.legacy._json(folder/"history.json");metrics=parent.legacy._json(folder/"metrics.json")
    require([r.get("epoch") for r in history]==list(range(1,201)) and metrics.get("status")=="complete"
            and metrics.get("epochs_completed")==200 and metrics.get("config")==doc["config"]
            and doc["training_metrics_sha256"]==raw["metrics"] and doc["training_history_sha256"]==raw["history"],
            "Complete200 history/metrics binding differs")
    for key in ("paired","frozen","fresh160_control_executed","repeat_executed","hidden_prediction","submission","full50k_refit","schedule_causality_established"):
        require(public["decision"].get(key) is False,"The rejected200 candidate has no further qualification")
    scope=public["audit_scope"]
    require(scope.get("last_checkpoint_update_state_independently_loaded") is False
            and scope.get("selected_checkpoint_tensor_reloaded_by_independent_audit") is False,
            "Do not upgrade a JSON audit into a last/optimizer/RNG tensor audit")
    require(public["training"].get("successful_updates")==62600 and public["training"].get("source_examples_seen")==8000000,
            "Complete-log workload arithmetic differs")
    row.update(qualification_status="rejected_incumbent_mean_noise_resize",submission_status="not_submitted",
        benchmark_sha256=WRN_BENCHMARK,source_summary_sha256=WRN_SUMMARY,
        workload={"successful_updates":62600,"skipped_updates":0,"source_examples_seen":8000000},
        workload_scope="Derived from complete200 history/logs and frozen direct-SGD loop; last optimizer/RNG was not independently reloaded.",
        audit_scope="JSON/hash/context/pixel/gate audit; original benchmark selected-tensor/architecture/history audit only; no independent last optimizer/RNG reload.",
        fresh160_control_executed=False,schedule_causality_established=False,
        supervisor_seconds=receipt["elapsed_seconds"])
    return row


def _kd_control(files,hashes,paths):
    require(hashes["kdcontrol_report"]==KD_REPORT and hashes["kdcontrol_supervisor"]==KD_SUPERVISOR
            and sha(ROOT/"research/run_scratch_kd.py")==KD_CONTROLLER,"Only the actual fixed control engineering run is supported")
    from research import run_scratch_kd as controller
    folder=(ROOT/"results/research/scratch_kd_control_native_preflight_20261103_v1").resolve()
    require(paths["kdcontrol_report"]==folder/"scratch_kd_preflight.json" and paths["kdcontrol_supervisor"]==folder/"supervisor.json"
            and paths["kdcontrol_inputs"]==folder/"inputs.json","Keep original engineering evidence locations")
    report,manifest,receipt,_=controller.completed_envelope(folder,controller.source_snapshot(),"preflight","control26")
    controller.validate_preflight_report(report,manifest,hashes["kdcontrol_inputs"])
    require(report==files["kdcontrol_report"] and receipt==files["kdcontrol_supervisor"] and manifest==files["kdcontrol_inputs"],
            "Explicit engineering roles differ from the actual closed bundle")
    guard=controller.guard_seconds(report["median_step_seconds"])
    require(guard==report["conservative_pipeline_seconds"] and guard>10800 and receipt.get("timing_guard_within_job_cap") is False
            and report["teacher_forward_passes"]==0 and report["accuracy_evaluated"] is False and receipt["formal_training_admitted"] is False,
            "Mandatory-control timing rejection is not a KD forward/accuracy experiment")
    public=files["kdcontrol_summary"]
    require(public.get("schema")=="research_scratch_kd_control_preflight_public_results_v1","Explicit control-pref public schema required")
    # Public aggregate cannot replace the actual report/closed native receipt.
    raw=public.get("raw_evidence_sha256",{})
    require(raw.get("preflight_report")==KD_REPORT and raw.get("supervisor")==KD_SUPERVISOR
            and raw.get("inputs")==hashes["kdcontrol_inputs"],"Public/raw control engineering bindings differ")
    return {"id":"scratch_kd_control_preflight","label":"Scratch KD mandatory alpha0 control engineering",
        "role":"control26","alpha":0.0,"temperature":4.0,"status":"timing_rejected","formal_training_admitted":False,
        "median_step_seconds":report["median_step_seconds"],"conservative_pipeline_seconds":guard,"job_budget_seconds":10800,
        "supervisor_seconds":receipt["elapsed_seconds"],"successful_updates":23,"skipped_updates":0,"unique_source_examples":2944,
        "student_parameters":4289754,"teacher_parameters":4166298,"teacher_forward_passes":0,"accuracy_evaluated":False,
        "KD_positive_alpha_preflight_run":False,"formal_control_run":False,"KD_training_run":False,"public_score":None,
        "preflight_report_sha256":KD_REPORT,"supervisor_sha256":KD_SUPERVISOR,
        "interpretation":"The mandatory alpha0 control is timing-ineligible; no positive-alpha KD speed or accuracy was measured. Full pipeline/proof costs and the original1.5 multiplier are retained.",
        "audit_scope":"Report rechecks JSON/arithmetic/file hashes and archived authenticated native cleanup; native worker performed actual tensors/updates, not this adapter."}


def load_campaign(manifest_path):
    path=Path(manifest_path).resolve();digest=sha(path);manifest=parent.legacy._json(path)
    require(set(manifest)=={"schema","recorded_at","files"} and manifest["schema"]==INPUT_SCHEMA
            and isinstance(manifest["files"],dict) and set(manifest["files"])==ROLES,"Explicit completed200/control-engineering report roles required")
    recorded=parent.legacy._stamp(manifest["recorded_at"]);files={};hashes={};paths={};sources=[]
    for role,spec in manifest["files"].items():
        document,public,actual=parent._read_record(spec,path.parent);files[role]=document;hashes[role]=spec["sha256"];paths[role]=actual
        sources.append({"role":role,**public})
        for key in ("recorded_at","finished_at"):
            if key in document:require(parent.legacy._stamp(document[key])<=recorded,"Input evidence postdates the report")
    result=copy.deepcopy(parent.load_campaign(paths["campaign_manifest"]))
    require(parent.legacy._stamp(result["recorded_at"])<=recorded and result["schema"]==parent.SCHEMA,"Actual parent projection required")
    original_receipts=copy.deepcopy(result["official_submissions"])
    base_manifest=files["campaign_manifest"];base_files={}
    for role in ("legacy_summary","baseline_benchmark","incumbent_benchmark"):
        base_files[role]=parent._read_record(base_manifest["files"][role],paths["campaign_manifest"].parent)[0]
    baseline=base_files["legacy_summary"]["baseline"]
    pixels={case:base_files["baseline_benchmark"]["policies"]["flip"]["cases"][case]["input_pixel_sha256"] for case in parent.CASES}
    incumbent=parent._full_benchmark(base_files["incumbent_benchmark"],baseline,pixels)
    row=_wrn200(files,hashes,paths,baseline,pixels,incumbent)
    require(not any(r["id"]==row["id"] for r in result["rows"]),"Duplicate200 report row")
    result["rows"].insert(2,row);result["engineering_preflights"]=[_kd_control(files,hashes,paths)]
    result.update(schema=SCHEMA,recorded_at=manifest["recorded_at"],source_manifest_sha256=digest,
        parent_source_manifest_sha256=hashes["campaign_manifest"],parent_adapter_source_sha256=PARENT_SOURCE,adapter_source_sha256=sha(__file__))
    result["sources"]+=sources
    result["engineering"]+=["WRN16/200 passed the old baseline gate but failed the prospectively frozen incumbent mean/noise/resize gates; no fresh160 control or schedule-causality result exists.",
        "Scratch KD's mandatory alpha0 control completed23 engineering updates but exceeded the unchanged3h guard. No positive-alpha preflight, formal control/KD training or KD accuracy exists."]
    result["limitations"]+=["The200 workload is complete-log/loop arithmetic; its original selected-tensor benchmark is not an independent last optimizer/RNG tensor reload."]
    require(result["official_submissions"]==original_receipts and row["public_score"] is None and result["best_observed_public"]==.878,
            "Development and engineering records cannot invent an official score")
    require(sha(path)==digest and all(sha(paths[role])==hashes[role] for role in ROLES),"Explicit report inputs changed")
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--manifest",type=Path);args=parser.parse_args(argv)
    if args.manifest is None:parser.print_help();return
    print(json.dumps(load_campaign(args.manifest),sort_keys=True,indent=2,allow_nan=False))


if __name__=="__main__":main()
