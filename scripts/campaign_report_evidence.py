"""Independent campaign JSON projection; no old report source/global is changed.

Inputs are explicit fingerprinted JSON roles. No tensors, images, predictions,
notebook kernels, PDF renderers, Torch, Kaggle client or qualification mutation.
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path,PurePosixPath
import argparse
import hashlib
import json
import math
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
LEGACY_SOURCE_SHA="888c8c509d5dc3034847bcdb9b8b8b213babe1e04e336345a94bdd920306d0ef"
if hashlib.sha256((ROOT/"scripts/research_report_evidence.py").read_bytes()).hexdigest()!=LEGACY_SOURCE_SHA:
    raise ValueError("Frozen original report adapter bytes changed before import")
from scripts import research_report_evidence as legacy

INPUT_SCHEMA="campaign_report_input_manifest_v1"
SCHEMA="audited_campaign_report_projection_v1"
POLICY=legacy.POLICY
CASES=("clean","noise_004","boxblur3","contrast065","brightness080","resize24")
TA_CHECKPOINT="dcfe213533481aa6c90d2fdd9ac853dc291c124098c9fe0ab8866337ab3f0fba"
TA_BENCHMARK="47cfe0ebedbd3f01be162648e8362c71df7607dd5389929d67da14342e79b205"
INCUMBENT_CHECKPOINT="53d8d64df899ebab79ae77144ff36e912c45e65178c0b50bd9af381dc48187bc"
INCUMBENT_BENCHMARK="e3c586afc302d1422c80eb3831562f6b67f684d53c7b2b5390c9d4623001570b"
BASELINE_CHECKPOINT="502979bc03301aa2f0273a7414f86799302b908a3abada622a88e4578c4b734f"
BASELINE_BENCHMARK="eced314d97fcb5ddef4cad25c6968ba299d0e8a2c79af6bdcbdbf305b79c632d"
REQUIRED={"legacy_summary","legacy_results","baseline_benchmark","incumbent_benchmark",
          "ta_summary","ta_benchmark","ta_pair_summary","ta_pair","ta_frozen",
          "se_summary","se_benchmark","resizemix300_summary","resizemix300_benchmark",
          "wrn16_summary","wrn16_benchmark","augmix_summary","augmix_benchmark"}
REPEAT={"ta_repeat_benchmark","ta_repeat_result","ta_repeat_receipt"}
DELIVERY={"ta_delivery","ta_prediction_receipt","ta_prediction_inputs","ta_tensor_audit"}
CONFIG_KEYS={"depth","widen_factor","dropout","num_classes","epochs","batch_size","lr","min_lr","warmup_epochs",
             "momentum","weight_decay","label_smoothing","cutout","mixup_alpha","mixup_probability","ema_decay",
             "seed","split_seed","train_all","final_variant","fixed_flip_tta","ta_space","ta_strengths",
             "resizemix_probability","resizemix_tau_min","resizemix_tau_max"}
require=legacy.require


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def _read_record(record,base):
    require(isinstance(record,dict) and set(record)<={"path","sha256","public_ref"} and set(record)>={"path","sha256"},"Explicit path/hash JSON record required")
    legacy._hash(record["sha256"],"input role")
    require(isinstance(record["path"],str) and record["path"],"Input role path is missing")
    path=Path(record["path"]);path=path if path.is_absolute() else base/path
    require(not path.is_symlink(),"Input-role symlinks are unsupported");path=path.resolve()
    require(path.is_file() and path.suffix.lower()==".json" and _sha(path)==record["sha256"],"Actual JSON role fingerprint differs")
    public={"sha256":record["sha256"]}
    if "public_ref" in record:
        ref=record["public_ref"]
        require(isinstance(ref,str) and re.fullmatch(r"[A-Za-z0-9_./-]+",ref) and not PurePosixPath(ref).is_absolute()
           and ".." not in PurePosixPath(ref).parts and ref.startswith("docs/") and (ROOT/ref).resolve()==path,"Only matching public-relative docs references may be projected")
        public["public_ref"]=ref
    return legacy._json(path),public,path


def _source_map(mapping):
    require(isinstance(mapping,dict) and mapping,"Recorded source closure required")
    for name,digest in mapping.items():
        require(isinstance(name,str) and re.fullmatch(r"[A-Za-z0-9_./-]+",name) and not PurePosixPath(name).is_absolute()
           and ".." not in PurePosixPath(name).parts and (name in {"train.py","model.py","data.py","runtime.py"} or name.startswith(("research/","scripts/")))
           and PurePosixPath(name).suffix in {".py",".json",".md",".txt"},"Only source-relative names are accepted")
        legacy._hash(digest,"source");require((ROOT/name).is_file() and _sha(ROOT/name)==digest,"Actual recorded source bytes differ")


def _training_identity(document,baseline,seed):
    """Shared metadata/source checks, with explicit ordinary-WRN and SE types."""
    require(document.get("bn_recalibrated") is False and document.get("stress_cases")==list(CASES)
       and type(document.get("batch_size")) is int and document["batch_size"]==128
       and document.get("confirmation_evaluated") is False and document.get("hidden_images_loaded") is False
       and document.get("external_data_loaded") is False,"Complete native-v3 original-BN official scope required")
    config=document.get("config",{});require(config.get("seed")==seed and type(config.get("seed")) is int
       and config.get("train_all") is False and config.get("split_seed")==2026,"Exact independent40k primary/repeat scope required")
    legacy._integer(config.get("epochs"),"complete epoch count")
    legacy._integer(document.get("checkpoint_epoch"),"selected epoch",high=config["epochs"])
    constructor={k:config.get(k) for k in ("depth","widen_factor","dropout","num_classes")}
    marker=document.get("research_training_integrity",{}).get("training",{})
    if "architecture" in config:
        require(config["architecture"]=="se_wrn22_4_r16_v1" and type(config.get("se_reduction")) is int and config["se_reduction"]==16
           and constructor=={"depth":22,"widen_factor":4,"dropout":0.,"num_classes":10}
           and marker.get("schema")=="research_se_sam_training_v1" and marker.get("architecture")==config["architecture"]
           and type(marker.get("reduction")) is int and marker["reduction"]==config["se_reduction"],"Explicit recorded SE architecture/reduction differs")
        constructor={"architecture":config["architecture"],**constructor,"reduction":config["se_reduction"]}
    require(document.get("checkpoint_variant") in {"model","ema"} and document.get("model_config")==constructor,"Recorded selected constructor/variant differs")
    for row in document["policies"].values():
        for case in row["cases"].values():
            count=case["accuracy"]*5000;require(abs(count-round(count))<=1e-8,"Every complete5000 accuracy must encode an integer correct count")
    core=document.get("source",{}).get("source_sha256",{})
    require(core==baseline["evaluation_context"]["training_core_sha256"],"Core source context differs")
    _source_map(core)
    for key in ("training_metrics_sha256","training_history_sha256","benchmark_source_sha256"):legacy._hash(document.get(key),key)
    require(document["benchmark_source_sha256"]==_sha(ROOT/"research/benchmark.py"),"Frozen benchmark engine differs")
    if marker:_source_map(marker.get("source_sha256"))


def _full_benchmark(document,baseline,pixels,seed=2026):
    """Candidate benchmarks must explicitly reference the frozen origin."""
    require(document.get("baseline_sha256")==baseline["benchmark_sha256"],"Candidate benchmark must bind the actual origin baseline")
    report=legacy._benchmark(document,baseline,pixels)
    _training_identity(document,baseline,seed)
    return report


def _origin_baseline(document,baseline,pixels,actual_sha):
    """Audit the origin directly; its real null baseline reference stays null."""
    require(actual_sha==baseline.get("benchmark_sha256")==BASELINE_BENCHMARK
       and document.get("checkpoint_sha256")==baseline.get("checkpoint_sha256")==BASELINE_CHECKPOINT
       and document.get("parameter_count")==4298970 and type(document.get("parameter_count")) is int
       and document.get("baseline_sha256") is None,"Actual fixed origin baseline identity/reference differs")
    require(document.get("schema")==legacy.BENCHMARK_SCHEMA and document.get("status")=="complete"
       and type(document.get("examples")) is int and document["examples"]==5000
       and type(document.get("training_examples")) is int and document["training_examples"]==40000
       and document.get("data_signature")==baseline["data_signature"] and document.get("evaluation_context")==baseline["evaluation_context"]
       and set(document.get("policies",{}))==legacy.POLICIES,"Complete independent native-v3 origin context required")
    legacy._hash(document.get("checkpoint_sha256"),"origin checkpoint")
    for policy,row in document["policies"].items():
        require(isinstance(row.get("cases"),dict) and set(row["cases"])==set(CASES),"Every origin case/policy is required")
        for case,value in row["cases"].items():
            require(type(value.get("examples")) is int and value["examples"]==5000,"Origin case is partial")
            legacy._number(value.get("accuracy"),"origin accuracy",high=1);legacy._number(value.get("loss"),"origin NLL")
            require(legacy._hash(value.get("input_pixel_sha256"),"origin pixels")==pixels[case],"Origin policy pixel fingerprints differ")
        require(legacy._close(row.get("clean_accuracy"),row["cases"]["clean"]["accuracy"])
           and legacy._close(row.get("clean_loss"),row["cases"]["clean"]["loss"])
           and legacy._close(row.get("stress_accuracy"),sum(row["cases"][c]["accuracy"] for c in CASES[1:])/5),"Origin complete-case aggregate differs")
        legacy._check_gate(document.get("promotion",{}).get(policy),legacy._gate(row,document["policies"]["flip"]))
    _training_identity(document,baseline,2026)
    flip=document["policies"]["flip"]
    require(all(legacy._close(flip[k],baseline["flip"][k]) for k in ("clean_accuracy","clean_loss","stress_accuracy")),"Archived origin flip summary differs from full raw baseline")
    return document["policies"][POLICY]


def _incremental(report,incumbent):
    gate=legacy._gate(report,incumbent)
    deltas={c:report["cases"][c]["accuracy"]-incumbent["cases"][c]["accuracy"] for c in CASES}
    floor=all(deltas[c]>=-.01-1e-12 for c in CASES[1:])
    return gate,floor,deltas


def _safe_config(config):
    result={}
    for key,value in config.items():
        if key not in CONFIG_KEYS:continue
        if key in {"train_all","fixed_flip_tta"}:require(type(value) is bool,"Configuration boolean differs")
        elif key=="final_variant":require(type(value) is str and value in {"model","ema","swa"},"Unknown final variant")
        elif key=="ta_space":require(type(value) is str and value=="fixed_standard","Unknown TA space")
        else:require(type(value) in (int,float) and math.isfinite(value) and value>=0,"Only finite numeric configuration scalars may be published")
        result[key]=value
    return result


def _workload(declared,expected,integrity):
    require(all(type(declared.get(k)) is int and declared[k]==v for k,v in expected.items()),"Published complete update/view counts differ")
    for key,value in expected.items():
        if key in integrity:require(type(integrity[key]) is int and integrity[key]==value,"Recorded typed workload differs from public aggregate")
    return dict(expected)


def _row(identity,label,document,report,baseline,incumbent,precision,epochs):
    old=legacy._gate(report,baseline["flip"]);inc,floor,deltas=_incremental(report,incumbent)
    config=document.get("config",{})
    return {"id":identity,"label":label,"parameters":document["parameter_count"],"epochs":epochs,"precision":precision,
       "evaluation_precision":"native_bf16","policy":POLICY,"training_examples":40000,"development_examples":5000,
       "checkpoint_epoch":document["checkpoint_epoch"],"checkpoint_variant":document["checkpoint_variant"],
       "checkpoint_sha256":document["checkpoint_sha256"],"clean_accuracy":report["clean_accuracy"],"clean_nll":report["clean_loss"],
       "stress_accuracy":report["stress_accuracy"],"clean_delta":old["clean_delta"],"stress_delta":old["stress_delta"],
       "original_gate_passed":old["passed"],"incumbent_mean_passed":inc["passed"],"case_floor_passed":floor,
       "incumbent_clean_delta":inc["clean_delta"],"incumbent_stress_delta":inc["stress_delta"],
       "case_accuracy_deltas_vs_incumbent":deltas,"noise_delta_vs_incumbent":deltas["noise_004"],
       "clean_ci95":None,"stress_ci95":None,"ci_reference":{"checkpoint_sha256":baseline["checkpoint_sha256"],"policy":"flip"},
       "incumbent_clean_ci95":None,"incumbent_stress_ci95":None,
       "incumbent_ci_reference":{"checkpoint_sha256":INCUMBENT_CHECKPOINT,"policy":POLICY},
       "gate_passed":old["passed"] and inc["passed"] and floor,"qualification_status":"not_promoted",
       "gate_scope":"primary_original_and_incumbent_mean_and_each_stress_floor",
       "paired_status":"not_performed","repeat_status":"not_promoted","repeat_clean_accuracy":None,"repeat_stress_accuracy":None,
       "repeat_complete":False,"repeat_passed":None,
       "repeat_epochs_completed":None,"repeat_planned_epochs":epochs,"delivery_status":"not_promoted","public_score":None,
       "cases":{c:{"examples":5000,"accuracy":report["cases"][c]["accuracy"],"nll":report["cases"][c]["loss"]} for c in CASES},
       "config":_safe_config(config),
       "audit_scope":"JSON/source/hash/context/pixel/scalar projection; actual tensor audits belong to archived typed engines."}


def _legacy_rows(summary,baseline,pixels,incumbent):
    rows=[]
    for identity,label,key,pair_key,repeat_key,epochs,precision in legacy.ROWS:
        if key not in summary:continue
        document=summary[key];report=legacy._benchmark(document,baseline,pixels)
        row=_row(identity,label,document,report,baseline,incumbent,precision,epochs)
        row["gate_passed"]=row["original_gate_passed"]
        row["qualification_status"]="historical_frozen_protocol"
        row["gate_scope"]="historical_frozen_protocol_not_replaced_by_incumbent_screen"
        row["incumbent_comparison_scope"]="retrospective_descriptive_not_a_changed_historical_gate"
        pair_passed=False
        if identity=="swa40":
            parent=summary["wrn34_3_primary"]
            require(document.get("parent_checkpoint_sha256")==parent["checkpoint_sha256"] and document.get("parent_benchmark_sha256")==parent["benchmark_sha256"],"Fixed SWA parent differs")
            for policy in legacy.POLICIES:legacy._check_gate(document["promotion_vs_parent"][policy],legacy._gate(document["policies"][policy],parent["policies"][policy]))
            row["gate_passed"]=row["gate_passed"] and document["promotion_vs_parent"][POLICY]["passed"]
        if pair_key and pair_key in summary:
            pair=summary[pair_key];ci=legacy._pair(pair,document,baseline,document.get("benchmark_sha256",pair.get("candidate_benchmark_sha256")))
            row.update(clean_ci95=ci["clean"],stress_ci95=ci["stress"],paired_status="complete")
            clean_path=row["clean_delta"]>=.001-1e-12 and row["stress_delta"]>=-.002-1e-12
            pair_passed=clean_path or ci["stress"][0]>0
        if repeat_key and repeat_key in summary:
            repeated=summary[repeat_key];r=legacy._benchmark(repeated,baseline,pixels)
            require(repeated["checkpoint_sha256"]!=document["checkpoint_sha256"] and repeated.get("config",{}).get("seed")==2027
               and repeated["config"].get("epochs")==epochs and document.get("config") is not None
               and {k:v for k,v in repeated["config"].items() if k!="seed"}=={k:v for k,v in document["config"].items() if k!="seed"},"Historical repeat recipe differs")
            passed=legacy._gate(r,baseline["flip"])["passed"]
            row.update(repeat_status="complete_passed" if passed else "complete_failed",repeat_clean_accuracy=r["clean_accuracy"],
               repeat_stress_accuracy=r["stress_accuracy"],repeat_epochs_completed=epochs,repeat_complete=True,repeat_passed=passed)
            row["delivery_status"]="qualified_unscored" if row["gate_passed"] and pair_passed and passed else "not_promoted"
        rows.append(row)
    return rows


def _public_values(summary,document,identity):
    report=document["policies"][POLICY]
    if identity in {"trivialaugment","resizemix300"}:
        selection=summary["selected_checkpoint"];bench=summary["benchmark"]
        require(selection["recorded_sha256"]==document["checkpoint_sha256"] and selection["epoch"]==document["checkpoint_epoch"]
           and selection["weight_variant"]==document["checkpoint_variant"] and summary["parameter_count"]==document["parameter_count"]
           and summary["config"]==document["config"],"Public completed recipe/checkpoint differs")
        values=(bench["weighted_clean_accuracy"],bench["weighted_clean_nll"],bench["weighted_mean_stress_accuracy"])
        cases=bench["weighted_cases"]
    elif identity=="se_sam":
        bench=summary["policies"][POLICY];values=(bench["clean_accuracy"],bench["clean_nll"],bench["mean_synthetic_stress_accuracy"]);cases=bench["cases"]
        require(summary["checkpoint_sha256"]==document["checkpoint_sha256"] and summary["parameters"]==document["parameter_count"],"SE completed identity differs")
    elif identity=="wrn16":
        bench=summary["wrn16_completed"];values=(bench["clean_accuracy"],bench["clean_nll"],bench["mean_synthetic_stress_accuracy"]);cases=None
        require(bench["parameter_count"]==document["parameter_count"],"WRN16 public parameter count differs")
    else:
        bench=summary["policies"][POLICY];values=(bench["clean_accuracy"],bench["clean_nll"],bench["stress_accuracy"]);cases=bench["cases"]
        require(summary["raw_selected_checkpoint_sha256"]==document["checkpoint_sha256"] and summary["parameter_count"]==document["parameter_count"],"AugMix selected identity differs")
    require(all(legacy._close(a,b) for a,b in zip(values,(report["clean_accuracy"],report["clean_loss"],report["stress_accuracy"]))),"Public completed metrics differ from full raw benchmark")
    if cases is not None:
        require(set(cases)==set(CASES),"All six public completed cases required")
        for c,row in cases.items():
            require(row.get("examples")==5000 and legacy._close(row.get("accuracy"),report["cases"][c]["accuracy"])
               and legacy._close(row.get("nll",row.get("loss")),report["cases"][c]["loss"]),"Public case values differ from raw benchmark")


def _bootstrap(boot,a,b):
    require(boot.get("original_images")==5000 and boot.get("replicates")==10000 and boot.get("seed")==20261031
       and boot.get("sampling_unit")=="original_image_with_all_five_stress_views" and boot.get("confidence_level")==.95,"Exact original-image cluster bootstrap required")
    ci={}
    for kind,total in (("clean",5000),("stress",25000)):
        row=boot[kind];counts=[legacy._integer(row.get(k),"paired count",low=0,high=total) for k in ("candidate_only_correct","baseline_only_correct","both_correct","both_wrong")]
        delta=a[kind+"_accuracy"]-b[kind+"_accuracy"]
        require(sum(counts)==total and legacy._close(row.get("delta_accuracy"),delta) and legacy._close((counts[0]-counts[1])/total,delta)
           and legacy._close((counts[0]+counts[2])/total,a[kind+"_accuracy"]) and legacy._close((counts[1]+counts[2])/total,b[kind+"_accuracy"]),"Paired marginals/delta differ")
        interval=row.get("percentile_ci95");require(isinstance(interval,list) and len(interval)==2,"Paired interval missing")
        low,high=[legacy._number(v,"paired interval",low=-1,high=1) for v in interval];require(low<=high,"Paired interval inverted")
        ci[kind]=[low,high]  # No new sign/bracketing/promotion requirement.
    return ci


def _ta_pair(files,hashes,ta,baseline,incumbent,row):
    pair,public,frozen=files["ta_pair"],files["ta_pair_summary"],files["ta_frozen"]
    require(pair.get("schema")=="research_trivialaugment_dual_image_bootstrap_v2" and pair.get("status")=="complete"
       and pair.get("candidate_checkpoint_sha256")==TA_CHECKPOINT and pair.get("candidate_benchmark_sha256")==hashes["ta_benchmark"]==TA_BENCHMARK
       and pair.get("incumbent_checkpoint_sha256")==INCUMBENT_CHECKPOINT and pair.get("incumbent_benchmark_sha256")==INCUMBENT_BENCHMARK
       and pair.get("baseline_checkpoint_sha256")==baseline["checkpoint_sha256"] and pair.get("baseline_benchmark_sha256")==baseline["benchmark_sha256"]
       and pair.get("candidate_policy")==pair.get("incumbent_policy")==POLICY and pair.get("baseline_policy")=="flip"
       and pair.get("data_signature")==baseline["data_signature"] and pair.get("evaluation_context")==baseline["evaluation_context"]
       and pair.get("shape")==[5000,6] and pair.get("original_ids_aligned") is True and pair.get("raw_flags_persisted") is False,
       "Actual typed TA dual pair has a different comparator or scope")
    for key in ("original_id_order_sha256","development_index_order_sha256","candidate_correctness_sha256","incumbent_correctness_sha256","baseline_correctness_sha256"):
        legacy._hash(pair.get(key),key)
    old_ci=_bootstrap(pair["legacy_bootstrap"],ta["policies"][POLICY],baseline["flip"])
    inc_ci=_bootstrap(pair["incumbent_bootstrap"],ta["policies"][POLICY],incumbent)
    require(public.get("schema")=="research_trivialaugment_paired_public_results_v2" and public["candidate"]["checkpoint_sha256"]==TA_CHECKPOINT
       and public["candidate"]["benchmark_sha256"]==TA_BENCHMARK and public["candidate"]["fixed_prediction_policy"]==POLICY
       and public["decision"].get("paired_complete") is True and public["decision"].get("primary_frozen") is True,
       "Latest public TA paired stage is not linked to this primary")
    for public_key,raw_key,refpolicy in (("paired_against_current_incumbent","incumbent_bootstrap",POLICY),
                                       ("paired_against_original_baseline","legacy_bootstrap","flip")):
        declared=public[public_key]
        require(declared.get("candidate_policy")==POLICY and declared.get("reference_policy")==refpolicy
           and all(declared.get(k)==pair[raw_key].get(k) for k in ("original_images","replicates","seed","sampling_unit","confidence_level")),
           "Public and raw TA paired intervals/counts differ")
        for kind in ("clean","stress"):
            a,b=declared[kind],pair[raw_key][kind]
            require(all(type(a.get(k)) is int and a[k]==b[k] for k in ("candidate_only_correct","baseline_only_correct","both_correct","both_wrong","original_image_net_wins","original_image_net_losses"))
               and legacy._close(a.get("delta_accuracy"),b["delta_accuracy"])
               and isinstance(a.get("percentile_ci95"),list) and len(a["percentile_ci95"])==2
               and all(legacy._close(x,y) for x,y in zip(a["percentile_ci95"],b["percentile_ci95"])),"Public TA paired scalars differ from raw evidence")
            if "ci_includes_zero" in a:
                require(a["ci_includes_zero"] is (a["percentile_ci95"][0]<=0<=a["percentile_ci95"][1]),"Public CI zero annotation differs")
    require(frozen.get("schema")=="research_trivialaugment_primary_freeze_v2" and frozen.get("status")=="frozen"
       and frozen.get("checkpoint_sha256")==TA_CHECKPOINT and frozen.get("benchmark_sha256")==TA_BENCHMARK
       and frozen.get("paired_sha256")==hashes["ta_pair"] and frozen.get("policy")==POLICY
       and frozen.get("planned_repeat_seed")==2027 and frozen.get("planned_repeat_config_sha256")=="9686ddea85d0f56ba1f58ff8668056c146a78b675f12f0624799f2fd0b2f4981",
       "Actual pre-repeat TA freeze differs")
    require(public["raw_evidence_sha256"]["pair_result"]==hashes["ta_pair"]
       and public["raw_evidence_sha256"]["primary_freeze"]==public["primary_freeze"]["freeze_sha256"]==hashes["ta_frozen"]
       and public["primary_freeze"]["frozen_at"]==frozen["frozen_at"],"Public and actual pair/freeze identities differ")
    legacy._stamp(frozen["frozen_at"]);_source_map(frozen.get("planned_repeat_source_sha256"))
    row.update(clean_ci95=old_ci["clean"],stress_ci95=old_ci["stress"],incumbent_clean_ci95=inc_ci["clean"],incumbent_stress_ci95=inc_ci["stress"],
       paired_status="complete_dual_comparison",repeat_status="pending",delivery_status="awaiting_repeat",qualification_status="awaiting_independent_repeat")


def _ta_repeat(files,hashes,baseline,pixels,row):
    bench,result,receipt=files["ta_repeat_benchmark"],files["ta_repeat_result"],files["ta_repeat_receipt"]
    frozen=files["ta_frozen"];report=_full_benchmark(bench,baseline,pixels,seed=2027)
    expected=frozen["planned_repeat_effective_config"]
    require(bench["config"]==result.get("config")==expected
       and {k:v for k,v in expected.items() if k!="seed"}=={k:v for k,v in files["ta_benchmark"]["config"].items() if k!="seed"}
       and bench["checkpoint_sha256"]!=TA_CHECKPOINT and bench["parameter_count"]==4166298,"Exact fresh27 recipe/checkpoint required")
    marker=bench.get("research_training_integrity",{}).get("training",{})
    require(marker.get("schema")=="research_trivialaugment_training_v1" and marker.get("precision")=="bf16" and marker.get("training_seed")==2027
       and marker.get("ta_rng_seed")==2027 and marker.get("input_config_sha256")==frozen["planned_repeat_config_sha256"]
       and marker.get("effective_config_sha256")==_digest(expected),"Typed TA repeat marker/config differs")
    require(result.get("schema")=="research_trivialaugment_repeat_result_v1" and result.get("status")=="complete"
       and result.get("role")=="development_repeat" and result.get("seed")==2027 and result.get("epochs_completed")==160
       and result.get("checkpoint_sha256")==bench["checkpoint_sha256"] and result.get("benchmark_sha256")==hashes["ta_repeat_benchmark"]
       and result.get("frozen_manifest_sha256")==hashes["ta_frozen"] and result.get("source_sha256")==frozen["planned_repeat_source_sha256"]
       and result.get("config_sha256")==frozen["planned_repeat_config_sha256"] and result.get("repeat_weights_hidden_use_authorized") is False
       and result.get("primary_replacement_authorized") is False and result.get("hidden_images_loaded") is False
       and result.get("confirmation_evaluated") is False,"Complete actual source-bound TA repeat result required")
    require(receipt.get("schema")=="research_trivialaugment_repeat_supervisor_v1" and receipt.get("status")=="complete"
       and receipt.get("mode")=="formal" and receipt.get("backend")=="local" and receipt.get("role")=="development_repeat"
       and receipt.get("seed")==2027 and receipt.get("source_sha256")==result["source_sha256"]
       and receipt.get("artifact")=="repeat_result.json" and receipt.get("artifact_sha256")==hashes["ta_repeat_result"]
       and receipt.get("frozen_manifest_sha256")==hashes["ta_frozen"] and receipt.get("input_manifest_sha256")==result.get("input_manifest_sha256")
       and receipt.get("process_tree_cleanup_verified") is True and receipt.get("confirmation_evaluated") is False,
       "Complete original native TA repeat receipt required")
    require(legacy._stamp(frozen["frozen_at"])<legacy._stamp(receipt["started_at"])<=legacy._stamp(receipt["finished_at"]),"Primary freeze must precede repeat outcomes")
    proofs=receipt.get("process_tree_guard_results",[])
    require(len(proofs)==1 and proofs[0].get("cleanup_verified") is True and proofs[0].get("completion_authenticated") is True
       and proofs[0].get("completion_exit_code")==0 and proofs[0].get("timed_out") is False,"Recorded repeat native cleanup proof missing")
    for key,value in (("successful_updates",50080),("skipped_updates",0),("source_examples_seen",6400000),("model_views_seen",6400000)):
        require(type(result.get(key)) is int and result[key]==value,"TA repeat update/view totals differ")
    inc,_,_= _incremental(report,files["incumbent_benchmark"]["policies"][POLICY])
    floor=all(report["cases"][c]["accuracy"]-files["incumbent_benchmark"]["policies"][POLICY]["cases"][c]["accuracy"]>=-.01-1e-12 for c in CASES[1:])
    passed=legacy._gate(report,baseline["flip"])["passed"] and inc["passed"] and floor
    require(result.get("scalar_screen",{}).get("scalar_screen_passed") is passed and bench.get("point_and_case_gates_passed") is passed,"Actual repeat fixed gate status differs")
    row.update(repeat_status="complete_passed" if passed else "complete_failed",repeat_clean_accuracy=report["clean_accuracy"],
       repeat_stress_accuracy=report["stress_accuracy"],repeat_epochs_completed=160,repeat_complete=True,repeat_passed=passed,
       delivery_status="repeat_screen_passed_unscored" if passed else "not_promoted",
       qualification_status="complete_repeat_screen_passed" if passed else "complete_repeat_screen_failed")
    return passed


def _ta_delivery(files,hashes,row):
    doc,receipt,inputs,audit=[files[k] for k in ("ta_delivery","ta_prediction_receipt","ta_prediction_inputs","ta_tensor_audit")]
    require(row["repeat_status"]=="complete_passed","Completed passing repeat must precede delivery projection")
    require(doc.get("schema")=="research_trivialaugment_prediction_delivery_v1" and doc.get("status")=="complete"
       and doc.get("checkpoint_sha256")==TA_CHECKPOINT and doc.get("frozen_manifest_sha256")==hashes["ta_frozen"]
       and doc.get("repeat_checkpoint_sha256")==files["ta_repeat_benchmark"]["checkpoint_sha256"]
       and doc.get("repeat_receipt_sha256")==hashes["ta_repeat_receipt"] and doc.get("policy")==POLICY and doc.get("seed")==2026
       and doc.get("training_examples")==40000 and doc.get("parameter_count")==4166298 and doc.get("rows")==2000
       and doc.get("ids_match_official_order") is True and doc.get("hidden_test_labels_loaded") is False
       and doc.get("repeat_weights_used_for_hidden_prediction") is False and doc.get("confirmation_evaluated") is False,
       "Actual frozen-primary-only TA delivery identity differs")
    legacy._hash(doc.get("submission_sha256"),"TA prediction CSV")
    require(receipt.get("schema")=="research_trivialaugment_prediction_supervisor_v1" and receipt.get("status")=="complete"
       and receipt.get("backend")=="local" and receipt.get("mode")=="predict" and receipt.get("process_tree_cleanup_verified") is True
       and receipt.get("artifact_sha256")==hashes["ta_delivery"] and receipt.get("input_manifest_sha256")==hashes["ta_prediction_inputs"]
       and doc.get("prediction_input_manifest_sha256")==hashes["ta_prediction_inputs"]
       and inputs.get("schema")=="research_trivialaugment_prediction_inputs_v1" and inputs.get("checkpoint_sha256")==TA_CHECKPOINT
       and doc.get("input_sha256")==receipt.get("input_sha256")==inputs.get("input_sha256")
       and doc.get("source_sha256")==receipt.get("source_sha256")==inputs.get("source_sha256"),"Original closed TA prediction receipt/input/source differs")
    _source_map(doc["source_sha256"])
    require(audit.get("schema")=="research_trivialaugment_guarded_tensor_audit_v1" and audit.get("status")=="complete"
       and audit.get("performed_in_authenticated_prediction_worker") is True and audit.get("frozen_manifest_sha256")==hashes["ta_frozen"]
       and doc.get("tensor_audit_sha256")==hashes["ta_tensor_audit"] and audit["artifacts"]["primary"]["checkpoint_sha256"]==TA_CHECKPOINT
       and audit["artifacts"]["repeat"]["checkpoint_sha256"]==files["ta_repeat_benchmark"]["checkpoint_sha256"]
       and all(audit["artifacts"][k].get("actual_selected_last_optimizer_parent_and_TA_RNG_reloaded") is True for k in ("primary","repeat")),
       "Actual guarded typed tensor-audit reference differs")
    require(legacy._stamp(files["ta_repeat_receipt"]["finished_at"])<=legacy._stamp(receipt["started_at"])<=legacy._stamp(doc["created_at"])<=legacy._stamp(receipt["finished_at"]),"Repeat/prediction chronology differs")
    row.update(delivery_status="prediction_complete_unscored",qualification_status="typed_delivery_complete")


def _ta_receipt(record,checksum):
    expected={"competition":legacy.COMPETITION,"team":"Saber","status":"complete","seed":2026,"training_examples":40000,
       "parameter_count":4166298,"policy":POLICY,"checkpoint_sha256":TA_CHECKPOINT,"delivery_schema":"research_trivialaugment_prediction_delivery_v1",
       "run":"wrn34_trivialaugment_bf16_s2026","repeat_receipt_schema":"research_trivialaugment_repeat_supervisor_v1"}
    require(all(type(record.get(k)) is type(v) and record[k]==v for k,v in expected.items()) and record.get("schema") is None
       and record.get("composition") in (None,False),"A canonical actual scored TA receipt is required")
    require(isinstance(record.get("run"),str) and legacy.RUN.fullmatch(record["run"]) and not record["run"].startswith("stability"),"Scored TA run identity differs")
    for key in ("submission_sha256","manifest_sha256","frozen_manifest_sha256","repeat_receipt_sha256","repeat_validator_source_sha256"):legacy._hash(record.get(key),key)
    public=legacy._number(record.get("public_score"),"actual official public score",high=1)
    private=record.get("private_score");private=legacy._number(private,"actual official private score",high=1) if private is not None else None
    require(legacy._stamp(record["submitted_at"])<=legacy._stamp(record["checked_at"]),"Scored receipt timestamps differ")
    return {k:record[k] for k in ("run","submission_sha256","checkpoint_sha256","policy","training_examples")} | {
       "submission_ref":legacy._integer(record.get("submission_ref"),"official reference"),"public_score":public,"private_score":private,
       "receipt_sha256":checksum,"validation_examples":record.get("validation_examples")}


def _receipts(records,base,files,hashes,rows,original):
    require(isinstance(records,list),"Explicit canonical scored receipt list required")
    old_refs={r["submission_ref"] for r in original};old_csv={r["submission_sha256"] for r in original}
    seen={};by_ref={};by_csv={};public_sources=[]
    for item in records:
        record,source,path=_read_record(item,base);is_ta=record.get("delivery_schema")=="research_trivialaugment_prediction_delivery_v1"
        if is_ta:
            require(DELIVERY<=set(files) and REPEAT<=set(files),"Scored TA requires the explicit completed repeat/delivery chain")
            projected=_ta_receipt(record,item["sha256"])
            require(record["manifest_sha256"]==hashes["ta_delivery"] and record["frozen_manifest_sha256"]==hashes["ta_frozen"]
               and record["repeat_receipt_sha256"]==hashes["ta_repeat_receipt"]
               and record["repeat_validator_source_sha256"]==files["ta_delivery"]["repeat_validator_source_sha256"]
               ==files["ta_frozen"]["planned_repeat_source_sha256"]["research/run_trivialaugment_repeat.py"]
               and record["submission_sha256"]==files["ta_delivery"]["submission_sha256"],"Actual scored TA receipt does not bind its delivery chain")
        else:
            raw,_,projected=legacy._receipt(path);require(raw.get("status")=="complete","Pending/intent records are not scored receipts")
        ref,digest=projected["submission_ref"],projected["submission_sha256"]
        require(path.parent in {(ROOT/"results/research/submission_receipts").resolve(),(ROOT/"docs/research-submission-receipts").resolve()}
           and path.name==str(ref)+".json","Use only canonical reference-named scored receipts")
        require(ref not in old_refs and digest not in old_csv,"Original entries cannot be relabeled as new research")
        require(by_ref.get(ref,digest)==digest and by_csv.get(digest,ref)==ref,"Official reference/CSV identities conflict")
        by_ref[ref]=digest;by_csv[digest]=ref
        matches=[r for r in rows if r["checkpoint_sha256"]==projected["checkpoint_sha256"] and r["policy"]==projected["policy"]
           and record["training_examples"]==40000 and record["parameter_count"]==r["parameters"]]
        require(len(matches)==1,"Scored receipt needs a unique audited checkpoint/policy/parameter scope")
        require((is_ta and matches[0].get("delivery_status") in {"prediction_complete_unscored","scored"}) or
           (not is_ta and matches[0]["id"] in {"wrn34_160","sam","resizemix","wrn34_300"}
            and matches[0].get("gate_passed") is True and matches[0].get("repeat_status")=="complete_passed"),
           "A failed/pending method cannot acquire qualified status from an unrelated receipt")
        projected["recipe_id"]=matches[0]["id"];projected["label"]=matches[0]["label"]
        key=(ref,digest)
        if key in seen:require(seen[key]==projected,"Duplicate canonical receipt metadata/score conflicts")
        seen[key]=projected;public_sources.append({"role":"official_receipt","submission_ref":ref,**source})
        matches[0].update(delivery_status="scored",submission_status="scored",public_score=projected["public_score"])
    return sorted(seen.values(),key=lambda r:r["submission_ref"]),public_sources


def load_campaign(manifest_path):
    """Read only explicit JSON/hash roles and project audited public scalars."""
    require(_sha(ROOT/"scripts/research_report_evidence.py")==LEGACY_SOURCE_SHA,"Frozen original report adapter bytes changed")
    path=Path(manifest_path).resolve();manifest_sha=_sha(path);manifest=legacy._json(path)
    require(manifest.get("schema")==INPUT_SCHEMA and set(manifest)=={"schema","recorded_at","files","scored_receipts"},"Explicit campaign report input schema required")
    recorded=legacy._stamp(manifest["recorded_at"])
    specs=manifest["files"];require(isinstance(specs,dict) and REQUIRED<=set(specs) and set(specs)<=REQUIRED|REPEAT|DELIVERY,"Unknown/missing campaign evidence roles")
    for group in (REPEAT,DELIVERY):require(not (set(specs)&group) or group<=set(specs),"Optional complete evidence groups cannot be partial")
    require(not (set(specs)&DELIVERY) or REPEAT<=set(specs),"Delivery requires the complete actual repeat group")
    files={};hashes={};sources=[]
    for role,record in specs.items():
        document,public,_=_read_record(record,path.parent);files[role]=document;hashes[role]=record["sha256"];sources.append({"role":role,**public})
    summary=files["legacy_summary"];baseline=summary["baseline"]
    require(summary.get("schema")=="audited_followup_research_aggregate_v1","Explicit frozen original follow-up snapshot required")
    for document in files.values():
        if "recorded_at" in document:require(legacy._stamp(document["recorded_at"])<=recorded,"Input snapshot is newer than the declared reporting time")
    for role in ("ta_repeat_receipt","ta_prediction_receipt"):
        if role in files:require(legacy._stamp(files[role]["finished_at"])<=recorded,"Completed optional evidence postdates the reporting snapshot")
    require(hashes["baseline_benchmark"]==baseline["benchmark_sha256"] and files["baseline_benchmark"]["checkpoint_sha256"]==baseline["checkpoint_sha256"]
       and baseline["evaluation_context"].get("device_type")=="cuda" and baseline["evaluation_context"].get("amp_dtype")=="torch.bfloat16",
       "Actual complete fixed native baseline required")
    pixels={c:files["baseline_benchmark"]["policies"]["flip"]["cases"][c]["input_pixel_sha256"] for c in CASES}
    _origin_baseline(files["baseline_benchmark"],baseline,pixels,hashes["baseline_benchmark"])
    require(hashes["incumbent_benchmark"]==INCUMBENT_BENCHMARK and files["incumbent_benchmark"]["checkpoint_sha256"]==INCUMBENT_CHECKPOINT,"Fixed current-incumbent bytes required")
    incumbent=_full_benchmark(files["incumbent_benchmark"],baseline,pixels)
    old_best,old_primary,original=legacy._legacy(_read_record(specs["legacy_results"],path.parent)[2])
    rows=_legacy_rows(summary,baseline,pixels,incumbent)
    methods=(("trivialaugment","TA / WRN34-3","ta",4166298,160,"bf16","research_trivialaugment_completed_public_results_v1"),
       ("se_sam","SE+SAM / WRN22-4","se",4332654,160,"bf16","se_sam_completed_public_scalar_v1"),
       ("resizemix300","ResizeMix / WRN22-4 / 300","resizemix300",4298970,300,"bf16","research_resizemix300_public_results_v1"),
       ("wrn16","WRN16-5 / 160","wrn16",4289754,160,"bf16","research_reassessment_scalar_v1"),
       ("augmix_nativebn","AugMix native BN / WRN22-4","augmix",4298970,160,"fp16","research_augmix_nativebn_public_results_v1"))
    for identity,label,key,params,epochs,precision,schema in methods:
        public,document=files[key+"_summary"],files[key+"_benchmark"]
        require(public.get("schema")==schema and document.get("parameter_count")==params and document.get("config",{}).get("epochs")==epochs,"Declared method architecture/epochs/schema differs")
        report=_full_benchmark(document,baseline,pixels);_public_values(public,document,identity)
        row=_row(identity,label,document,report,baseline,incumbent,precision,epochs)
        integrity=document.get("research_training_integrity",{})
        if identity!="wrn16":require(integrity.get("training",{}).get("precision")==precision,"Recorded method training precision differs")
        if "source_sha256" in public:_source_map(public["source_sha256"])
        row["benchmark_sha256"]=hashes[key+"_benchmark"];row["source_summary_sha256"]=hashes[key+"_summary"]
        row["submission_status"]="not_submitted"
        if identity in {"trivialaugment","resizemix300"}:
            require(public["raw_evidence_sha256"]["benchmark"]==hashes[key+"_benchmark"],"Public/raw method benchmark binding differs")
            require(public["original_baseline_screen"]["passed"] is row["original_gate_passed"]
               and public["incumbent_screen"]["mean_gate_passed"] is row["incumbent_mean_passed"]
               and public["incumbent_screen"]["individual_stress_gate_passed"] is row["case_floor_passed"],"Published prospective decision differs")
            training=public["training"]
            if identity=="trivialaugment":
                require(document["checkpoint_sha256"]==TA_CHECKPOINT and hashes["ta_benchmark"]==TA_BENCHMARK and row["gate_passed"] is True,"Exact completed TA primary screen required")
                row["workload"]=_workload(training,{"successful_updates":50080,"skipped_updates":0,"source_examples_seen":6400000,"model_views_seen":6400000,"forward_passes":50080},integrity)
            else:
                require(public["decision"]["outcome"]=="rejected_before_pairing" and row["gate_passed"] is False,"ResizeMix300 negative decision differs")
                row["workload"]=_workload(training,{"successful_updates":93900,"skipped_updates":0,"source_examples_seen":12000000},integrity)
        elif identity=="se_sam":
            require(public["benchmark_sha256"]==hashes["se_benchmark"] and public["original_fixed_weighted_gate_passed"] is row["original_gate_passed"]
               and public["dual_gate_passed"] is True and public["fixed_SAM_incremental_weighted_gate_passed"] is True
               and public["prospective_incumbent"]["scalar_improvement_gate_passed"] is row["incumbent_mean_passed"]
               and public["prospective_incumbent"]["per_case_gate_passed"] is row["case_floor_passed"] and row["gate_passed"] is False,
               "SE original dual versus prospective rejection must remain distinct")
            row["workload"]=_workload(public,{"successful_updates":50080,"skipped_updates":0,"sam_forward_passes":100160},integrity)
        elif identity=="wrn16":
            require(public["wrn16_completed"]["old_baseline_gate_passed"] is row["original_gate_passed"],"Historical WRN16 original gate differs")
            old=public["wrn16_completed"]
            require(legacy._close(old["incumbent_same_policy_clean_delta"],row["incumbent_clean_delta"])
               and legacy._close(old["incumbent_same_policy_mean_stress_delta"],row["incumbent_stress_delta"])
               and legacy._close(old["incumbent_same_policy_noise_delta"],row["noise_delta_vs_incumbent"]),"Recorded retrospective WRN16 comparison differs from raw benchmark")
            row.update(gate_passed=False,qualification_status="not_prioritized_after_recorded_incumbent_comparison",
               incumbent_comparison_scope="retrospective_descriptive_not_a_preregistered_rejection",workload=None)
        else:
            require(public["raw_benchmark_sha256"]==hashes["augmix_benchmark"] and public["promotion_passed"] is False
               and public["qualification"]=="rejected_all_policy_clean_noninferiority" and row["original_gate_passed"] is False,"AugMix original clean rejection differs")
            row.update(qualification_status="rejected_original_clean_noninferiority",incumbent_comparison_scope="retrospective_descriptive_not_a_changed_gate")
            row["workload"]=_workload(public,{"successful_updates":50068,"skipped_updates":12,"attempted_updates":50080,
               "source_examples_seen":6400000,"model_views_seen":19200000,"forward_calls":50080},integrity)
        rows.append(row)
    ta=next(r for r in rows if r["id"]=="trivialaugment")
    _ta_pair(files,hashes,files["ta_benchmark"],baseline,incumbent,ta)
    ta["submission_status"]="unscored_or_not_submitted"
    if REPEAT<=set(files):_ta_repeat(files,hashes,baseline,pixels,ta)
    if DELIVERY<=set(files):_ta_delivery(files,hashes,ta)
    for item in manifest["scored_receipts"]:
        require(legacy._stamp(_read_record(item,path.parent)[0]["checked_at"])<=recorded,"Official receipt postdates the reporting snapshot")
    scored,receipt_sources=_receipts(manifest["scored_receipts"],path.parent,files,hashes,rows,original)
    order=("trivialaugment","wrn34_300","se_sam","resizemix300","wrn16","augmix_nativebn","sam","swa40","wrn34_160","resizemix")
    rows.sort(key=lambda r:order.index(r["id"]))
    best=max((r["public_score"] for r in scored),default=None)
    projection={"schema":SCHEMA,"recorded_at":manifest["recorded_at"],"repository":legacy.REPOSITORY,"source_manifest_sha256":manifest_sha,
       "baseline":{"clean_accuracy":baseline["flip"]["clean_accuracy"],"clean_nll":baseline["flip"]["clean_loss"],"stress_accuracy":baseline["flip"]["stress_accuracy"],
          "policy":"flip","examples":5000,"training_examples":40000,"checkpoint_sha256":baseline["checkpoint_sha256"],"benchmark_sha256":baseline["benchmark_sha256"]},
       "rows":rows,"official_submissions":scored,"original_primary_public":old_primary,"original_best_public":old_best,
       "best_new_public":best,"best_observed_public":max(old_best,best) if best is not None else old_best,"unscored_receipts_count":0,
       "sources":sources+receipt_sources,"legacy_adapter_source_sha256":LEGACY_SOURCE_SHA,"adapter_source_sha256":_sha(__file__),
       "references":[{"title":"Mueller and Hutter. TrivialAugment (ICCV2021)","url":"https://arxiv.org/html/2103.10158v2"},
          {"title":"Hendrycks et al. AugMix (ICLR2020)","url":"https://arxiv.org/abs/1912.02781"},
          {"title":"Hu et al. Squeeze-and-Excitation Networks (CVPR2018)","url":"https://arxiv.org/abs/1709.01507"}],
       "limitations":["All follow-on rows reuse5000 development images; none inherits or repeats the original one-time confirmation.",
          "Five synthetic stress views are engineering proxies, not hidden accuracy or evidence of the hidden distribution.",
          "clean/stress CI fields compare to original baseline flip; incumbent CI fields compare to fixed WRN300 weighted.10000 original-image cluster draws keep five stress views together.",
          "Conditional intervals do not establish independent confirmation, training-seed variance or the private winner. Only canonical complete official receipts establish scores.",
          "This adapter checks JSON, current source bytes, hashes, context, pixels and scalars; tensor/OS execution audits remain attributed to the archived typed engines."],
       "engineering":["Historical protocols and negatives are preserved; WRN16/AugMix incumbent deltas are retrospective descriptions, not rewritten preregistered decisions.",
          "TA completed and paired snapshots are joined by exact checkpoint/benchmark/policy; its older paired=false field does not override the later real dual pair.",
          "Training precision and workload differ: SE uses SAM double forwards; AugMix has overflow skips and three views. No missing WRN16 update ledger is invented."]}
    require(_sha(path)==manifest_sha and all(_sha(_read_record(spec,path.parent)[2])==spec["sha256"] for spec in [*specs.values(),*manifest["scored_receipts"]]),"Report inputs changed during projection")
    return projection


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--manifest",type=Path);args=parser.parse_args(argv)
    if args.manifest is None:parser.print_help();return
    print(json.dumps(load_campaign(args.manifest),sort_keys=True,indent=2,allow_nan=False))


if __name__=="__main__":main()
