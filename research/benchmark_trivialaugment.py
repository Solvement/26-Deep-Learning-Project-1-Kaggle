"""Typed full native-v3 TA benchmark; source/state audit precedes scoring.

This is an independent TA method schema. Neither original baseline gates nor
frozen incumbent decisions change; no pairing, prediction or upload is added.
"""
from __future__ import annotations

from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from runtime import seed_all, write_json
import argparse
import hashlib
import json
import math
import time
import torch
from data import load_training_arrays
from model import build_model, count_parameters
from research import benchmark as core
from research import incumbent_compare as incumbent
from research import paired_compare_v2 as original_pair
from research import train_trivialaugment as trainer
from train import code_provenance, tensor_images, training_source

POLICY="center80_shift20"
PROSPECTIVE_PLAN="research/incumbent-promotion-plan.md"
PROSPECTIVE_PLAN_SHA="10e59e56f8c2a3c1f9ae0ad2681c3db5b3129ffd2d33c6f4d3087710e2860ec7"
INCUMBENT_CHECKPOINT=ROOT/"results/research/colab_wrn34_3_300_fp16_s2026_v3/candidate/best.pt"
BASELINE_SHA="eced314d97fcb5ddef4cad25c6968ba299d0e8a2c79af6bdcbdbf305b79c632d"
require=trainer.require


def selected_epoch(history):
    return trainer.best_selection(history)


def validate_training(checkpoint,metrics,history,provenance,environment,last,raw):
    """Actual selected/last tensors, optimizer, RNG and all160 epochs required."""
    config=checkpoint.get("config",{});trainer.validate_config(config);raw=trainer.input_config(config,raw)
    marker=checkpoint.get("provenance",{}).get("research_training",{})
    expected=trainer.research_provenance(marker.get("precision"),config,hashlib.sha256(raw).hexdigest())
    require(all(document.get("schema")==trainer.SCHEMA for document in (checkpoint,metrics,provenance,environment,last)),"TA typed schema mismatch")
    require(marker==expected==metrics.get("research_training")==provenance.get("code",{}).get("research_training")==environment.get("research_training"),"TA source/raw/effective configuration/RNG/precision mismatch")
    require(checkpoint.get("provenance")==provenance.get("code")==last.get("provenance")
            and checkpoint["provenance"].get("source_sha256")==code_provenance()["source_sha256"],"Actual frozen core provenance differs")
    commit=checkpoint["provenance"].get("git_commit")
    require(isinstance(commit,str) and len(commit)==40 and all(c in "0123456789abcdef" for c in commit),"Recorded training source revision required")
    model_config={k:config[k] for k in ("depth","widen_factor","dropout","num_classes")}
    require(checkpoint.get("model_config")==last.get("model_config")==model_config
            and checkpoint.get("parameter_count")==metrics.get("parameter_count")==provenance.get("parameter_count")==last.get("parameter_count")==trainer.PARAMETERS,
            "TA model constructor or exact parameter count differs")
    require(environment.get("trivialaugment_training_precision")==expected["precision"] and environment.get("grad_scaler_enabled") is False,"Actual unscaled training precision differs")
    require(provenance.get("normalization")=={"mean":list(trainer.MEAN),"std":list(trainer.STD)},"Parent normalization differs")
    require(metrics.get("status")=="complete" and type(metrics.get("epochs_completed")) is int and metrics["epochs_completed"]==160
            and metrics.get("config")==last.get("config")==config and metrics.get("training_examples")==provenance.get("train_examples")==40000
            and metrics.get("validation_examples")==provenance.get("validation_examples")==5000
            and provenance.get("data_signature")==last.get("data_signature")==checkpoint.get("data_signature"),"Complete official40k/5000 and160 epochs required")
    counts=trainer.validate_history(history,160)
    totals={"successful_updates":50080,"skipped_updates":0,"forward_passes":50080,"source_examples_seen":6400000,"model_views_seen":6400000}
    require(all(type(metrics.get(k)) is int and metrics[k]==value for k,value in totals.items())
            and metrics.get("ta_operation_counts")==counts and sum(counts.values())==6400000,"Complete one-view TA totals differ")
    require(last.get("formal_training_state") is True and last.get("preflight_state") is False and type(last.get("epoch")) is int and last["epoch"]==160
            and last.get("history")==history and last.get("updates")==50080 and last.get("forward_passes")==50080 and last.get("skipped_updates")==0
            and last.get("source_examples_seen")==last.get("model_views_seen")==6400000 and last.get("ta_operation_counts")==counts
            and last.get("input_config_sha256")==hashlib.sha256(raw).hexdigest()
            and last.get("ta_rng_state_sha256")==metrics.get("ta_rng_state_sha256")==history[-1]["ta_rng_state_sha256"],"Actual formal last-state TA ledger differs")
    choice,best=selected_epoch(history)
    require(choice==(checkpoint.get("epoch"),checkpoint.get("variant"))
            and best==(metrics.get("best_validation_accuracy"),-metrics.get("best_validation_loss",float("inf")))
            and (checkpoint.get("validation_accuracy"),-checkpoint.get("validation_loss",float("inf")))==best
            and last.get("best_accuracy")==best[0] and last.get("best_loss")==-best[1]
            and metrics.get("elapsed_seconds")==last.get("elapsed_seconds")==history[-1]["elapsed_seconds"]
            and checkpoint.get("selected_ta_rng_state_sha256")==history[choice[0]-1]["ta_rng_state_sha256"],"Actual selected raw/EMA epoch and TA state fingerprint differ")
    trainer.validate_saved_state(last,config)
    with torch.random.fork_rng(devices=[]):reference=build_model(model_config)
    require(count_parameters(reference)==count_parameters(reference,True)==trainer.PARAMETERS,"Actual total/trainable parameter count differs")
    expected_state=reference.state_dict();actual=checkpoint.get("model",{})
    require(set(actual)==set(expected_state),"Selected TA tensor topology differs")
    for name,value in expected_state.items():
        t=actual[name]
        require(isinstance(t,torch.Tensor) and t.shape==value.shape and t.dtype==value.dtype and torch.isfinite(t).all().item(),"Selected TA tensor shape/dtype/finite state differs")
        if name.endswith("num_batches_tracked"):require(t.item()==checkpoint["epoch"]*313,"Selected BN one-forward counter differs")
    return {"schema":"research_trivialaugment_integrity_v1","training":expected,**totals,
            "ta_operation_counts":counts,"ta_rng_state_sha256":last["ta_rng_state_sha256"],
            "formal_last_state_tensor_optimizer_parent_and_TA_RNG_audited":True,
            "confirmation_evaluated":False,"hidden_images_loaded":False,"external_data_loaded":False}


def prospective_gate(reports,reference,signature,context):
    """Same frozen weighted incumbent; no mean can conceal a >1pp case drop."""
    require(core.sha256(ROOT/PROSPECTIVE_PLAN)==PROSPECTIVE_PLAN_SHA,"Predeclared prospective decision protocol changed")
    require(reference.get("schema")==core.SCHEMA and reference.get("status")=="complete" and reference.get("examples")==5000
            and reference.get("training_examples")==40000 and reference.get("checkpoint_sha256")==incumbent.INCUMBENT_CHECKPOINT_SHA
            and reference.get("data_signature")==signature and reference.get("evaluation_context")==context and reference.get("batch_size")==128
            and reference.get("bn_recalibrated") is False and reference.get("benchmark_source_sha256")==core.sha256(ROOT/"research/benchmark.py")
            and reference.get("stress_cases")==list(core.CASES) and set(reference.get("policies",{}))==set(core.POLICIES)
            and set(reports)==set(core.POLICIES),"Complete identical native local incumbent context required")
    for policy in core.POLICIES:
        for case in core.CASES:
            a=reports[policy]["cases"][case];b=reference["policies"][policy]["cases"][case]
            require(type(a.get("examples")) is int and type(b.get("examples")) is int and a["examples"]==b["examples"]==5000
                    and a.get("input_pixel_sha256")==b.get("input_pixel_sha256")
                    and all(type(row.get("accuracy")) in (int,float) and math.isfinite(row["accuracy"]) and 0<=row["accuracy"]<=1
                            and type(row.get("loss")) in (int,float) and math.isfinite(row["loss"]) and row["loss"]>=0 for row in (a,b)),"All18 candidate/incumbent pixel/count/finite bindings required")
        for row in (reports[policy],reference["policies"][policy]):
            require(row.get("clean_accuracy")==row["cases"]["clean"]["accuracy"] and row.get("clean_loss")==row["cases"]["clean"]["loss"]
                    and type(row.get("stress_accuracy")) in (int,float) and math.isfinite(row["stress_accuracy"])
                    and math.isclose(row["stress_accuracy"],sum(row["cases"][case]["accuracy"] for case in core.CASES[1:])/5,rel_tol=0,abs_tol=1e-12),"Complete case/summary arithmetic differs")
    report,base=reports[POLICY],reference["policies"][POLICY]
    gate=core.promotion_gate(report,base)
    deltas={case:report["cases"][case]["accuracy"]-base["cases"][case]["accuracy"] for case in core.CASES[1:]}
    failed=[case for case,delta in deltas.items() if delta<-.01-1e-12]
    return {"policy":POLICY,"incumbent_checkpoint_sha256":incumbent.INCUMBENT_CHECKPOINT_SHA,
            "incumbent_benchmark_sha256":incumbent.INCUMBENT_BENCHMARK_SHA,"mean_gate":gate,
            "clean_nll_delta":report["clean_loss"]-base["clean_loss"],"case_accuracy_deltas":deltas,
            "noise_accuracy_delta":deltas["noise_004"],"maximum_allowed_case_accuracy_drop":.01,
            "per_case_noninferiority_passed":not failed,"failed_cases":failed,"passed":gate["passed"] and not failed,
            "prospective_plan_sha256":PROSPECTIVE_PLAN_SHA,"automatic_delivery_authorization":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--baseline",type=Path,required=True);parser.add_argument("--data-dir",type=Path,default=ROOT/"data")
    parser.add_argument("--device",choices=("cuda",),default="cuda");args=parser.parse_args()
    require(not args.output.exists(),"Existing TA benchmark cannot be overwritten")
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False),"Full TA benchmark requires actual native local BF16")
    device=torch.device(args.device);seed_all(2026);started=time.perf_counter();folder=args.checkpoint.parent
    reference_path=incumbent.INCUMBENT_PATH
    require(core.sha256(reference_path)==incumbent.INCUMBENT_BENCHMARK_SHA
            and core.sha256(INCUMBENT_CHECKPOINT)==incumbent.INCUMBENT_CHECKPOINT_SHA
            and core.sha256(args.baseline)==BASELINE_SHA,"Actual frozen baseline/incumbent benchmark/checkpoint differ")
    paths=[args.checkpoint,*(folder/name for name in ("last.pt","metrics.json","history.json","provenance.json","environment.json","config.json","input_config.json")),
           args.baseline,reference_path,INCUMBENT_CHECKPOINT,ROOT/PROSPECTIVE_PLAN,
           *(INCUMBENT_CHECKPOINT.parent/name for name in ("metrics.json","history.json","provenance.json","environment.json")),
           reference_path.with_name(reference_path.stem+".core.json"),Path(__file__),ROOT/"research/paired_compare_v2.py",ROOT/"research/paired_compare.py",
           ROOT/"research/benchmark_shadow.py",ROOT/"research/benchmark.py",ROOT/"research/incumbent_compare.py",ROOT/trainer.BASE_CONFIG,
           *(args.data_dir/name for name in ("training_manifest.json","official_train_images.npy","official_train_labels.npy","official_train_ids.npy","official_split_indices.npz")),
           *(ROOT/name for name in trainer.SOURCE_KEYS),*(ROOT/name for name in code_provenance()["source_sha256"])]
    hashes={str(path.resolve()):core.sha256(path) for path in paths}
    checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=False);last=torch.load(folder/"last.pt",map_location="cpu",weights_only=False)
    metrics,history,provenance,env=[json.loads((folder/name).read_text()) for name in ("metrics.json","history.json","provenance.json","environment.json")]
    require(json.loads((folder/"config.json").read_text())==checkpoint["config"],"Actual effective config file differs")
    integrity=validate_training(checkpoint,metrics,history,provenance,env,last,(folder/"input_config.json").read_bytes());del last
    require(integrity["training"]["precision"]=="bf16" and env.get("device")=="cuda" and env.get("amp_dtype")=="torch.bfloat16",
            "Formal promotion benchmark rejects CPU math-test training states")
    images,labels,train_ids,val_ids=load_training_arrays(args.data_dir);manifest=json.loads((args.data_dir/"training_manifest.json").read_text())
    signature={"archive_sha256":training_source(args.data_dir)["archive_sha256"],"split_sha256":manifest["split"]["indices_sha256"]}
    core_source=code_provenance()["source_sha256"]
    context={"device_type":device.type,"amp_dtype":str(core.benchmark_dtype(device)),"torch_version":torch.__version__,"training_core_sha256":core_source}
    require(context==incumbent.CONTEXT and signature==incumbent.DATA,"Exact local native v3 context/split required")
    core.validate_checkpoint(checkpoint,train_ids,val_ids,signature,metrics,core_source,history)
    old_reference,reference=original_pair.load_experiment(INCUMBENT_CHECKPOINT,reference_path,signature,context,core_source,train_ids,val_ids);del old_reference
    model=build_model(checkpoint["model_config"]).to(device=device,memory_format=torch.channels_last);model.load_state_dict(checkpoint["model"],strict=True);model.eval()
    val_images,val_labels=tensor_images(images[val_ids],device),torch.as_tensor(labels[val_ids],device=device);del images,labels
    reports={policy:core.score(model,val_images,val_labels,device,policy,128) for policy in core.POLICIES}
    baseline=json.loads(args.baseline.read_text());core.validate_baseline(baseline,reports,signature,128,context,core.sha256(ROOT/"research/benchmark.py"))
    gates={policy:core.promotion_gate(report,baseline["policies"]["flip"]) for policy,report in reports.items()}
    prospective=prospective_gate(reports,reference,signature,context)
    require(all(core.sha256(path)==checksum for path,checksum in hashes.items()),"TA training/reference/source evidence changed during benchmark")
    result={"schema":core.SCHEMA,"status":"complete","examples":5000,"training_examples":40000,"parameter_count":trainer.PARAMETERS,
            "checkpoint_sha256":core.sha256(args.checkpoint),"checkpoint_epoch":checkpoint["epoch"],"checkpoint_variant":checkpoint["variant"],
            "model_config":checkpoint["model_config"],"config":checkpoint["config"],"data_signature":signature,"stress_cases":list(core.CASES),
            "batch_size":128,"evaluation_context":context,"bn_recalibrated":False,"policies":reports,"promotion":gates,
            "fixed_primary_policy":POLICY,"prospective_incumbent":prospective,
            "point_and_case_gates_passed":gates[POLICY]["passed"] and prospective["passed"],
            "baseline_sha256":core.sha256(args.baseline),"fixed_incumbent_benchmark_sha256":core.sha256(reference_path),
            "training_metrics_sha256":core.sha256(folder/"metrics.json"),"training_history_sha256":core.sha256(folder/"history.json"),
            "source":code_provenance(),"benchmark_source_sha256":core.sha256(ROOT/"research/benchmark.py"),
            "trivialaugment_benchmark_source_sha256":core.sha256(Path(__file__)),"research_training_integrity":integrity,
            "evidence_sha256":hashes,"elapsed_seconds":time.perf_counter()-started,
            "confirmation_evaluated":False,"hidden_images_loaded":False,"external_data_loaded":False,
            "paired_comparison_performed":False,"delivery_authorized":False,"prediction_authorized":False,
            "interpretation":"Typed complete TA training and same native-v3 benchmark. Fixed weighted original gate plus prospective incumbent mean/case criteria are screening only. Pairing, frozen primary, exact fresh27 and explicit TA delivery support remain required; no hidden-score guarantee."}
    write_json(args.output,result)
    print(json.dumps({"event":"trivialaugment_benchmark_complete","point_and_case_gates_passed":result["point_and_case_gates_passed"],"output":str(args.output)}),flush=True)


if __name__=="__main__":main()
