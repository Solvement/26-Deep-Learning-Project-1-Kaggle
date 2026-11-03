"""Typed complete scratch-KD benchmark. Importing this module is stdlib-only.

The controller authenticates its worker before calling run_benchmark. No unknown
KD state is passed through an old method marker or delivery allowlist.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research import train_scratch_kd as trainer
POLICY="center80_shift20"
PROTOCOL="research/incumbent-promotion-plan.md"
PROTOCOL_SHA="10e59e56f8c2a3c1f9ae0ad2681c3db5b3129ffd2d33c6f4d3087710e2860ec7"
BASELINE_SHA="eced314d97fcb5ddef4cad25c6968ba299d0e8a2c79af6bdcbdbf305b79c632d"
require=trainer.require


def validate_training(checkpoint,metrics,history,provenance,environment,last,raw,teacher_reference):
    """Reloaded selected/last/SGD/RNG plus complete exact teacher lineage."""
    config=checkpoint.get("config",{});trainer.validate_config(config)
    marker=checkpoint.get("provenance",{}).get("research_training",{})
    expected=trainer.research_provenance(config,raw,teacher_reference,marker.get("precision"))
    require(all(obj.get("schema")==trainer.SCHEMA for obj in (checkpoint,metrics,provenance,environment,last)),"Independent KD schemas required")
    require(marker==expected==metrics.get("research_training")==provenance.get("code",{}).get("research_training")==environment.get("research_training"),"Exact KD/control source/config/teacher marker differs")
    m=trainer._deps();t=m.t
    require(checkpoint.get("provenance")==last.get("provenance")==provenance.get("code") and
        checkpoint["provenance"].get("source_sha256")==m.core.code_provenance()["source_sha256"],"Actual core lineage differs")
    role="development_control" if config["kd_alpha"]==0 else "primary" if config["seed"]==2026 else "development_repeat"
    require(all(obj.get("role")==role for obj in (checkpoint,metrics,provenance,environment,last)),"Control/primary/repeat role differs")
    model_config={k:config[k] for k in ("depth","widen_factor","dropout","num_classes")}
    require(checkpoint.get("model_config")==last.get("model_config")==model_config and
        all(obj.get("parameter_count")==trainer.PARAMETERS for obj in (checkpoint,metrics,provenance,last)),"Exact single-student topology/parameter count differs")
    require(metrics.get("status")=="complete" and metrics.get("epochs_completed")==160 and last.get("epoch")==160
        and last.get("formal_training_state") is True and last.get("preflight_state") is False and last.get("updates")==50080
        and last.get("config")==metrics.get("config")==config and last.get("history")==history,"Complete actual160 training/state required")
    totals=trainer.validate_history(history,config)
    require(all(type(obj.get(k)) is int and obj[k]==value for obj in (metrics,last) for k,value in totals.items()),"Full source/student/teacher accounting differs")
    require(last.get("input_config_sha256")==hashlib.sha256(raw).hexdigest() and all(obj.get("teacher_reference_sha256")==trainer.json_sha(teacher_reference)
        for obj in (checkpoint,metrics,last)),"Actual raw config and teacher-reference bindings differ")
    require(metrics.get("training_examples")==40000 and metrics.get("validation_examples")==5000 and provenance.get("data_signature")==checkpoint.get("data_signature")==last.get("data_signature")==trainer.DATA,
        "Exact official independent split required")
    require(provenance.get("normalization")=={"mean":list(m.core.MEAN),"std":list(m.core.STD)},"Original normalization differs")
    (epoch,variant),best=trainer.best_selection(history)
    require((checkpoint.get("epoch"),checkpoint.get("variant"))==(epoch,variant) and
        (checkpoint.get("validation_accuracy"),-checkpoint.get("validation_loss",float("inf")))==best==
        (metrics.get("best_validation_accuracy"),-metrics.get("best_validation_loss",float("inf")))==
        (last.get("best_accuracy"),-last.get("best_loss",float("inf"))) and
        metrics.get("elapsed_seconds")==last.get("elapsed_seconds")==history[-1]["elapsed_seconds"],"Actual selected model/EMA and final elapsed state differ")
    trainer.validate_saved_state(last,config)
    with m.kd.isolated_teacher_rng():reference=m.build_model(model_config)
    require(m.count_parameters(reference)==m.count_parameters(reference,True)==trainer.PARAMETERS,"Actual reconstructed parameter count differs")
    state=checkpoint.get("model",{});expected_state=reference.state_dict()
    require(set(state)==set(expected_state) and not any(key in checkpoint for key in ("teacher","teacher_model","teacher_weights")),"Selected artifact must contain only student tensors")
    for name,value in expected_state.items():
        actual=state[name]
        require(isinstance(actual,t.Tensor) and actual.shape==value.shape and actual.dtype==value.dtype and bool(t.isfinite(actual).all().item()),"Selected student shape/dtype/finite state differs")
        if name.endswith("num_batches_tracked"):require(actual.item()==epoch*313,"Selected online/EMA BN counter differs")
    return {"schema":"research_scratch_kd_integrity_v1","training":expected,**totals,"role":role,
        "actual_selected_last_optimizer_parent_RNG_tensors_audited":True,"actual_teacher_lineage_audited":True,
        "teacher_weights_embedded":False,"confirmation_evaluated":False,"hidden_images_loaded":False}


def reference_screen(reports,reference,signature,context,*,incumbent=False):
    """Same policy/context/pixels. Only incumbent gets the preregistered floor."""
    m=trainer._deps();from research import benchmark as core
    require(reference.get("schema")==core.SCHEMA and reference.get("status")=="complete" and reference.get("examples")==5000
        and reference.get("training_examples")==40000 and reference.get("bn_recalibrated") is False and reference.get("data_signature")==signature
        and reference.get("evaluation_context")==context and reference.get("batch_size")==128 and set(reference.get("policies",{}))==set(core.POLICIES),"Complete matched reference context required")
    for policy in core.POLICIES:
        require(set(reference["policies"][policy].get("cases",{}))==set(core.CASES),"All reference cases required")
        for case in core.CASES:
            a=reports[policy]["cases"][case];b=reference["policies"][policy]["cases"][case]
            require(type(a.get("examples")) is int and type(b.get("examples")) is int and a["examples"]==b["examples"]==5000 and a.get("input_pixel_sha256")==b.get("input_pixel_sha256")
                and all(type(value.get("accuracy")) in (int,float) and math.isfinite(value["accuracy"]) and 0<=value["accuracy"]<=1 and
                    type(value.get("loss")) in (int,float) and math.isfinite(value["loss"]) and value["loss"]>=0 for value in (a,b)),"All18 reference pixel/count/finite bindings required")
        for row in (reports[policy],reference["policies"][policy]):
            require(row.get("clean_accuracy")==row["cases"]["clean"]["accuracy"] and row.get("clean_loss")==row["cases"]["clean"]["loss"] and
                type(row.get("stress_accuracy")) in (int,float) and math.isfinite(row["stress_accuracy"]) and
                math.isclose(row["stress_accuracy"],sum(row["cases"][case]["accuracy"] for case in core.CASES[1:])/5,rel_tol=0,abs_tol=1e-12),"Reference aggregate differs from all cases")
    report,base=reports[POLICY],reference["policies"][POLICY];gate=core.promotion_gate(report,base)
    changes={case:report["cases"][case]["accuracy"]-base["cases"][case]["accuracy"] for case in core.CASES[1:]}
    floor=all(delta>=-.01-1e-12 for delta in changes.values()) if incumbent else None
    return {"policy":POLICY,"mean_gate":gate,"clean_nll_delta":report["clean_loss"]-base["clean_loss"],"case_accuracy_deltas":changes,
        "noise_accuracy_delta":changes["noise_004"],"per_case_noninferiority_passed":floor,"passed":gate["passed"] and (floor if incumbent else True),
        "comparator":"fixed_incumbent" if incumbent else "fresh_alpha0_control","control_floor_added":False,"automatic_delivery_authorization":False}


def run_benchmark(checkpoint_path,output,baseline_path,data_dir,*,control_benchmark=None,control_checkpoint=None):
    """Actual CUDA-only full18 scoring after typed state and teacher audit."""
    m=trainer._deps();t=m.t
    from research import benchmark as core
    from research import incumbent_compare as incumbent
    checkpoint_path,output,baseline_path,data_dir=map(Path,(checkpoint_path,output,baseline_path,data_dir))
    require(not output.exists() and t.cuda.is_available() and t.cuda.is_bf16_supported(including_emulation=False),"Fresh native CUDA benchmark required")
    require(trainer.sha(ROOT/PROTOCOL)==PROTOCOL_SHA and trainer.sha(baseline_path)==BASELINE_SHA,"Frozen baseline/prospective protocol differs")
    folder=checkpoint_path.parent;reference=trainer.read(folder/"teacher_reference.json");trainer.validate_teacher_reference(reference)
    paths=[checkpoint_path,*(folder/name for name in ("last.pt","metrics.json","history.json","provenance.json","environment.json","input_config.json","config.json","teacher_reference.json")),
        baseline_path,Path(reference["checkpoint_path"]),Path(reference["benchmark_path"]),Path(reference["frozen_manifest_path"]),
        *(Path(name) for name in reference["evidence_sha256"]),ROOT/PROTOCOL,Path(__file__),*(ROOT/name for name in trainer.SOURCE_KEYS),
        *(ROOT/name for name in ("train.py","model.py","data.py","runtime.py")),
        *(data_dir/name for name in ("training_manifest.json","official_train_images.npy","official_train_labels.npy","official_train_ids.npy","official_split_indices.npz"))]
    if control_benchmark is not None:
        require(control_checkpoint is not None,"Actual complete alpha0 control checkpoint required")
        control_benchmark,control_checkpoint=Path(control_benchmark),Path(control_checkpoint)
        paths+=[control_benchmark,control_checkpoint,*(control_checkpoint.parent/name for name in ("last.pt","metrics.json","history.json","provenance.json","environment.json","input_config.json","config.json","teacher_reference.json"))]
    hashes={str(path.resolve()):trainer.sha(path) for path in paths};started=time.perf_counter();device=t.device("cuda")
    m.runtime.seed_all(2026)
    checkpoint=t.load(checkpoint_path,map_location="cpu",weights_only=False);last=t.load(folder/"last.pt",map_location="cpu",weights_only=False)
    metrics,history,provenance,environment=[trainer.read(folder/name) for name in ("metrics.json","history.json","provenance.json","environment.json")]
    require(trainer.read(folder/"config.json")==checkpoint["config"],"Actual effective config file differs")
    integrity=validate_training(checkpoint,metrics,history,provenance,environment,last,(folder/"input_config.json").read_bytes(),reference);del last
    require(integrity["training"]["precision"]=="bf16" and environment.get("device")=="cuda" and environment.get("amp_dtype")=="torch.bfloat16","CPU fixture states cannot become a native benchmark")
    images,labels,train_ids,val_ids=m.load_training_arrays(data_dir);signature=trainer.DATA;core_source=m.core.code_provenance()["source_sha256"]
    actual_manifest=trainer.read(data_dir/"training_manifest.json")
    require(actual_manifest["split"]["indices_sha256"]==signature["split_sha256"] and m.core.training_source(data_dir)["archive_sha256"]==signature["archive_sha256"],"Actual official data signature differs")
    context={"device_type":"cuda","amp_dtype":str(core.benchmark_dtype(device)),"torch_version":t.__version__,"training_core_sha256":core_source}
    require(context==incumbent.CONTEXT,"Exact original native-v3 context required")
    core.validate_checkpoint(checkpoint,train_ids,val_ids,signature,metrics,core_source,history)
    old_teacher,teacher_doc=m.paired.load_experiment(Path(reference["checkpoint_path"]),Path(reference["benchmark_path"]),signature,context,core_source,train_ids,val_ids);del old_teacher
    model=m.build_model(checkpoint["model_config"]).to(device=device,memory_format=t.channels_last);model.load_state_dict(checkpoint["model"]);model.eval()
    val_x=m.core.tensor_images(images[val_ids],device);val_y=t.as_tensor(labels[val_ids],device=device);del images,labels
    reports={policy:core.score(model,val_x,val_y,device,policy,128) for policy in core.POLICIES}
    baseline=trainer.read(baseline_path);core.validate_baseline(baseline,reports,signature,128,context,trainer.sha(ROOT/"research/benchmark.py"))
    gates={policy:core.promotion_gate(row,baseline["policies"]["flip"]) for policy,row in reports.items()}
    prospective=reference_screen(reports,teacher_doc,signature,context,incumbent=True)
    control_screen=None
    if checkpoint["config"]["kd_alpha"]>0:
        require(control_benchmark is not None,"KD benchmark requires its newly completed matched control")
        control_folder=control_checkpoint.parent
        c=t.load(control_checkpoint,map_location="cpu",weights_only=False);clast=t.load(control_folder/"last.pt",map_location="cpu",weights_only=False)
        cm,ch,cp,ce=[trainer.read(control_folder/name) for name in ("metrics.json","history.json","provenance.json","environment.json")]
        cref=trainer.read(control_folder/"teacher_reference.json")
        control_integrity=validate_training(c,cm,ch,cp,ce,clast,(control_folder/"input_config.json").read_bytes(),cref);del clast
        require(c["config"]==trainer.fixed_config(0.,2026) and cref==reference and trainer.sha(control_checkpoint)!=trainer.sha(checkpoint_path),"Fresh exact alpha0 control required")
        core.validate_checkpoint(c,train_ids,val_ids,signature,cm,core_source,ch)
        control_doc=trainer.read(control_benchmark)
        require(control_doc.get("checkpoint_sha256")==trainer.sha(control_checkpoint) and control_doc.get("training_metrics_sha256")==trainer.sha(control_folder/"metrics.json")
            and control_doc.get("training_history_sha256")==trainer.sha(control_folder/"history.json") and control_doc.get("scratch_kd_benchmark_source_sha256")==trainer.sha(__file__)
            and control_doc.get("research_training_integrity")==control_integrity,"Actual typed fresh-control benchmark link differs")
        require(isinstance(control_doc.get("evidence_sha256"),dict) and bool(control_doc["evidence_sha256"]),"Actual complete control benchmark evidence required")
        for path,digest in control_doc["evidence_sha256"].items():require(trainer.sha(path)==digest,"Actual control benchmark evidence bytes differ")
        control_screen=reference_screen(reports,control_doc,signature,context,incumbent=False)
    require(all(trainer.sha(path)==digest for path,digest in hashes.items()),"KD/control/source/data/teacher evidence changed during benchmark")
    passed=gates[POLICY]["passed"] and prospective["passed"] and (control_screen["passed"] if control_screen is not None else False)
    result={"schema":core.SCHEMA,"status":"complete","examples":5000,"training_examples":40000,"parameter_count":trainer.PARAMETERS,
        "checkpoint_sha256":trainer.sha(checkpoint_path),"checkpoint_epoch":checkpoint["epoch"],"checkpoint_variant":checkpoint["variant"],"model_config":checkpoint["model_config"],"config":checkpoint["config"],
        "data_signature":signature,"stress_cases":list(core.CASES),"batch_size":128,"evaluation_context":context,"bn_recalibrated":False,"policies":reports,"promotion":gates,
        "fixed_primary_policy":POLICY,"prospective_incumbent":prospective,"matched_control_screen":control_screen,"point_and_case_gates_passed":passed,
        "baseline_sha256":trainer.sha(baseline_path),"fixed_incumbent_benchmark_sha256":trainer.TEACHER_BENCHMARK,
        "training_metrics_sha256":trainer.sha(folder/"metrics.json"),"training_history_sha256":trainer.sha(folder/"history.json"),"source":m.core.code_provenance(),
        "benchmark_source_sha256":trainer.sha(ROOT/"research/benchmark.py"),"scratch_kd_benchmark_source_sha256":trainer.sha(__file__),"research_training_integrity":integrity,
        "evidence_sha256":hashes,"teacher_reference_sha256":trainer.json_sha(reference),"control_benchmark_sha256":trainer.sha(control_benchmark) if control_benchmark else None,
        "elapsed_seconds":time.perf_counter()-started,"confirmation_evaluated":False,"hidden_images_loaded":False,"external_data_loaded":False,
        "paired_comparison_performed":False,"delivery_authorized":False,"prediction_authorized":False}
    m.runtime.write_json(output,result);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args(argv);parser.print_help()


if __name__=="__main__":main()
