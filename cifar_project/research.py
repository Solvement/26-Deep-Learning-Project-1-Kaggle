"""Bounded Kaggle experiment execution, evidence collection and screening decisions.

Methods/hypotheses are curated in the research queue. This module does not call
an LLM, change code, publish notebooks, use the confirmation set, or submit predictions.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import uuid

from .model import CifarResNet
from .train import Config
from .utils import write_json

ALLOWED_OVERRIDES = {"base_width", "blocks_per_stage", "learning_rate", "weight_decay", "label_smoothing"}


def load_plan(path):
    queue = json.loads(Path(path).read_text())
    baseline = json.loads(Path(queue["base_config"]).read_text())
    config = Config(**baseline)
    budget = queue["budget"]
    if any(type(budget[k]) is not int or budget[k] <= 0 for k in ["max_new_runs", "per_run_seconds", "total_reserved_seconds"]):
        raise ValueError("Research budgets must be positive integer limits")
    if not 0 <= budget["prior_reserved_seconds"] < budget["total_reserved_seconds"]:
        raise ValueError("Invalid prior resource reservation")
    if type(queue["screening_epochs"]) is not int or not config.warmup_epochs + 3 <= queue["screening_epochs"] <= config.epochs:
        raise ValueError("Screening must include at least three post-warmup epochs")
    ids, experiments = set(), []
    for candidate in queue["candidates"]:
        identity = candidate["id"]
        if not identity or not all(c.islower() or c.isdigit() or c == "-" for c in identity) or identity in ids:
            raise ValueError("Experiment IDs must be unique lowercase slugs")
        ids.add(identity)
        if not set(candidate["overrides"]).issubset(ALLOWED_OVERRIDES):
            raise ValueError("Candidates may not change frozen data splits, seeds, epoch schedules or resource settings")
        if len(candidate["overrides"]) > 1:
            raise ValueError("First-round comparisons change at most one factor")
        if not candidate.get("hypothesis") or not candidate.get("sources"):
            raise ValueError("Every candidate needs a hypothesis and sources")
        resolved = Config(**{**baseline, **candidate["overrides"]})
        parameters = CifarResNet(resolved.base_width, resolved.blocks_per_stage).parameter_count
        experiments.append({**candidate, "config": asdict(resolved), "parameters": parameters})
    if not experiments or experiments[0]["overrides"]:
        raise ValueError("The first formal experiment must be the frozen baseline")
    plan = {**queue, "experiments": experiments}
    plan["plan_sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    return plan


def reservation_available(plan, state):
    budget = plan["budget"]
    used = budget["prior_reserved_seconds"] + sum(r["reserved_seconds"] for r in state["runs"])
    return len(state["runs"]) < budget["max_new_runs"] and used + budget["per_run_seconds"] <= budget["total_reserved_seconds"]


def verify_result(summary, run, history, experiment, plan, baseline_run=None):
    if summary["smoke_test"] or run["smoke_test"] or any(run["batch_limits"].values()):
        raise ValueError("Smoke or partial-batch results cannot enter formal research rankings")
    if summary["completed_epochs"] != plan["screening_epochs"] or len(history) != plan["screening_epochs"]:
        raise ValueError("Incomplete fidelity budget")
    if run["config"] != experiment["config"] or run["trainable_parameters"] != experiment["parameters"]:
        raise ValueError("Executed configuration/parameter count differs from the planned experiment")
    if [run[k] for k in ["training_examples", "validation_examples", "confirmation_examples"]] != [40000, 5000, 5000]:
        raise ValueError("Research results require the complete frozen 40k/5k/5k dataset")
    if run["provenance"]["device"] != "cuda":
        raise ValueError("Formal GPU experiment unexpectedly ran on CPU")
    if baseline_run:
        for key in ["dataset_sha256", "split_sha256"]:
            if run[key] != baseline_run[key]:
                raise ValueError(f"Incomparable experiments: {key} differs")
        for key in ["source_sha256", "torch", "gpu"]:
            if run["provenance"][key] != baseline_run["provenance"][key]:
                raise ValueError(f"Incomparable experiment environment: {key} differs")
    if [row["epoch"] for row in history] != list(range(1, plan["screening_epochs"] + 1)):
        raise ValueError("Metrics contain missing or duplicate epochs")
    for row in history:
        for partition, count in [("train", 40000), ("validation", 5000)]:
            if row[partition]["examples"] != count or not 0 <= row[partition]["accuracy"] <= 1 or not math.isfinite(row[partition]["loss"]):
                raise ValueError("Invalid metrics or example count")
        if not math.isfinite(row["seconds"]) or row["seconds"] <= 0:
            raise ValueError("Invalid runtime measurement")
    tail = history[-3:]
    return {"last3_development_accuracy": statistics.mean(r["validation"]["accuracy"] for r in tail),
            "last3_accuracy_range": max(r["validation"]["accuracy"] for r in tail) - min(r["validation"]["accuracy"] for r in tail),
            "median_epoch_seconds": statistics.median(r["seconds"] for r in tail),
            "best_development_accuracy": summary["validation_accuracy"],
            "official_test_accuracy": None, "confirmation_evaluated": False}


def decisions(state):
    completed = [r for r in state["runs"] if r["status"] == "complete"]
    if not completed:
        return {"stage": "screening", "next": "Await a valid formal baseline; engineering pilot results are excluded."}
    baseline = completed[0]["result"]
    comparisons = []
    for record in completed[1:]:
        result = record["result"]
        delta = result["last3_development_accuracy"] - baseline["last3_development_accuracy"]
        ratio = result["median_epoch_seconds"] / baseline["median_epoch_seconds"]
        outcome = "promising: extend fidelity and replicate with another seed" if delta >= 0.005 and ratio <= 2.5 else "inconclusive or costlier at this fidelity; retain baseline and review curves"
        comparisons.append({"candidate": record["id"], "development_accuracy_delta": delta,
                            "epoch_cost_ratio": ratio, "decision": outcome})
    return {"stage": "screening", "comparisons": comparisons,
            "final_model_selected": False, "confirmation_evaluated": False,
            "next": "Review curves and costs, then explicitly allocate a larger-fidelity/seed round. No automatic full training or Kaggle submission."}


def make_notebook(template, experiment, plan):
    notebook = json.loads(Path(template).read_text())
    all_code = "\n".join("".join(c["source"]) if isinstance(c["source"], list) else c["source"]
                         for c in notebook["cells"] if c["cell_type"] == "code")
    for marker in ['config = json.loads((PROJECT_ROOT / "configs/baseline.json").read_text())',
                   'os.environ.get("CIFAR_RUN_MODE", "inspect")', '["--stop-after-epoch", "5"]']:
        if all_code.count(marker) != 1:
            raise ValueError("Notebook template changed; inspect it before spending GPU resources")
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        source = source.replace('os.environ.get("CIFAR_RUN_MODE", "inspect")', 'os.environ.get("CIFAR_RUN_MODE", "pilot")')
        marker = 'config = json.loads((PROJECT_ROOT / "configs/baseline.json").read_text())'
        source = source.replace(marker, marker + "\nconfig.update(" + repr(experiment["config"]) + ')\n(PROJECT_ROOT / "configs/baseline.json").write_text(json.dumps(config))')
        source = source.replace('["--stop-after-epoch", "5"]', '["--stop-after-epoch", "' + str(plan["screening_epochs"]) + '"]')
        source = source.replace('"outputs/baseline-seed42"', '"outputs/' + experiment["id"] + '"')
        cell["source"] = source
        cell["outputs"] = []
        cell["execution_count"] = None
        compile(source, "research-notebook-cell", "exec")
    return notebook


def _execute(plan, folder, template):
    from kaggle import api
    folder.mkdir(parents=True, exist_ok=True)
    state_path = folder / "campaign.json"
    template_sha = hashlib.sha256(Path(template).read_bytes()).hexdigest()
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "plan_sha256": plan["plan_sha256"], "notebook_sha256": template_sha, "started_at": time.time(), "runs": []}
    if state["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("Existing campaign has a different plan; preserve it and start a new campaign directory")
    if state["notebook_sha256"] != template_sha:
        raise ValueError("Notebook/source changed during this campaign; preserve results and start a new round")
    frozen_template = folder / "frozen-template.ipynb"
    if not frozen_template.exists():
        frozen_template.write_bytes(Path(template).read_bytes())
    if hashlib.sha256(frozen_template.read_bytes()).hexdigest() != template_sha:
        raise ValueError("Frozen notebook was modified")
    quota = api.quota_view().gpu_quota
    required = plan["budget"]["max_new_runs"] * plan["budget"]["per_run_seconds"]
    if quota is None or (quota.total_time_allowed - quota.time_used).total_seconds() < required:
        raise RuntimeError("Insufficient reported GPU quota for this bounded round; do not fall back to paid resources")
    username = api.get_config_value("username")
    if not username:
        raise RuntimeError("Kaggle did not provide the authenticated username")
    baseline_run = None
    for experiment in plan["experiments"]:
        record = next((r for r in state["runs"] if r["id"] == experiment["id"]), None)
        if record and record["status"] == "complete":
            baseline_run = baseline_run or record["run_manifest"]
            continue
        if record and record["status"] != "running":
            raise RuntimeError("An earlier failure/uncertain submission requires diagnosis; refusing to repeat it")
        work = folder / experiment["id"]
        work.mkdir(exist_ok=True)
        if record is None:
            if not reservation_available(plan, state):
                print("Research stopped at the configured run/time budget", flush=True)
                break
            if time.time() - state["started_at"] >= plan["budget"]["total_reserved_seconds"]:
                print("Research stopped at the wall-clock budget", flush=True)
                break
            slug = "cifar-research-" + experiment["id"] + "-" + uuid.uuid4().hex[:6]
            ref = username + "/" + slug
            prepared = make_notebook(frozen_template, experiment, plan)
            record = {"id": experiment["id"], "ref": ref, "status": "submitting",
                      "reserved_seconds": plan["budget"]["per_run_seconds"], "submitted_at": time.time()}
            state["runs"].append(record)
            write_json(state_path, state)  # Persist before submission: ambiguous failures must not create duplicates.
            metadata = {"id": ref, "title": slug, "code_file": "experiment.ipynb", "language": "python",
                        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": False,
                        "competition_sources": [plan["competition"]], "dataset_sources": [], "kernel_sources": [], "model_sources": []}
            write_json(work / "kernel-metadata.json", metadata)
            write_json(work / "experiment.ipynb", prepared)
            response = json.loads(str(api.kernels_push(str(work), timeout=str(record["reserved_seconds"]))))
            if response.get("error") or any(response.get(k) for k in ["invalidCompetitionSources", "invalidDatasetSources", "invalidKernelSources", "invalidModelSources"]):
                record["status"] = "submission_failed"
                write_json(state_path, state)
                raise RuntimeError("Kaggle rejected the experiment or its inputs; inspect the prepared notebook")
            record.update({"ref": response["ref"].removeprefix("/code/"), "status": "running", "url": response["url"]})
            write_json(state_path, state)
            print("Submitted private experiment:", record["url"], flush=True)
        while True:
            status = json.loads(str(api.kernels_status(record["ref"])))
            print(experiment["id"], status["status"], flush=True)
            if status["status"] == "COMPLETE":
                break
            if status["status"] in {"ERROR", "CANCELLED", "CANCELED", "FAILED"}:
                record.update({"status": "failed", "failure": status.get("failureMessage")})
                write_json(state_path, state)
                raise RuntimeError("Experiment failed; stop the round and diagnose before another GPU run")
            if time.time() - record["submitted_at"] > record["reserved_seconds"] + 600:
                raise RuntimeError("Server run exceeded timeout plus queue grace; inspect the existing job, do not resubmit")
            time.sleep(30)
        api.kernels_output(record["ref"], str(work / "output"), file_pattern=r"(summary\.json|metrics\.jsonl|run\.json)$", quiet=True, page_size=200)
        def artifact(name):
            matches = list((work / "output").rglob(name))
            if len(matches) != 1:
                raise RuntimeError(f"Expected exactly one {name} artifact from the completed run")
            return matches[0]
        summary = json.loads(artifact("summary.json").read_text())
        run = json.loads(artifact("run.json").read_text())
        history = [json.loads(line) for line in artifact("metrics.jsonl").read_text().splitlines()]
        try:
            record["result"] = verify_result(summary, run, history, experiment, plan, baseline_run)
        except Exception as exc:
            record.update({"status": "invalid_result", "validation_failure": str(exc)})
            write_json(state_path, state)
            raise
        record.update({"status": "complete", "finished_at": time.time(), "run_manifest": run})
        baseline_run = baseline_run or run
        write_json(state_path, state)
        write_json(folder / "decision.json", decisions(state))
        print(json.dumps(record["result"]), flush=True)
    write_json(folder / "decision.json", decisions(state))
    print(json.dumps(decisions(state), indent=2), flush=True)


def execute(plan, folder, template):
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / ".campaign.lock"
    with lock.open("x") as handle:
        handle.write(str(os.getpid()))
    try:
        return _execute(plan, folder, template)
    finally:
        lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "run"])
    parser.add_argument("--queue", type=Path, default=Path("research/queue.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/research-round1"))
    parser.add_argument("--notebook", type=Path, default=Path("notebooks/train.ipynb"))
    args = parser.parse_args(argv)
    plan = load_plan(args.queue)
    if args.command == "plan":
        print(json.dumps(plan, indent=2))
    else:
        execute(plan, args.output_dir, args.notebook)


if __name__ == "__main__":
    main()
