"""Monitor one existing Kaggle run, retain live metrics and collect its outputs.

This watcher reports stop recommendations. It does not guess a session ID, delete
a notebook, or claim that remote cancellation has been verified.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time


def metrics_from_log(path):
    epochs = {}
    if not path.is_file():
        return []
    for line in path.read_text(errors="replace").splitlines():
        try:
            record = json.loads(line)
            if isinstance(record, dict) and "epoch" in record and "train" in record and "validation" in record:
                epochs[record["epoch"]] = record
        except ValueError:
            continue
    return [epochs[k] for k in sorted(epochs)]


def assess(records):
    if not records:
        return {"decision": "await_metrics", "reason": "No new epoch measurements; absence of output is not proof of failure."}
    for record in records:
        for partition in ["train", "validation"]:
            if not math.isfinite(record[partition]["loss"]) or not math.isfinite(record[partition]["accuracy"]):
                return {"decision": "stop_recommended", "reason": "Non-finite loss or accuracy."}
    latest = records[-1]
    if latest["epoch"] >= 50 and len(records) >= 10:
        tail = records[-10:]
        if max(r["validation"]["accuracy"] for r in tail) < 0.60:
            return {"decision": "stop_recommended", "reason": "Sustained severe development failure after epoch 50."}
    if latest["epoch"] >= 100 and latest["learning_rate"] <= 0.02 and len(records) >= 25:
        previous = max((r["validation"]["accuracy"] for r in records[:-20]), default=0.8088)
        recent = max(r["validation"]["accuracy"] for r in records[-20:])
        if recent < previous + 0.001:
            return {"decision": "review_early_finish", "reason": "Late low-LR plateau; preserve and verify best checkpoint before stopping."}
    return {"decision": "continue", "reason": "No sustained failure criterion; single high-LR validation drops are insufficient."}


def finite_report(value):
    if isinstance(value, dict):
        return {key: finite_report(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_report(item) for item in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--maximum-seconds", type=int, default=7800)
    args = parser.parse_args()
    from kaggle import api
    args.state_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    failures = 0
    while time.monotonic() - started < args.maximum_seconds:
        records = metrics_from_log(args.log)
        health = assess(records)
        report = {"observed_at_utc": datetime.now(timezone.utc).isoformat(), "kernel": args.kernel,
                  "health": health, "remote_auto_cancel_verified": False,
                  "latest": records[-1] if records else None,
                  "best_observed_development_accuracy": max([0.8088] + [r["validation"]["accuracy"] for r in records])}
        try:
            status = json.loads(str(api.kernels_status(args.kernel)))
            report["server"] = status
            failures = 0
        except Exception as exc:
            failures += 1
            report["server"] = {"status": "UNKNOWN", "error_type": type(exc).__name__}
        temporary = args.state_dir / "monitor.json.tmp"
        temporary.write_text(json.dumps(finite_report(report), indent=2, allow_nan=False) + "\n")
        temporary.replace(args.state_dir / "monitor.json")
        print(json.dumps({"epoch": report["latest"]["epoch"] if report["latest"] else None,
                          "status": report["server"]["status"], "health": health}), flush=True)
        if report["server"]["status"] in {"COMPLETE", "ERROR", "FAILED", "CANCELLED", "CANCELED"}:
            api.kernels_output(args.kernel, str(args.state_dir / "output"), quiet=True, page_size=200)
            print("Terminal run outputs collected", flush=True)
            return
        if failures >= 5:
            raise RuntimeError("Repeated monitoring access failures; report UNKNOWN rather than infer training state")
        time.sleep(60)
    raise RuntimeError("Watcher time budget exhausted; query the existing remote run before restarting")


if __name__ == "__main__":
    main()
