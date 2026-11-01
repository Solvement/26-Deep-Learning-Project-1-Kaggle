"""Build an evidence-backed <=8-page report and an executed results notebook.

All results are read from local artifacts. Missing measurements remain pending.
Generated report/notebook outputs are intentionally kept outside Git.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent
REPOSITORY = "https://github.com/Solvement/26-Deep-Learning-Project-1-Kaggle/tree/feat/cifar10-resnet-submission"
COMPETITION = "https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl"
REFERENCES = [
    ("He, Zhang, Ren and Sun (2016). Deep Residual Learning for Image Recognition.", "https://arxiv.org/abs/1512.03385"),
    ("Zagoruyko and Komodakis (2016). Wide Residual Networks.", "https://arxiv.org/abs/1605.07146"),
    ("Zhang et al. (2018). mixup: Beyond Empirical Risk Minimization.", "https://arxiv.org/abs/1710.09412"),
    ("DeVries and Taylor (2017). Improved Regularization of Convolutional Neural Networks with Cutout.", "https://arxiv.org/abs/1708.04552"),
    ("Loshchilov and Hutter (2017). SGDR: Stochastic Gradient Descent with Warm Restarts. Cosine schedule reference; this project need not use restarts.", "https://arxiv.org/abs/1608.03983"),
    ("Cubuk, Zoph, Shlens and Le (2019). RandAugment: Practical automated data augmentation with a reduced search space.", "https://arxiv.org/abs/1909.13719"),
]
RUN_ORDER = ("baseline", "architecture", "regularized", "final_full")
RUN_NAMES = set(RUN_ORDER)
FOLLOWUP_REFERENCES = [
    ("Izmailov et al. (2018). Averaging Weights Leads to Wider Optima and Better Generalization.", "https://arxiv.org/abs/1803.05407"),
    ("Foret et al. (2021). Sharpness-Aware Minimization for Efficiently Improving Generalization.", "https://arxiv.org/abs/2010.01412"),
    ("Yun et al. (2019). CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features.", "https://arxiv.org/abs/1905.04899"),
    ("Qin et al. (2020). ResizeMix: Mixing Data with Preserved Object Information and True Labels.", "https://arxiv.org/abs/2012.11101"),
    ("Shanmugam et al. (2021). Better Aggregation in Test-Time Augmentation.", "https://arxiv.org/abs/2011.11156"),
    ("Hu, Shen and Sun (2018). Squeeze-and-Excitation Networks. Engineering-only candidate reference.", "https://arxiv.org/abs/1709.01507"),
]


def read_json(path: Path, default=None):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def first(mapping, keys, default=None):
    for key in keys:
        if isinstance(mapping, dict) and mapping.get(key) is not None:
            return mapping[key]
    return default


def number(value):
    if value is None or isinstance(value, bool) or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def accuracy(value):
    value = number(value)
    if value is None:
        return "Pending"
    return f"{100 * value if value <= 1 else value:.2f}%"


def val_accuracy(metrics):
    return number(first(metrics, ("best_val_accuracy", "best_validation_accuracy", "val_accuracy", "validation_accuracy", "best_val_acc")))


def public_accuracy(submission):
    return number(first(submission, ("public_score", "publicScore", "public_accuracy", "public_test_accuracy")))


def wilson_interval(measured_accuracy, examples):
    """Descriptive binomial interval only when an integer success count is known."""
    p = number(measured_accuracy)
    if p is None or not examples or not 0 <= p <= 1:
        return None
    successes = round(p * examples)
    if not math.isclose(successes, p * examples, abs_tol=1e-6):
        return None
    z = 1.959963984540054
    denominator = 1 + z*z/examples
    center = (p + z*z/(2*examples))/denominator
    margin = z * math.sqrt(p*(1-p)/examples + z*z/(4*examples*examples))/denominator
    return (center-margin, center+margin, successes, examples)


def history_rows(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return first(value, ("history", "epochs"), [])
    return []


def collect_evidence(results_dir: Path):
    runs = []
    if results_dir.is_dir():
        for folder in sorted(p for p in results_dir.iterdir() if p.is_dir()):
            if folder.name not in RUN_NAMES:
                continue
            history = history_rows(read_json(folder / "history.json", []))
            metrics = read_json(folder / "metrics.json", {})
            config = read_json(folder / "config.json", first(metrics, ("config", "configuration"), {}))
            if not config:
                config = read_json(folder / "run.json", {}).get("config", {})
            if history or metrics:
                runs.append({"name": folder.name, "folder": str(folder), "history": history,
                             "metrics": metrics, "config": config,
                             "evaluation": read_json(folder / "evaluation.json", {})})
    submission = read_json(results_dir / "kaggle_submission.json", {})
    confirmation = read_json(results_dir / "confirmation.json", {})
    selection = read_json(results_dir / "selection.json", {})
    stability = read_json(results_dir / "stability.json", {})
    exploratory = {"config": read_json(results_dir / "robust_full/config.json", {}),
                   "metrics": read_json(results_dir / "robust_full/metrics.json", {}),
                   "evaluation": read_json(results_dir / "robust_full/evaluation.json", {})}
    submissions = read_json(results_dir / "kaggle_submissions.json", [])
    if isinstance(submissions, dict):
        submissions = submissions.get("submissions", [])
    if not isinstance(submissions, list):
        raise ValueError("Kaggle benchmark receipts must be a list")
    runs.sort(key=lambda run: RUN_ORDER.index(run["name"]))
    return {"runs": runs, "submission": submission, "confirmation": confirmation,
            "selection": selection, "stability": stability, "submissions": submissions,
            "exploratory": exploratory,
            "repository": REPOSITORY,
            "team": "Saber", "author": "Kevin Yang", "netid": "ky2738"}


def followup_output_dir(requested: Path | None, enabled: bool):
    """Require an unused explicit destination for the optional research edition."""
    output_dir = requested if requested is not None else ROOT / "outputs"
    if enabled:
        if requested is None:
            raise ValueError("Research reports require an explicit new --output-dir")
        target = output_dir.resolve()
        frozen = (ROOT / "outputs" / "primary_delivery_20261031").resolve()
        if target == frozen or frozen in target.parents:
            raise ValueError("The frozen primary delivery directory cannot contain research outputs")
        if target.exists():
            raise ValueError("Research --output-dir must not already exist")
    return output_dir


def attach_followup(evidence, summary_path: Path, receipts_dir: Path):
    """Attach the helper's public projection without replacing primary evidence."""
    from scripts.research_report_evidence import load_followup
    evidence["followup"] = load_followup(summary_path, receipts_dir)
    return evidence


def followup_snapshot(followup):
    """Serialize only the evidence helper's public projection, never raw records."""
    payload = json.dumps(followup, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    return payload, hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_followup_snapshot(path: Path, followup):
    payload, digest = followup_snapshot(followup)
    path.write_bytes(payload.encode("utf-8"))
    return digest


def report_references(evidence):
    return REFERENCES + FOLLOWUP_REFERENCES if "followup" in evidence else REFERENCES


def delta_interval(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return "Not available"
    lower, upper = (number(part) for part in value)
    return f"[{100*lower:+.2f}, {100*upper:+.2f}] pp" if lower is not None and upper is not None else "Not available"


def repeat_label(row):
    labels = {"complete_passed": "Complete; passed", "complete_failed": "Complete; failed",
              "in_progress": "In progress", "not_verified": "Not verified", "not_promoted": "Not promoted"}
    label = labels.get(row.get("repeat_status"), "Not verified")
    completed, planned = row.get("repeat_epochs_completed"), row.get("repeat_planned_epochs")
    if row.get("repeat_status") == "in_progress" and completed is not None and planned is not None:
        label += f" ({completed}/{planned} epochs)"
    if row.get("repeat_clean_accuracy") is not None:
        label += f"; clean {accuracy(row['repeat_clean_accuracy'])}, stress {accuracy(row.get('repeat_stress_accuracy'))}"
    return label


def matched_receipt_row(followup, receipt):
    """Match a held-out receipt to one audited row, without changing either."""
    checkpoint, policy = receipt.get("checkpoint_sha256"), receipt.get("policy")
    if (followup.get("schema") != "research_report_followup_v1" or
            followup.get("baseline", {}).get("training_examples") != 40000 or
            type(receipt.get("training_examples")) is not int or receipt["training_examples"] != 40000 or
            not isinstance(checkpoint, str) or len(checkpoint) != 64 or
            any(c not in "0123456789abcdef" for c in checkpoint) or not isinstance(policy, str) or not policy):
        return None
    # The public projection's benchmark rows have verified 40k training scope.
    matches = [row for row in followup.get("rows", [])
               if row.get("checkpoint_sha256") == checkpoint and row.get("policy") == policy and
               type(row.get("training_examples", 40000)) is int and row.get("training_examples", 40000) == 40000]
    if len(matches) == 1 and isinstance(matches[0].get("label"), str) and matches[0]["label"].strip():
        return matches[0]
    return None


def receipt_label(followup, receipt):
    row = matched_receipt_row(followup, receipt)
    return row["label"] if row is not None else receipt["run"]


def scored_outcome_text(followup):
    """Describe only actual public receipt scores, including a genuine zero."""
    def valid_score(value):
        value = number(value)
        return value if value is not None and math.isfinite(value) and 0 <= value <= 1 else None
    scored = [(receipt, valid_score(receipt.get("public_score")))
              for receipt in followup.get("official_submissions", [])]
    scored = [(receipt, score) for receipt, score in scored if score is not None]
    if not scored:
        return ""
    highest = max(score for _, score in scored)
    winners = sorted({row["label"] for receipt, score in scored if score == highest
                      if (row := matched_receipt_row(followup, receipt)) is not None})
    statements = []
    if winners:
        statements.append(f"Best newly scored research result: {', '.join(winners)} at {accuracy(highest)} public accuracy.")
    previous = [valid_score(followup.get(key)) for key in ("original_primary_public", "original_best_public")]
    best = max([score for _, score in scored] + [score for score in previous if score is not None])
    sam = [score for receipt, score in scored
           if (row := matched_receipt_row(followup, receipt)) is not None and row.get("id") == "sam"]
    if sam and max(sam) < best:
        statements.append(f"SAM scored {accuracy(max(sam))}, below the best observed public score of {accuracy(best)}; "
                          "it did not improve the public best.")
    return " ".join(statements)


def confirmation_scope(evidence):
    confirmation = evidence.get("confirmation", {})
    run = confirmation.get("run", evidence.get("selection", {}).get("run"))
    measured = number(confirmation.get("accuracy"))
    if run == "regularized" and measured is not None:
        return (f"The original {accuracy(measured)} confirmation belongs only to regularized "
                "trained on 40,000 images.")
    return "The original confirmation result is unavailable or its recipe has not been verified."


def completed_benchmarks(evidence):
    return [record for record in evidence.get("submissions", [])
            if str(record.get("status", "")).lower() == "complete" and public_accuracy(record) is not None]


def exploratory_method(evidence):
    """Keep the public-feedback exploration separate from the frozen main study."""
    scored = any(record.get("run") == "robust_full" for record in completed_benchmarks(evidence))
    config = evidence.get("exploratory", {}).get("config", {})
    metrics = evidence.get("exploratory", {}).get("metrics", {})
    epochs = first(metrics, ("epochs_completed", "epochs"), config.get("epochs", 300)) if scored else config.get("epochs", 300)
    ema = config.get("ema_decay", .999)
    introduction = "The separately scored robust_full benchmark" if scored else "The planned robust_full exploration"
    original = next((run for run in evidence.get("runs", []) if run["name"] == "regularized"), None)
    receipt = next((record for record in completed_benchmarks(evidence) if record.get("run") == "regularized"), None)
    gap = ""
    if original and receipt:
        dev = number(checkpoint_details(original)["development_accuracy"])
        public = public_accuracy(receipt)
        if dev is not None and public is not None:
            gap = (f"The regularized candidate's development/public accuracies were {accuracy(dev)}/{accuracy(public)}; "
                   "the cause of this gap was not established. ")
    return (
        gap + f"{introduction}, specified after official public feedback, uses a scratch WRN-22-4 on all 50,000 "
        f"permitted training images: {epochs} SGD/cosine epochs, RandAugment N=2/M=9 [6], crop/flip, "
        f"Cutout and MixUp, the fixed final-epoch EMA checkpoint (decay {ema}), "
        "clean-training-only batch-normalization recalibration, "
        "and fixed reflect-pad-four five-crop-plus-flip logit averaging. The original 160-epoch study "
        "and its confirmation result remain unchanged. This bundled, public-feedback-driven extension "
        "has no held-out development score or confirmation re-evaluation and is not a causal ablation. "
        "The earlier confirmation accuracy belongs only to the selected 40,000-image candidate. "
        "Execution or scoring is claimed only when actual artifacts and official receipts support it.")


def choose_run(evidence, requested=None):
    runs = evidence["runs"]
    hint = requested or first(evidence["submission"], ("run", "run_name", "experiment"))
    if hint:
        matched = [run for run in runs if run["name"] == Path(str(hint)).name]
        if not matched:
            raise ValueError(f"Requested run {hint!r} has no history or metrics")
        return matched[0]
    scored = [run for run in runs if val_accuracy(run["metrics"]) is not None]
    if scored:
        return max(scored, key=lambda run: val_accuracy(run["metrics"]))
    return runs[-1] if runs else None


def architecture_config(run):
    config = run["config"] if run else {}
    nested = first(config, ("model", "model_config", "architecture"), {})
    nested = nested if isinstance(nested, dict) else {}
    return {"depth": config.get("depth", nested.get("depth", 22)),
            "widen_factor": config.get("widen_factor", nested.get("widen_factor", 4))}


def variant_label(value):
    return {"model": "raw", "ema": "EMA"}.get(value, str(value) if value is not None else "Pending")


def checkpoint_details(run):
    evaluation = run.get("evaluation", {})
    config, metrics = run["config"], run["metrics"]
    full = config.get("train_all", False)
    epoch = evaluation.get("epoch", metrics.get("epochs_completed") if full else None)
    variant = evaluation.get("variant", config.get("final_variant") if full else None)
    flip = evaluation.get("flip_tta", config.get("fixed_flip_tta") if full else None)
    return {"epoch": epoch, "variant": variant_label(variant),
            "inference": "flip average" if flip is True else "plain" if flip is False else "Pending",
            "plain_accuracy": evaluation.get("plain_validation_accuracy"),
            "flip_accuracy": evaluation.get("flip_validation_accuracy"),
            "development_accuracy": evaluation.get("validation_accuracy", val_accuracy(metrics)),
            "full_data": full}


def measured_interpretation(evidence):
    """Describe completed matched runs without claiming seed-level significance."""
    runs = {run["name"]: run for run in evidence["runs"]}
    findings = []
    comparisons = [
        ("baseline", "architecture", {"depth", "widen_factor"}, "Architecture configuration", "depth and width jointly"),
        ("architecture", "regularized", {"cutout", "mixup_alpha", "mixup_probability"}, "Regularization bundle", "Cutout and MixUp jointly"),
    ]
    for left_name, right_name, permitted, label, changed in comparisons:
        left, right = runs.get(left_name), runs.get(right_name)
        if not left or not right or any(run["metrics"].get("status") != "complete" for run in (left,right)):
            continue
        ld, rd = checkpoint_details(left), checkpoint_details(right)
        la, ra = number(ld["plain_accuracy"]), number(rd["plain_accuracy"])
        if la is None or ra is None:
            continue
        delta = 100 * (ra-la)
        differing = {key for key in set(left["config"]) | set(right["config"])
                     if left["config"].get(key) != right["config"].get(key)}
        left_split, right_split = left.get("evaluation", {}).get("data_signature"), right.get("evaluation", {}).get("data_signature")
        equal_split = bool(left_split and right_split) and left_split == right_split
        matched = differing <= permitted and equal_split and left["config"].get("epochs") == right["config"].get("epochs")
        sentence = (f"{label}: using plain inference for both checkpoints, {right_name} achieved "
                    f"{accuracy(ra)} versus {accuracy(la)} for {left_name} ({delta:+.2f} percentage points). ")
        if matched:
            sentence += (f"Both runs used the same {left['config'].get('epochs')}-epoch budget, split, seed, "
                         f"optimizer and selection protocol; the changed factors were {changed}. ")
        else:
            sentence += "Saved configurations or split records differ beyond the intended comparison; treat this as descriptive. "
        if label == "Regularization bundle":
            sentence += "This comparison does not separate the individual effects of Cutout and MixUp."
        else:
            sentence += "This measures the joint architecture configuration, not an isolated depth or width effect."
        findings.append(sentence)
    stability = evidence.get("stability", {})
    comparison = runs.get(stability.get("comparison_run", "regularized"))
    if stability.get("status") == "complete" and stability.get("used_for_selection") is False and comparison:
        original = number(checkpoint_details(comparison)["development_accuracy"])
        repeated = number(stability.get("validation_accuracy"))
        if original is not None and repeated is not None:
            findings.append(
                f"Seed stability check: the regularized recipe obtained {accuracy(original)} with seed "
                f"{comparison['config'].get('seed', 2026)} and {accuracy(repeated)} with seed "
                f"{stability.get('seed', 2027)} on the same fixed development partition "
                f"({100*(repeated-original):+.2f} percentage points). The repeat was excluded from recipe "
                "selection and confirmation. These two observations do not support a formal seed-variance "
                "estimate, confidence interval or significance claim.")
    return findings


def followup_plot_size(followup):
    return 10.0, max(3.4, .60 * len(followup["rows"]) + .7)


def make_followup_plot(followup, output_dir: Path):
    """Plot public aggregate deltas; absent paired intervals remain absent."""
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = followup["rows"]
    with plt.rc_context({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13,
                         "xtick.labelsize": 13, "ytick.labelsize": 13}):
        fig, axes = plt.subplots(1, 2, figsize=followup_plot_size(followup), sharey=True,
                                 layout="constrained")
        labels = []
        for row in rows:
            label = textwrap.fill(str(row["label"]), width=24)
            if row.get("gate_passed") is False:
                label += "\n[gate failed]"
            labels.append(label)
        for axis, metric, title in zip(axes, ("clean", "stress"), ("Clean accuracy", "Synthetic stress mean")):
            for y, row in enumerate(rows):
                delta = number(row.get(metric + "_delta"))
                if delta is None:
                    continue
                failed = row.get("gate_passed") is False
                color = "#ae443c" if failed else "#167e75"
                interval = row.get(metric + "_ci95")
                if interval is not None:
                    lower, upper = interval
                    axis.hlines(y, 100*lower, 100*upper, color=color, linewidth=2.1)
                    axis.vlines([100*lower, 100*upper], y-.07, y+.07, color=color, linewidth=1.2)
                axis.plot(100*delta, y, marker="x" if failed else "o", markersize=7,
                          markeredgewidth=1.7, color=color, linestyle="none")
            axis.axvline(0, color="#6b7680", linewidth=.9, linestyle="--")
            axis.set(title=title, xlabel="Difference vs flip baseline (pp)")
            axis.set_yticks(range(len(rows)), labels)
            axis.set_ylim(max(len(rows), 1)-.5, -.5)
            axis.grid(axis="x", alpha=.18)
            axis.grid(axis="y", visible=False)
            axis.margins(x=.12)
            if not rows:
                axis.text(.5, .5, "No completed research benchmark", transform=axis.transAxes,
                          ha="center", va="center")
        fig.legend(handles=[Line2D([0], [0], color="#167e75", marker="o", linewidth=2,
                                  label="Paired 95% interval when available"),
                            Line2D([0], [0], color="#ae443c", marker="x", linestyle="none",
                                  label="Gate failed; point only without paired CI")],
                   loc="outside lower center", ncol=1, frameon=False, fontsize=13)
        target = output_dir / "research_conditional_deltas.png"
        fig.savefig(target, dpi=200)
        plt.close(fig)
    return target


def make_plots(evidence, output_dir: Path, selected=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold",
                         "figure.facecolor": "white", "axes.grid": True,
                         "grid.alpha": .15})
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.4), layout="constrained")
    specs = [(("train_loss", "loss"), "Training objective", False),
             (("val_accuracy", "validation_accuracy", "val_acc"), "Validation accuracy (%)", True),
             (("lr", "learning_rate"), "Learning rate", False),
             (("epoch_seconds", "seconds", "elapsed_seconds"), "Epoch or elapsed time (s)", False)]
    any_data = False
    for run in evidence["runs"]:
        rows = run["history"]
        for axis, (keys, title, percent) in zip(axes.flat, specs):
            points = [(first(row, ("epoch",), index + 1), number(first(row, keys)))
                      for index, row in enumerate(rows)]
            points = [(x, y) for x, y in points if y is not None]
            if points:
                xx, yy = zip(*points)
                if percent:
                    yy = [v * 100 if v <= 1 else v for v in yy]
                axis.plot(xx, yy, label=run["name"], linewidth=1.55)
                any_data = True
            if percent:
                ema_points = [(first(row, ("epoch",), index + 1), number(row.get("ema_validation_accuracy")))
                              for index, row in enumerate(rows)]
                ema_points = [(x,y) for x,y in ema_points if y is not None]
                if ema_points:
                    xx, yy = zip(*ema_points)
                    yy = [v * 100 if v <= 1 else v for v in yy]
                    axis.plot(xx, yy, linestyle="--", label=run["name"] + " EMA", linewidth=1.25)
    for axis, (_, title, _) in zip(axes.flat, specs):
        axis.set(title=title, xlabel="Epoch")
        if axis.lines:
            axis.legend(fontsize=7, frameon=False)
        else:
            axis.text(.5, .5, "No recorded measurements yet", transform=axis.transAxes,
                      ha="center", va="center", color="#637181")
    figure_path = output_dir / "learning_curves.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10.4, 1.7), layout="constrained")
    axis.set(xlim=(0, 12), ylim=(0, 1.8))
    axis.axis("off")
    selected = selected or choose_run(evidence)
    config = architecture_config(selected)
    width = config["widen_factor"]
    blocks = (config["depth"] - 4) // 6
    boxes = [("RGB input\n32 x 32", "#e8eef5"), ("3x3 stem\n16 channels", "#dceceb"),
             (f"Stage 1\n{16*width} channels\n{blocks} blocks", "#c6e3df"),
             (f"Stage 2\n{32*width} channels\n{blocks} blocks", "#9fd0c9"),
             (f"Stage 3\n{64*width} channels\n{blocks} blocks", "#70b7ad"),
             ("Pool + linear\n10 logits", "#dceceb")]
    from matplotlib.patches import FancyBboxPatch
    for index, (label, color) in enumerate(boxes):
        x = index * 2.0
        axis.add_patch(FancyBboxPatch((x + .03, .45), 1.7, .95,
                       boxstyle="round,pad=0.02,rounding_size=0.06", linewidth=0,
                       facecolor=color))
        axis.text(x + .88, .925, label, ha="center", va="center", fontsize=9, color="#173346")
        if index < len(boxes) - 1:
            axis.annotate("", xy=(x + 1.99, .925), xytext=(x + 1.77, .925),
                          arrowprops={"arrowstyle": "->", "color": "#627386"})
    diagram = output_dir / "architecture.png"
    fig.savefig(diagram, dpi=180, bbox_inches="tight")
    plt.close(fig)
    result = {"learning_curves": figure_path, "architecture": diagram, "has_measurements": any_data}
    confusion = evidence.get("confirmation", {}).get("confusion_matrix")
    if confusion is not None:
        import numpy as np
        counts = np.asarray(confusion)
        if counts.shape != (10,10) or np.any(counts < 0):
            raise ValueError("Confirmation confusion matrix must be nonnegative and 10 by 10")
        names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        fig, axis = plt.subplots(figsize=(7.0, 5.8), layout="constrained")
        proportions = counts / counts.sum(axis=1, keepdims=True).clip(min=1)
        image = axis.imshow(proportions, cmap="Blues", vmin=0, vmax=1)
        for row in range(10):
            for column in range(10):
                axis.text(column,row,str(counts[row,column]),ha="center",va="center",fontsize=7.5,
                          color="white" if proportions[row,column] > .55 else "#173346")
        axis.set(xticks=range(10),yticks=range(10),xticklabels=names,yticklabels=names,
                 xlabel="Predicted class",ylabel="True class",title="One-time confirmation errors")
        axis.tick_params(axis="x",labelrotation=40)
        axis.grid(False)
        fig.colorbar(image, ax=axis, shrink=.8, label="Fraction of true class")
        target = output_dir / "confirmation_confusion.png"
        fig.savefig(target,dpi=180)
        plt.close(fig)
        result["confirmation"] = target
    if "followup" in evidence:
        result["followup"] = make_followup_plot(evidence["followup"], output_dir)
    return result


def compact_config(config):
    preferred = ["epochs", "batch_size", "lr", "learning_rate", "weight_decay", "momentum",
                 "warmup_epochs", "mixup_alpha", "mixup_probability", "cutout", "cutout_size",
                 "ema_decay", "final_variant", "fixed_flip_tta", "train_all", "seed",
                 "label_smoothing", "precision", "device"]
    rows = [(key, config[key]) for key in preferred if key in config and
            (config.get("train_all") or key not in {"final_variant", "fixed_flip_tta"})]
    if not rows:
        rows = list(config.items())[:12]
    return rows[:16]


def create_pdf(evidence, selected, plots, output_dir):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, Flowable

    navy, teal, muted = colors.HexColor("#173346"), colors.HexColor("#167e75"), colors.HexColor("#596c7d")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleProject", fontName="Helvetica-Bold", fontSize=25,
                             leading=29, textColor=navy, spaceAfter=13))
    styles.add(ParagraphStyle(name="SubtitleProject", fontName="Helvetica", fontSize=11,
                             leading=15, textColor=muted, spaceAfter=10))
    styles["BodyText"].fontName = "Helvetica"
    styles["BodyText"].fontSize = 9.5
    styles["BodyText"].leading = 13.5
    styles["BodyText"].spaceAfter = 7
    styles["Heading1"].textColor = navy
    styles["Heading1"].fontSize = 17
    styles["Heading1"].leading = 21
    styles["Heading1"].spaceAfter = 12
    styles["Heading2"].textColor = teal
    styles["Heading2"].fontSize = 11.5
    styles["Heading2"].leading = 15
    styles["Heading2"].spaceBefore = 9
    styles["Heading2"].spaceAfter = 6
    styles.add(ParagraphStyle(name="SmallProject", parent=styles["BodyText"], fontSize=8, leading=10.5))
    styles.add(ParagraphStyle(name="CaptionProject", parent=styles["SmallProject"], textColor=muted))
    styles.add(ParagraphStyle(name="TableProject", parent=styles["SmallProject"], spaceAfter=0))

    story = []
    research_page_numbers = []
    class ResearchBoundary(Flowable):
        def draw(self):
            research_page_numbers.append(self.canv.getPageNumber())
    def p(text, style="BodyText"):
        story.append(Paragraph(text, styles[style]))
    def h(text):
        p(text, "Heading2")
    def table(rows, widths=None, compact=False):
        wrapped = [[Paragraph(escape(str(value)), styles["TableProject"]) for value in row] for row in rows]
        obj = Table(wrapped, colWidths=widths, hAlign="LEFT", repeatRows=1)
        obj.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e8f1f2")),
                                ("TEXTCOLOR", (0,0), (-1,0), navy),
                                ("VALIGN", (0,0), (-1,-1), "TOP"),
                                ("BOTTOMPADDING", (0,0), (-1,-1), 3 if compact else 7),
                                ("TOPPADDING", (0,0), (-1,-1), 3 if compact else 7),
                                ("LINEBELOW", (0,0), (-1,0), .6, colors.HexColor("#b9cccf")),
                                ("LINEBELOW", (0,1), (-1,-1), .3, colors.HexColor("#e0e8ed"))]))
        story.append(obj)
        story.append(Spacer(1, 8))

    metrics = selected["metrics"] if selected else {}
    config = selected["config"] if selected else {}
    confirmation = evidence.get("confirmation", {})
    recipe_name = first(evidence.get("selection", {}), ("run",), confirmation.get("run"))
    recipe = next((run for run in evidence["runs"] if run["name"] == recipe_name), None)
    if recipe is None:
        scored = [run for run in evidence["runs"] if val_accuracy(run["metrics"]) is not None]
        recipe = max(scored, key=lambda run: val_accuracy(run["metrics"])) if scored else None
    recipe_accuracy = first(evidence.get("selection", {}), ("validation_accuracy",),
                            recipe.get("evaluation", {}).get("validation_accuracy") if recipe else None)
    if recipe_accuracy is None and recipe:
        recipe_accuracy = val_accuracy(recipe["metrics"])
    parameter_count = first(metrics, ("trainable_parameters", "parameters", "parameter_count", "total_parameters"))
    if isinstance(parameter_count, dict):
        parameter_count = first(parameter_count, ("trainable", "total"))
    model_config = architecture_config(selected)
    if parameter_count is None and model_config.get("depth", 22) == 22 and model_config.get("widen_factor", 4) == 4:
        parameter_count = 4298970
    parameter_text = f"{int(parameter_count):,}" if number(parameter_count) is not None else "Pending"
    score = public_accuracy(evidence["submission"])
    followup = evidence.get("followup")
    references = report_references(evidence)
    benchmarks = completed_benchmarks(evidence)
    best_benchmark = max(benchmarks, key=public_accuracy) if benchmarks else None
    val = recipe_accuracy
    run_name = selected["name"] if selected else "No measured run yet"

    p("Efficient residual learning<br/>for CIFAR-10", "TitleProject")
    p("Deep Learning Project 1 | Team Saber<br/>Kevin Yang (ky2738)", "SubtitleProject")
    p(f'<b>Public code:</b> <link href="{REPOSITORY}" color="#167e75">{REPOSITORY}</link>')
    story.append(Spacer(1, 10))
    h("Overview: frozen original study (2026-10-31 UTC)" if followup else "Overview")
    p("This project designs a convolutional residual classifier for the course's CIFAR-10 competition. "
      "The implementation uses a CIFAR-sized WideResNet, starts from random weights, and enforces the "
      "five-million-parameter limit at model construction. The final prediction pipeline uses one model, "
      "with preprocessing and any inference averaging documented in the saved run artifacts.")
    architecture_name = f"WRN-{model_config['depth']}-{model_config['widen_factor']}"
    details = checkpoint_details(selected) if selected else {}
    checkpoint_text = (f"Epoch {details['epoch']}; {details['variant']}; {details['inference']}"
                       if details.get("epoch") is not None else "Pending")
    table([["Evidence", "Recorded value"], ["Selected local run", run_name],
           ["Selected architecture", architecture_name],
           ["Selected checkpoint / inference", checkpoint_text],
           ["Model trainable parameters", parameter_text], ["Selected recipe development accuracy", accuracy(val)],
           ["One-time confirmation accuracy", accuracy(confirmation.get("accuracy"))],
           ["Primary full-data public test accuracy", accuracy(score)],
           ["Highest original official public benchmark" if followup else "Highest completed official public benchmark",
            f"{accuracy(public_accuracy(best_benchmark))} ({best_benchmark.get('run', 'unknown')})" if best_benchmark else "Pending"],
           ["Private leaderboard test accuracy", "Not released / unavailable"]], [2.75*inch, 3.65*inch])
    h("Measured findings")
    if score is not None:
        p(f"The recorded Kaggle submission obtained <b>{accuracy(score)}</b> public leaderboard accuracy. "
          "This measures the public half of the hidden competition set; it does not establish the final private score.")
        interval = wilson_interval(score, 1000)
        if interval:
            low, high, successes, total = interval
            p(f"Public: {successes:,}/{total:,} correct; descriptive Wilson 95% interval "
              f"{accuracy(low)} to {accuracy(high)} (binomial approximation; private outcome unknown).", "SmallProject")
    else:
        p("Kaggle public test accuracy is pending. This draft reports only measurements available in saved artifacts "
          "and does not substitute a target accuracy for an observed result.")
    if val is not None:
        p(f"The selected recipe's best development accuracy is <b>{accuracy(val)}</b>. "
          "Development data choose the recipe and inference averaging; a separate confirmation split is evaluated once.")
    if number(confirmation.get("accuracy")) is not None:
        p(f"The frozen selected recipe achieved <b>{accuracy(confirmation['accuracy'])}</b> on "
          f"{confirmation.get('examples', 5000):,} confirmation images before refitting a fresh "
          "model on all 50,000 official training images.")
        interval = wilson_interval(confirmation['accuracy'], confirmation.get('examples', 5000))
        if interval:
            low, high, successes, total = interval
            p(f"Confirmation: {successes:,}/{total:,} correct; descriptive Wilson 95% binomial "
              f"interval {accuracy(low)} to {accuracy(high)}; this is not training-seed uncertainty.", "SmallProject")
    if best_benchmark:
        scope = "original " if followup else ""
        p(f"Highest observed {scope}public benchmark: <b>{escape(str(best_benchmark.get('run', 'unknown')))} "
          f"at {accuracy(public_accuracy(best_benchmark))}</b> among {len(benchmarks)} completed entries. "
          "The primary full-data result is separate; public ranking does not identify the private winner.", "SmallProject")

    story.append(PageBreak())
    p("Architecture and training method", "Heading1")
    h("The selected CIFAR residual network")
    width = model_config["widen_factor"]
    blocks = (model_config["depth"]-4)//6
    p(f"The selected {architecture_name} [1,2] uses a stride-one 3x3 CIFAR stem. "
      f"Its three residual stages use widths {16*width}, {32*width} and {64*width}, "
      f"with {blocks} two-convolution residual blocks per stage. "
      "Pre-activation batch normalization and ReLU precede convolutions; a learned 1x1 shortcut handles "
      "channel or spatial changes. Spatial downsampling occurs at the second and third stages. "
      "Global average pooling and a ten-class linear head complete the network.")
    story.append(Image(str(plots["architecture"]), width=6.35*inch, height=1.10*inch))
    p(f"Figure 1. Selected {architecture_name} data flow. Residual shortcuts operate inside each stage; widths and block counts are derived from the selected run's saved configuration.", "CaptionProject")
    p(f"The selected architecture contains <b>{parameter_text} parameters</b>, including normalization and the classifier. "
      "Every parameter is trainable. The constructor rejects a total parameter count over 5,000,000, a stronger "
      "check than counting only trainable parameters. No pretrained weights or external teacher models are used.")
    h("Official data and validation isolation")
    p("Training uses only the competition's official <b>train_images/</b> files and <b>train_labels.csv</b>. "
      "A private export of the official Kaggle mount supports local training while retaining file provenance. "
      "The deterministic stratified research split contains 40,000 optimization images, 5,000 development "
      "images and 5,000 confirmation images, using seed 2026. Each holdout has 500 images per class. "
      "Recipe and checkpoint selection use development only. The selected recipe is evaluated once on "
      "confirmation, then retrained from new random weights on all 50,000 images with the fixed recipe. "
      "The data manifest records all disjoint indices and fingerprints.")
    p("The 2,000 unlabeled competition images are loaded in sample-submission order only for inference. "
      "No official CIFAR test labels, outside datasets, manual test labels or external label lookups are used.")
    h("Optimization and regularization")
    p("The training implementation supports random crop and horizontal flip, Cutout [4], MixUp [3], "
      "SGD with Nesterov momentum, a warmup/cosine learning-rate schedule [5], and exponential moving "
      "averages of weights. The table below records actual enabled values from the selected run rather "
      "than assuming that every supported option was used. Mixed precision and channels-last execution "
      "improve GPU throughput without changing the model's parameter budget.")
    rows = compact_config(config)
    if rows:
        pairs = [[key, json.dumps(value, ensure_ascii=True) if isinstance(value, (dict,list)) else value] for key,value in rows]
        packed = [pairs[i] + (pairs[i+1] if i+1 < len(pairs) else ["", ""])
                  for i in range(0, len(pairs), 2)]
        table([["Configuration", "Saved value", "Configuration", "Saved value"]] + packed,
              [1.3*inch,1.9*inch,1.3*inch,1.9*inch], compact=True)
    else:
        p("No final training configuration is recorded yet.", "SmallProject")

    story.append(PageBreak())
    p("Experimental evidence", "Heading1")
    rows = [["Run", "Params", "Trained epochs", "Checkpoint", "Dev plain / flip", "Chosen inference"]]
    for run in evidence["runs"][:8]:
        m = run["metrics"]
        count = first(m, ("trainable_parameters", "parameters", "parameter_count", "total_parameters"), "Pending")
        if isinstance(count, dict):
            count = first(count, ("trainable", "total"), "Pending")
        if number(count) is not None:
            count = f"{int(count):,}"
        detail = checkpoint_details(run)
        checkpoint = (f"e{detail['epoch']} {detail['variant']}" if detail['epoch'] is not None else "Pending")
        dev_text = ("Not evaluated" if detail['full_data'] else
                    f"{accuracy(detail['plain_accuracy'])} / {accuracy(detail['flip_accuracy'])}")
        chosen = detail['inference']
        if not detail['full_data'] and detail['development_accuracy'] is not None:
            chosen += f" ({accuracy(detail['development_accuracy'])})"
        rows.append([run["name"], count, m.get("epochs_completed", len(run["history"])),
                     checkpoint, dev_text, chosen])
    if len(rows) == 1:
        rows.append(["Pending", "Pending", "0", "Pending", "Pending", "Pending"])
    table(rows, [1.05*inch,.75*inch,.65*inch,1.05*inch,1.4*inch,1.5*inch])
    story.append(Image(str(plots["learning_curves"]), width=6.35*inch, height=3.91*inch))
    p("Figure 2. Curves are drawn only from saved epoch-level measurements. Training objectives with augmentation or mixed labels are not directly comparable to clean validation cross-entropy. No missing measurements are interpolated.", "CaptionProject")
    p("Checkpoint identifies the selected epoch and raw/EMA weights. Plain/flip columns use that same checkpoint; the last column gives the development-selected inference rule. Epoch curves show training-time raw and EMA scores. The fixed full-data refit has no development score.", "SmallProject")
    if evidence.get("submissions"):
        h("Official Kaggle benchmark entries")
        benchmark_rows = [["Entry", "Train images", "Parameters", "Public score", "Submission ref", "Status"]]
        for receipt in evidence["submissions"]:
            count = receipt.get("parameter_count")
            training = receipt.get("training_examples")
            benchmark_rows.append([
                receipt.get("run", "Pending"),
                f"{int(training):,}" if number(training) is not None else "Pending",
                f"{int(count):,}" if number(count) is not None else "Pending",
                accuracy(public_accuracy(receipt)),
                receipt.get("submission_ref", "Pending"),
                receipt.get("status", "Pending")])
        table(benchmark_rows, [1.35*inch,.85*inch,.85*inch,.8*inch,1.65*inch,.9*inch], compact=True)
        p("The five-entry plan compares baseline, architecture, regularized, primary full-data and exploratory robust_full models. The seed repeat remains validation-only. Public comparison and exploratory work are separate from the frozen main study; private scores remain unavailable.", "SmallProject")
    story.append(PageBreak())
    p("Measured findings and limitations", "Heading1")
    findings = measured_interpretation(evidence)
    if findings:
        for finding in findings:
            p(escape(finding))
    else:
        p("Matched architecture and regularization comparisons are pending completion; no effect size is reported yet.")
    p("Development is reused for model selection; confirmation is checked once after the recipe is frozen. "
      "Each architecture comparison uses one fixed split and seed, so it does not establish statistical "
      "significance or isolate seed variation. The full-data refit has no held-out development metric; "
      "public and private leaderboard outcomes can differ.", "SmallProject")
    if evidence.get("submissions"):
        p("We allocated the five permitted daily submissions to official benchmark comparisons. The primary 160-epoch recipe and "
          "confirmation remain frozen; public feedback additionally motivated a separately labeled "
          "robust_full exploration. Selecting the highest of "
          "five public scores can favor sampling noise on the 1,000-image public subset. The benchmark "
          "table and primary full-data result are both reported. No confirmation re-evaluation or claim "
          "of a known private winner is made.", "SmallProject")
    if "confirmation" in plots:
        h("One-time confirmation error analysis")
        story.append(Image(str(plots["confirmation"]), width=5.0*inch, height=4.14*inch))
        p("Figure 3. Counts from the independent confirmation split; colors are normalized within the true class. This analysis is descriptive and is not used to revise the frozen recipe.", "CaptionProject")
        confusion = evidence["confirmation"]["confusion_matrix"]
        class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        per_class = [(confusion[i][i]/sum(row), class_names[i]) for i,row in enumerate(confusion) if sum(row)]
        weakest = sorted(per_class)[:2]
        if weakest:
            p("The lowest confirmation recalls were " + " and ".join(f"{name} ({accuracy(value)})" for value,name in weakest) +
              ". These are measured error patterns, not evidence that a later intervention would improve the hidden test score.", "SmallProject")

    if followup:
        story.append(PageBreak())
        story.append(ResearchBoundary())
        p("Follow-on research: development evidence", "Heading1")
        p("This follow-on study is exploratory: it reuses the official 5,000-image development set and "
          "five fixed synthetic stress transformations. These proxies are not CIFAR-C or hidden-test accuracy. "
          f"<b>{escape(confirmation_scope(evidence))}</b> "
          "Follow-on research does not reuse that confirmation. A full-50k refit has no independent development score.", "SmallProject")
        baseline = followup["baseline"]
        p(f"Fixed v3 baseline ({escape(str(baseline['policy']))}): clean {accuracy(baseline['clean_accuracy'])}, "
          f"NLL {number(baseline['clean_nll']):.4f}, stress {accuracy(baseline['stress_accuracy'])}. "
          "Stress is the mean accuracy over five transformations of each original image. "
          "The original study's recorded development results are preserved.", "SmallProject")
        p("Point gate: clean gain >= +0.10 percentage points with stress change >= -0.20, or "
          "stress gain >= +0.50 percentage points with clean change >= -0.20. SWA must also pass "
          "this same gate against its fixed WRN34-3 parent at the same policy. Paired evidence and "
          "a complete independent repeat are required before candidate delivery.", "SmallProject")
        research_rows = [["Recipe / parameters", "Epoch / weights / policy", "Clean / NLL / stress", "Gate", "Independent repeat / delivery"]]
        for row in followup["rows"]:
            params = f"{int(row['parameters']):,}" if number(row.get("parameters")) is not None else "Pending"
            loss = number(row.get("clean_nll"))
            measured = f"{accuracy(row.get('clean_accuracy'))} / {loss:.4f} / {accuracy(row.get('stress_accuracy'))}" if loss is not None else "Not complete"
            research_rows.append([
                f"{row['label']} ({params}); {row.get('epochs', 'Pending')} epochs; {row.get('precision', 'Pending')}",
                f"e{row.get('checkpoint_epoch', 'Pending')} {variant_label(row.get('checkpoint_variant'))}; {row['policy']}",
                measured, "Passed" if row.get("gate_passed") is True else "Failed" if row.get("gate_passed") is False else "Pending",
                repeat_label(row) + "; delivery: " + str(row["delivery_status"]).replace("_", " ")])
        if len(research_rows) == 1:
            research_rows.append(["No completed research benchmark", "Pending", "Pending", "Pending", "Pending"])
        table(research_rows, [1.65*inch,1.35*inch,1.2*inch,.55*inch,2.45*inch], compact=True)
        h("Conditional paired differences versus the v3 baseline")
        plot_width, plot_height = followup_plot_size(followup)
        story.append(Image(str(plots["followup"]), width=7.2*inch, height=7.2*inch*plot_height/plot_width))
        p("Figure 4. Conditional accuracy differences on reused/postselected development. Candidate policies are "
          "listed above; center80_shift20 weights center/flip logits 80% and shifted/flip logits 20%, versus "
          "the flip-average baseline. Available paired 95% intervals use 5,000 original-image clusters and "
          "10,000 bootstrap replicates, keeping all five stress views together. Failed gates are marked; "
          "SWA without paired evidence has a point only. Exact intervals remain in the public snapshot and notebook. "
          "These are not independent confirmation, seed-variance intervals or hidden-test guarantees.", "CaptionProject")

        story.append(PageBreak())
        p("Follow-on research: methods and official outcomes", "Heading1")
        p("WRN depth/width alternatives retain the same residual-CNN parameter limit [2]. "
          "The separately recorded methods include SWA weight averaging [7], SAM optimization [8], "
          "CutMix [9] and ResizeMix [10]. Inference averaging is evaluated as its own policy [11]. "
          "Engineering-only candidates, including SE [12] where listed, are kept separate from measured outcomes. "
          "The tables report actual completed benchmarks; engineering preflights, unfinished training, "
          "failed gates and incomplete repeats do not become completed experiments or scored submissions.", "SmallProject")
        p("Full-50k refits use a frozen qualified recipe and fresh scratch initialization. Their overlapping "
          "training/development images cannot yield an independent development score. Research models "
          "do not inherit the original confirmation result. Algorithm, architecture, precision and schedule "
          "changes are descriptive comparisons unless a matched control supports a narrower conclusion.", "SmallProject")
        table([["Official public evidence", "Accuracy"],
               ["Original primary full-data entry", accuracy(followup["original_primary_public"])],
               ["Best original entry", accuracy(followup["original_best_public"])],
               ["Best newly scored research entry", accuracy(followup["best_new_public"])],
               ["Best observed across original and research", accuracy(followup["best_observed_public"])]],
              [4.65*inch,1.75*inch], compact=True)
        receipts = followup["official_submissions"]
        receipt_rows = [["Research entry", "Official reference", "Public / private"]]
        for receipt in receipts[:8]:
            private_text = accuracy(receipt['private_score']) if receipt.get('private_score') is not None else "Unavailable"
            receipt_rows.append([receipt_label(followup, receipt), receipt["submission_ref"],
                                 f"{accuracy(receipt['public_score'])} / {private_text}"])
        if not receipts:
            receipt_rows.append(["No newly scored research submission", "None", "Pending / unavailable"])
        table(receipt_rows, [3.1*inch,1.4*inch,1.9*inch], compact=True)
        p("Recipe labels use a unique checkpoint/policy/40k-training match; raw run names remain in the snapshot.", "SmallProject")
        outcome = scored_outcome_text(followup)
        if outcome:
            p(escape(outcome), "SmallProject")
        if len(receipts) > 8:
            p(f"The table shows the first 8 of {len(receipts)} scored research receipts in reference order; "
              "the executed notebook and public snapshot retain every scored receipt. The best values use all receipts.", "SmallProject")
        p(f"Unscored receipt records: {followup['unscored_receipts_count']}. A completed development benchmark, "
          "verified CSV, upload intent or pending scoring is not a scored Kaggle entry. Private accuracy stays "
          "unavailable unless a corresponding official receipt supplies it.", "SmallProject")
        for limitation in followup["limitations"][:4]:
            p(escape(limitation), "SmallProject")
        if followup["engineering"]:
            p("Engineering status: " + escape(" ".join(followup["engineering"][:3])), "SmallProject")
        _, snapshot_sha = followup_snapshot(followup)
        p(f"Reporting snapshot: {escape(str(followup['recorded_at']))}. "
          f"Public research source: <link href=\"{escape(followup['repository'])}\" color=\"#167e75\">research branch</link>. "
          f"Source-summary SHA-256: {followup['source_summary_sha256']}. "
          f"Public snapshot SHA-256: {snapshot_sha}. The snapshot contains aggregate evidence and receipt scalars only.", "SmallProject")
        story.append(ResearchBoundary())

    story.append(PageBreak())
    p("Reproduction, delivery and references", "Heading1")
    h("Reproduction record")
    provenance = read_json(Path(selected["folder"]) / "provenance.json", {}) if selected else {}
    commit = first(metrics, ("git_commit", "commit", "code_commit", "code_sha"),
                   first(provenance.get("code", {}), ("git_commit", "commit", "code_sha"), "See saved run provenance"))
    env = read_json(Path(selected["folder"]) / "environment.json", {}) if selected else {}
    table([["Record", "Value"], ["Code revision", commit], ["Selected experiment", run_name],
           ["Seed", first(config, ("seed",), 2026)],
           ["Data source", "Official Kaggle train_images/ + train_labels.csv"],
           ["GPU", first(env, ("gpu", "gpu_name"), first(metrics, ("gpu",), "See saved environment"))],
           ["Submission reference", first(evidence["submission"], ("ref", "submission_ref", "submission_id", "id"), "Pending")]], [1.55*inch,4.85*inch])
    p("The repository supplies model, data preparation, training, inference, submission validation and report "
      "generation code. Each completed run keeps its full configuration, split/data fingerprints, environment, "
      "metrics, learning history and selected checkpoint locally. Weights, raw images, prediction files and "
      "full unreviewed runtime outputs stay outside Git. An audited executed results notebook may contain "
      "scalar metrics, configurations, plots and aggregate confusion counts for review. "
      "Artifact checksums should be checked before reusing a checkpoint or prediction file.")
    h("Separately labeled exploratory benchmark")
    p(escape(exploratory_method(evidence)), "SmallProject")
    h("Remaining course actions")
    p("Private leaderboard results remain unavailable until released. Course Google Form registration and "
      "Brightspace upload must be confirmed separately. The verified Kaggle deadline is 6 November 2026 "
      "at 23:59:59 EST (UTC-05:00), equivalent to 7 November at 04:59:59 UTC. The assignment gives "
      "9 November 2026 at 23:59 for the report without a timezone; confirm that report timezone in "
      "the course announcements. Any added teammate must complete the formal course/Kaggle procedure.")
    h("References")
    for index, (title, url) in enumerate(references, 1):
        p(f'[{index}] {escape(title)} <link href="{url}" color="#167e75">{url}</link>', "SmallProject")
    p(f'[{len(references)+1}] Course Project 1 assignment, pp. 1-3; <link href="{COMPETITION}/rules" color="#167e75">Kaggle competition rules</link> (retrieved 30 October 2026).', "SmallProject")

    target = output_dir / "ky2738_Project1_Report.pdf"
    doc = SimpleDocTemplate(str(target), pagesize=letter, rightMargin=.55*inch,
                            leftMargin=.55*inch, topMargin=.62*inch, bottomMargin=.58*inch,
                            title="Efficient Residual Learning for CIFAR-10", author="Kevin Yang")
    def decorate(canvas, document):
        canvas.setStrokeColor(colors.HexColor("#cbd8df"))
        canvas.line(.55*inch, .41*inch, 7.95*inch, .41*inch)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(muted)
        canvas.drawString(.55*inch, .27*inch, "Saber | Kevin Yang (ky2738) | Deep Learning Project 1")
        canvas.drawRightString(7.95*inch, .27*inch, f"{document.page}")
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    if research_page_numbers and research_page_numbers[-1] - research_page_numbers[0] + 1 > 2:
        raise RuntimeError("Follow-on research exceeds its two-page allocation")
    from pypdf import PdfReader
    pages = len(PdfReader(str(target)).pages)
    if pages > 8:
        raise RuntimeError(f"Report contains {pages} pages; assignment limit is 8")
    return target, pages


def create_notebook(evidence, selected, plots, output_dir, execute=True):
    import nbformat
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "version": sys.version.split()[0]}
    references = report_references(evidence)
    chosen_name = selected["name"] if selected else None
    def portable_path(path):
        path = Path(path).resolve()
        try:
            return "ROOT / " + repr(path.relative_to(ROOT).as_posix())
        except ValueError:
            return "Path(" + repr(str(path)) + ")"
    nb.cells = [
        nbformat.v4.new_markdown_cell("# Project 1: executed experiment evidence\n\n**Kevin Yang (ky2738), Team Saber**\n\n"
            f"[Public GitHub source]({REPOSITORY})\n\n"
            "This notebook reads measured artifacts; it does not replace training logs with expected accuracy. "
            "The separate training entrypoint contains the main scratch-training implementation. "
            "The public-feedback-driven robust_full extension is exploratory and is reported separately. "
            "Private leaderboard accuracy is not yet available."),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\nimport json, sys\n"
            "ROOT = next((p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'model.py').is_file()), None)\n"
            "if ROOT is None:\n    raise FileNotFoundError('Run this notebook from the cloned project or its notebooks directory.')\n"
            "sys.path.insert(0, str(ROOT))\n"
            "from runtime import configure_runtime\nconfigure_runtime()\n"
            "from build_report import collect_evidence, choose_run, accuracy, public_accuracy, val_accuracy, checkpoint_details, measured_interpretation\n"
            f"evidence = collect_evidence({portable_path(evidence.get('results_dir', ROOT / 'results'))})\n"
            f"selected = choose_run(evidence, {chosen_name!r})\n"
            "print('Team: Saber | Kevin Yang | ky2738')\n"
            "print('Recorded runs:', [run['name'] for run in evidence['runs']])\n"
            "print('Selected run:', selected['name'] if selected else 'Pending')"),
        nbformat.v4.new_markdown_cell("## Architecture and parameter-budget check\n"
            "A fresh CPU construction matches the selected saved architecture configuration (default WRN-22-4 "
            "until a selection is available) and independently checks its parameter count. It does not load "
            "a pretrained or external model."),
        nbformat.v4.new_code_cell(
            "from model import build_model, count_parameters\n"
            "config = selected['config'] if selected else {}\n"
            "model_config = {k: config.get(k, v) for k,v in {'depth':22,'widen_factor':4,'dropout':0.0,'num_classes':10}.items()}\n"
            "model = build_model(model_config)\n"
            "total = count_parameters(model)\ntrainable = count_parameters(model, trainable_only=True)\n"
            "print(model)\n"
            "print(f'Total parameters: {total:,}')\n"
            "print(f'Trainable parameters: {trainable:,}')\n"
            "assert trainable == total\n"
            "if model_config['depth'] == 22 and model_config['widen_factor'] == 4:\n    assert total == 4_298_970\n"
            "assert total <= 5_000_000\ndel model"),
        nbformat.v4.new_markdown_cell("## Actual experimental configurations and results\n"
            "The matched protocol fixes the split, seed, schedule and optimizer. The architecture comparison "
            "changes depth and width jointly; the regularization comparison changes Cutout and MixUp jointly. "
            "The measured effects below do not establish statistical significance or separate the bundled factors.\n\n"
            "### Exploratory robust_full specification\n" + exploratory_method(evidence)),
        nbformat.v4.new_code_cell(
            "for run in evidence['runs']:\n"
            "    print('\\nRUN:', run['name'])\n"
            "    print('CONFIGURATION:', json.dumps(run['config'], indent=2))\n"
            "    print('METRICS:', json.dumps(run['metrics'], indent=2))\n"
            "    print('CHECKPOINT AND INFERENCE:', json.dumps(checkpoint_details(run), indent=2))\n"
            "    print('Epochs recorded:', len(run['history']))\n"
            "if not evidence['runs']:\n    print('Training artifacts are pending.')\n"
            "for finding in measured_interpretation(evidence):\n    print('\\nMEASURED FINDING:', finding)"),
        nbformat.v4.new_code_cell(
            "exploratory = evidence.get('exploratory', {})\n"
            "print('EXPLORATORY RUN: robust_full (separate from the primary study)')\n"
            "print('CONFIGURATION:', json.dumps(exploratory.get('config', {}), indent=2))\n"
            "print('METRICS:', json.dumps(exploratory.get('metrics', {}), indent=2))\n"
            "safe_exploratory_keys = ('epoch','variant','parameter_count','validation_examples','validation_accuracy','flip_tta','tta_views','tta_mode','checkpoint_sha256','submission_sha256','bn_recalibrated_on_training_only','exploratory_after_public_feedback','main_confirmation_endorses_this_run')\n"
            "evaluation = exploratory.get('evaluation', {})\n"
            "print('FIXED CHECKPOINT AND INFERENCE:', json.dumps({k:evaluation[k] for k in safe_exploratory_keys if k in evaluation}, indent=2))\n"
            "print('No development or confirmation accuracy is assigned to this full-data exploration.')"),
        nbformat.v4.new_code_cell(
            "from IPython.display import Image, display\n"
            f"display(Image(filename=str({portable_path(plots['learning_curves'])})))\n"
            f"display(Image(filename=str({portable_path(plots['architecture'])})))\n" +
            (f"display(Image(filename=str({portable_path(plots['confirmation'])})))" if "confirmation" in plots else "")),
        nbformat.v4.new_markdown_cell("## Final test accuracy\n"
            "The public leaderboard measures only half of the 2,000 hidden evaluation images. "
            "Development, confirmation, the primary full-data public score and the best observed official "
            "benchmark score are distinct. The five-entry plan includes a separately labeled robust_full "
            "exploration motivated by official public feedback. Public results may guide entry selection; "
            "the main 160-epoch recipe and its confirmation are not changed or repeated. "
            "Private accuracy remains unknown."),
        nbformat.v4.new_code_cell(
            "metrics = selected['metrics'] if selected else {}\n"
            "if selected and selected['config'].get('train_all'):\n"
            "    print('Final full-data model validation accuracy: not evaluated (recipe already frozen)')\n"
            "else:\n    print('Development accuracy:', accuracy(checkpoint_details(selected)['development_accuracy']) if selected else 'Pending')\n"
            "print('One-time confirmation accuracy:', accuracy(evidence['confirmation'].get('accuracy')))\n"
            "print('Primary full-data test accuracy (Kaggle PUBLIC leaderboard):', accuracy(public_accuracy(evidence['submission'])))\n"
            "print('Private leaderboard test accuracy: unavailable until released')\n"
            "print(f'Final model parameter count: {total:,}')\n"
            "safe_submission_keys = ('submission_ref','public_score','private_score','run','sha256','status','submitted_at','scored_at','commit','source_commit','checkpoint_sha256')\n"
            "submission_record = {k:evidence['submission'][k] for k in safe_submission_keys if k in evidence['submission']}\n"
            "print('Primary submission record (scalar fields only):', json.dumps(submission_record, indent=2))\n"
            "benchmark_keys = ('run','public_score','private_score','submission_ref','submission_sha256','submitted_at','training_examples','parameter_count','flip_tta','status')\n"
            "print('Official benchmark receipts (scalar fields only):')\n"
            "for receipt in evidence['submissions']:\n"
            "    print(json.dumps({k:receipt[k] for k in benchmark_keys if k in receipt}, indent=2))\n"
            "completed = [r for r in evidence['submissions'] if str(r.get('status','')).lower() == 'complete' and public_accuracy(r) is not None]\n"
            "if completed:\n"
            "    best = max(completed,key=public_accuracy)\n"
            "    print('Highest observed public benchmark:', best.get('run'), accuracy(public_accuracy(best)))\n"
            "    print('Best public benchmark is not evidence of the unknown private winner.')"),
        nbformat.v4.new_markdown_cell("## Reproduction and artifact policy\n"
            "Follow the repository README to prepare official competition data, train the chosen configuration, "
            "validate predictions, and generate this report. Data, weights and prediction files stay private and "
            "are excluded from Git. The course staff can inspect the saved manifests, checksums and logs.\n\n"
            "## References\n" + "\n".join(f"{i}. [{title}]({url})" for i,(title,url) in enumerate(references,1))),
    ]
    if "followup" in evidence:
        followup = evidence["followup"]
        snapshot, snapshot_sha = followup_snapshot(followup)
        nb.metadata["research_reporting_snapshot"] = {
            "source_summary_sha256": followup["source_summary_sha256"], "snapshot_sha256": snapshot_sha}
        nb.cells[-1:-1] = [
            nbformat.v4.new_markdown_cell("## Follow-on research: public reporting snapshot\n\n"
                "The original six executed code cells above remain the primary study. These two additional "
                "cells read an embedded, allowlisted public aggregate snapshot; they do not load data, "
                "checkpoints, individual predictions or private runtime records. The same projection is saved "
                "as `research_followup_evidence.json` beside this notebook. The original six cells still require "
                "their original private training artifacts to rerun.\n\n"
                "All follow-on development results reuse the original development partition. The original "
                f"study remains frozen. {confirmation_scope(evidence)} That confirmation was not reused. "
                "A full-50k refit has no independent development score. Conditional paired intervals describe "
                "postselected fitted checkpoints, not independent confirmation or hidden-test accuracy.\n\n"
                "SWA [7], SAM [8], CutMix [9], ResizeMix [10] and test-time aggregation [11] are reported "
                "only to the extent supported by the measured snapshot. Engineering-only SE candidates [12] "
                "remain separate from measured model outcomes."),
            nbformat.v4.new_code_cell(
                "import hashlib\n"
                f"_followup_snapshot = {snapshot!r}\n"
                f"_expected_snapshot_sha256 = {snapshot_sha!r}\n"
                "assert hashlib.sha256(_followup_snapshot.encode('utf-8')).hexdigest() == _expected_snapshot_sha256\n"
                "followup = json.loads(_followup_snapshot)\n"
                "print('EXPLORATORY PUBLIC SNAPSHOT:', followup['recorded_at'])\n"
                "print('Source-summary SHA-256:', followup['source_summary_sha256'])\n"
                "print('Public snapshot SHA-256:', _expected_snapshot_sha256)\n"
                "print('Research source:', followup['repository'])\n"
                "print('Fixed v3 development baseline:', json.dumps(followup['baseline'], indent=2))\n"
                "for row in followup['rows']:\n"
                "    print('RESEARCH AGGREGATE:', json.dumps(row, indent=2))\n"
                "ci_rows = [r for r in followup['rows'] if r['clean_ci95'] is not None and r['stress_ci95'] is not None]\n"
                "if ci_rows:\n"
                "    import matplotlib.pyplot as plt\n"
                "    import io\n"
                "    from IPython.display import Image, display\n"
                "    plt.switch_backend('Agg')\n"
                "    fig, axes = plt.subplots(1, 2, figsize=(10, max(2.5, .48*len(ci_rows))), sharey=True)\n"
                "    for axis, metric, title in zip(axes, ('clean', 'stress'), ('Clean', 'Synthetic stress mean')):\n"
                "        for y, row in enumerate(ci_rows):\n"
                "            lo, hi = row[metric+'_ci95']\n"
                "            axis.hlines(y, 100*lo, 100*hi, color='#167e75', linewidth=2)\n"
                "            axis.plot(100*row[metric+'_delta'], y, 'o', color='#173346')\n"
                "        axis.axvline(0, color='grey', linewidth=.8)\n"
                "        axis.set(title=title, xlabel='Difference vs v3 baseline (percentage points)')\n"
                "        axis.set_yticks(range(len(ci_rows)), [r['label'] for r in ci_rows])\n"
                "    fig.suptitle('Conditional paired 95% intervals: reused/postselected development')\n"
                "    fig.tight_layout()\n"
                "    figure_buffer = io.BytesIO()\n"
                "    fig.savefig(figure_buffer, format='png', dpi=160, bbox_inches='tight')\n"
                "    display(Image(data=figure_buffer.getvalue()))\n"
                "    plt.close(fig)\n"
                "for statement in followup['limitations']:\n    print('LIMITATION:', statement)\n"
                "for statement in followup['engineering']:\n    print('ENGINEERING STATUS:', statement)\n"
                f"print({(confirmation_scope(evidence) + ' Not reused for research.')!r})"),
            nbformat.v4.new_markdown_cell("## Follow-on official scoring\n\n"
                "Only validated completed official receipts appear below. Benchmark completion, a CSV, "
                "an upload intent and pending scoring do not establish a public score. Original primary, "
                "original best, newly scored research best and best observed across both are distinct. "
                "Public selection does not establish the private winner."),
            nbformat.v4.new_code_cell(
                "from build_report import receipt_label, scored_outcome_text\n"
                "print('Original primary public score:', accuracy(followup['original_primary_public']))\n"
                "print('Best original public score:', accuracy(followup['original_best_public']))\n"
                "print('Best newly scored research public score:', accuracy(followup['best_new_public']))\n"
                "print('Best observed public score (original + research):', accuracy(followup['best_observed_public']))\n"
                "print('Unscored receipt records:', followup['unscored_receipts_count'])\n"
                "print('Newly scored research receipts:', len(followup['official_submissions']))\n"
                "for receipt in followup['official_submissions']:\n"
                "    print('DISPLAY RECIPE LABEL:', receipt_label(followup, receipt))\n"
                "    print('OFFICIAL RESEARCH RECEIPT:', json.dumps(receipt, indent=2))\n"
                "outcome = scored_outcome_text(followup)\n"
                "if outcome:\n    print(outcome)\n"
                "if not followup['official_submissions']:\n"
                "    print('No newly scored research submission in this reporting snapshot.')\n"
                "print('Missing private scores remain unknown; no original confirmation result is assigned to these entries.')"),
        ]
    if execute:
        from nbclient import NotebookClient
        NotebookClient(nb, timeout=180, kernel_name="python3",
                       resources={"metadata": {"path": str(ROOT)}}).execute()
    path = output_dir / "Project1_Results_Executed.ipynb"
    nbformat.write(nb, path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--research-summary", type=Path,
                        help="Audited public follow-on research aggregates; primary study stays separate")
    parser.add_argument("--research-receipts", type=Path,
                        default=ROOT / "results" / "research" / "submission_receipts",
                        help="Actual scored research receipts, used only with --research-summary")
    parser.add_argument("--final-run", help="Run associated with the selected Kaggle submission")
    parser.add_argument("--no-execute", action="store_true", help="Create a notebook draft without execution")
    args = parser.parse_args()
    args.output_dir = followup_output_dir(args.output_dir, args.research_summary is not None)
    evidence = collect_evidence(args.results_dir)
    evidence["results_dir"] = str(args.results_dir.resolve())
    if args.research_summary is not None:
        attach_followup(evidence, args.research_summary, args.research_receipts)
    args.output_dir.mkdir(parents=True, exist_ok=args.research_summary is None)
    if "followup" in evidence:
        snapshot_sha = write_followup_snapshot(args.output_dir / "research_followup_evidence.json", evidence["followup"])
    selected = choose_run(evidence, args.final_run)
    plots = make_plots(evidence, args.output_dir, selected)
    pdf, pages = create_pdf(evidence, selected, plots, args.output_dir)
    notebook = create_notebook(evidence, selected, plots, args.output_dir, execute=not args.no_execute)
    artifacts = {"report": str(pdf), "report_pages": pages, "notebook": str(notebook),
                 "notebook_executed": not args.no_execute,
                 "selected_run": selected["name"] if selected else None,
                 "public_test_accuracy": public_accuracy(evidence["submission"]),
                 "private_test_accuracy": None,
                 "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in [pdf,notebook]}}
    if "followup" in evidence:
        followup = evidence["followup"]
        artifacts["followup"] = {key: followup[key] for key in (
            "recorded_at", "source_summary_sha256", "original_primary_public", "original_best_public",
            "best_new_public", "best_observed_public", "unscored_receipts_count")}
        artifacts["followup"].update(snapshot_sha256=snapshot_sha,
                                      scored_receipts_count=len(followup["official_submissions"]))
        artifacts["sha256"]["research_followup_evidence.json"] = snapshot_sha
        artifacts["sha256"][plots["followup"].name] = hashlib.sha256(plots["followup"].read_bytes()).hexdigest()
    (args.output_dir / "report_manifest.json").write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
    print(json.dumps(artifacts, indent=2))


if __name__ == "__main__":
    main()
