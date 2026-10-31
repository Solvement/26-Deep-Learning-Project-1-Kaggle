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
]
RUN_ORDER = ("baseline", "architecture", "regularized", "final_full")
RUN_NAMES = set(RUN_ORDER)


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
    submissions = read_json(results_dir / "kaggle_submissions.json", [])
    if isinstance(submissions, dict):
        submissions = submissions.get("submissions", [])
    if not isinstance(submissions, list):
        raise ValueError("Kaggle benchmark receipts must be a list")
    runs.sort(key=lambda run: RUN_ORDER.index(run["name"]))
    return {"runs": runs, "submission": submission, "confirmation": confirmation,
            "selection": selection, "stability": stability, "submissions": submissions,
            "repository": REPOSITORY,
            "team": "Saber", "author": "Kevin Yang", "netid": "ky2738"}


def completed_benchmarks(evidence):
    return [record for record in evidence.get("submissions", [])
            if str(record.get("status", "")).lower() == "complete" and public_accuracy(record) is not None]


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
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

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
    benchmarks = completed_benchmarks(evidence)
    best_benchmark = max(benchmarks, key=public_accuracy) if benchmarks else None
    val = recipe_accuracy
    run_name = selected["name"] if selected else "No measured run yet"

    p("Efficient residual learning<br/>for CIFAR-10", "TitleProject")
    p("Deep Learning Project 1 | Team Saber<br/>Kevin Yang (ky2738)", "SubtitleProject")
    p(f'<b>Public code:</b> <link href="{REPOSITORY}" color="#167e75">{REPOSITORY}</link>')
    story.append(Spacer(1, 10))
    h("Overview")
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
        p(f"Highest observed public benchmark: <b>{escape(str(best_benchmark.get('run', 'unknown')))} "
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
        p("Five daily entries benchmark the frozen baseline, architecture, regularized, full-data and seed-repeat models. Public scores may guide official Kaggle entry selection; they do not change the completed training/confirmation protocol. Private scores remain unavailable.", "SmallProject")
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
        p("The user requested use of the five-entry daily allowance for official comparisons. Predictions use "
          "existing fixed checkpoints and development-selected inference rules. Selecting the highest of "
          "five public scores can favor sampling noise on the 1,000-image public subset. The benchmark "
          "table and primary full-data result are both reported; no public-score-driven retraining, "
          "post-confirmation recipe change or claim of a known private winner is made.", "SmallProject")
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
    h("Design lessons and remaining work")
    p("The parameter budget favors widening a moderate-depth network rather than using an unmodified "
      "large ImageNet model. A CIFAR stem preserves the limited 32x32 spatial information. Separating "
      "data validation, checkpoint selection and Kaggle scoring makes an apparent score auditable. "
      "The local runtime required a process-local cuDNN path correction before importing PyTorch; "
      "no global CUDA package change was needed.")
    p("Private leaderboard results remain unavailable until released. Course Google Form registration and "
      "Brightspace upload must be confirmed separately. The verified Kaggle deadline is 6 November 2026 "
      "at 23:59:59 EST (UTC-05:00), equivalent to 7 November at 04:59:59 UTC. The assignment gives "
      "9 November 2026 at 23:59 for the report without a timezone; confirm that report timezone in "
      "the course announcements. Any added teammate must complete the formal course/Kaggle procedure.")
    h("References")
    for index, (title, url) in enumerate(REFERENCES, 1):
        p(f'[{index}] {escape(title)} <link href="{url}" color="#167e75">{url}</link>', "SmallProject")
    p(f'[6] Course Project 1 assignment, pp. 1-3; <link href="{COMPETITION}/rules" color="#167e75">Kaggle competition rules</link> (retrieved 30 October 2026).', "SmallProject")

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
            "The separate training entrypoint contains the full scratch-training implementation. "
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
            "The measured effects below do not establish statistical significance or separate the bundled factors."),
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
            "from IPython.display import Image, display\n"
            f"display(Image(filename=str({portable_path(plots['learning_curves'])})))\n"
            f"display(Image(filename=str({portable_path(plots['architecture'])})))\n" +
            (f"display(Image(filename=str({portable_path(plots['confirmation'])})))" if "confirmation" in plots else "")),
        nbformat.v4.new_markdown_cell("## Final test accuracy\n"
            "The public leaderboard measures only half of the 2,000 hidden evaluation images. "
            "Development, confirmation, the primary full-data public score and the best observed official "
            "benchmark score are distinct. Five daily submissions compare fixed models and may guide "
            "official Kaggle entry selection; they do not retune the frozen training/confirmation protocol. "
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
            "## References\n" + "\n".join(f"{i}. [{title}]({url})" for i,(title,url) in enumerate(REFERENCES,1))),
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
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--final-run", help="Run associated with the selected Kaggle submission")
    parser.add_argument("--no-execute", action="store_true", help="Create a notebook draft without execution")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence = collect_evidence(args.results_dir)
    evidence["results_dir"] = str(args.results_dir.resolve())
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
    (args.output_dir / "report_manifest.json").write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
    print(json.dumps(artifacts, indent=2))


if __name__ == "__main__":
    main()
