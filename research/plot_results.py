"""Plot public aggregate paired evidence; never load images, labels or weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"docs/research-followup-results.json")
    parser.add_argument("--output", type=Path, default=ROOT/"docs/figures/research-paired-v3.png")
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    rows = [
        ("WRN34-3", data["wrn34_3_paired"], "#2868a8", "o"),
        ("SAM / WRN22-4", data["SAM_paired"], "#bc641e", "s"),
        ("ResizeMix / WRN22-4", data["resizemix_local_paired"], "#287d62", "^"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, case, title, limits in zip(
        axes, ("clean", "stress"), ("Clean development", "Mean of five stress views"),
        ((-.75, 1.45), (-.2, 3.65)),
    ):
        ax.axvline(0, color="#656565", linestyle="--", linewidth=1)
        for index, (name, evidence, color, marker) in enumerate(rows):
            bootstrap = evidence["bootstrap"]
            if bootstrap["original_images"] != 5000 or bootstrap["replicates"] != 10000:
                raise ValueError("Complete 5,000-original-image paired evidence is required")
            value = bootstrap[case]
            delta = value["delta_accuracy"]*100
            low, high = [endpoint*100 for endpoint in value["percentile_ci95"]]
            if not low <= delta <= high:
                raise ValueError("Paired interval does not bracket its point estimate")
            ax.errorbar(delta, index, xerr=[[delta-low], [high-delta]], fmt=marker,
                        color=color, markersize=7, capsize=4, linewidth=1.8)
            ax.text(high+.04, index, f"{delta:+.3f} pp", va="center", fontsize=9, color=color)
        ax.set(title=title, xlabel="Accuracy difference vs baseline flip (percentage points)", xlim=limits)
        ax.set_yticks(range(len(rows)), [row[0] for row in rows])
        ax.grid(axis="x", alpha=.18)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].invert_yaxis()
    fig.suptitle("Exploratory paired comparison — primary seed 2026", fontsize=14, y=.97)
    fig.text(.02, .08, "Dots: measured differences. Whiskers: 95% image-cluster percentile bootstrap intervals.", fontsize=9)
    fig.text(.02, .035, "Reused/postselected development; conditional intervals. Weighted candidate TTA vs baseline flip; local v3 BF16 inference, differing training recipes/backends.", fontsize=8)
    fig.subplots_adjust(left=.18, right=.98, top=.83, bottom=.25, wspace=.18)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    plt.close(fig)
    print(json.dumps({"output": str(args.output), "aggregate_only": True, "models": len(rows)}))


if __name__ == "__main__":
    main()
