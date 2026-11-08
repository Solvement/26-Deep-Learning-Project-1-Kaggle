"""Aggregate matched development correctness without writing row-level data."""
from __future__ import annotations

import math

import numpy as np


CASES = ("clean", "noise_004", "boxblur3", "contrast065", "brightness080", "resize24")
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
SEED, REPLICATES = 20261031, 10000
FAMILY_SIZE = len(CASES) * (len(CLASSES) + 1)
ALPHA = 0.05


def _exact_mcnemar(candidate_only, incumbent_only):
    discordant = candidate_only + incumbent_only
    tail = sum(math.comb(discordant, j) for j in range(min(candidate_only, incumbent_only) + 1))
    return min(1.0, (2 * tail) / (1 << discordant))


def _joint_intervals(candidate, incumbent, rng, replicates):
    differences = candidate.astype(np.int8) - incumbent.astype(np.int8)
    patterns, counts = np.unique(differences, axis=0, return_counts=True)
    probabilities = counts / len(candidate)
    draws = np.empty((replicates, len(CASES)), dtype=np.int64)
    # Each draw samples original images with all six cases together. Equal
    # signed difference vectors can share one multinomial category exactly.
    for start in range(0, replicates, 256):
        chunk = draws[start:start + 256]
        counts = rng.multinomial(len(candidate), probabilities, size=len(chunk))
        chunk[:] = counts @ patterns
    return np.quantile(draws / len(candidate), (0.025, 0.975), axis=0).T


def _summarize(candidate, incumbent, interval):
    n = len(candidate)
    both_right = int(np.count_nonzero(candidate & incumbent))
    candidate_only = int(np.count_nonzero(candidate)) - both_right
    incumbent_only = int(np.count_nonzero(incumbent)) - both_right
    both_wrong = n - both_right - candidate_only - incumbent_only
    p = _exact_mcnemar(candidate_only, incumbent_only)
    adjusted_p = min(1.0, p * FAMILY_SIZE)
    return {
        "examples": n,
        "both_right": both_right,
        "both_wrong": both_wrong,
        "candidate_only": candidate_only,
        "incumbent_only": incumbent_only,
        "candidate_accuracy": (both_right + candidate_only) / n,
        "incumbent_accuracy": (both_right + incumbent_only) / n,
        "candidate_minus_incumbent": (candidate_only - incumbent_only) / n,
        "paired_bootstrap_ci95": interval.tolist(),
        "mcnemar_exact_p": p,
        "bonferroni_adjusted_p": adjusted_p,
        "passes_familywise_0_05": adjusted_p <= ALPHA,
    }


def summarize_slices(labels, candidate, incumbent, cases=CASES, *, seed=SEED, replicates=REPLICATES):
    """Use one fixed 5k development split and six matched cases per model."""
    labels = np.asarray(labels)
    candidate, incumbent = np.asarray(candidate), np.asarray(incumbent)
    cases = tuple(cases)
    if cases != CASES:
        raise ValueError("Fixed development cases and order required")
    if labels.shape != (5000,) or labels.dtype.kind not in "iu" or set(labels.tolist()) != set(range(10)):
        raise ValueError("Exactly 5000 integer development labels spanning the ten classes required")
    if any(flags.shape != (5000, len(CASES)) or flags.dtype != np.bool_ for flags in (candidate, incumbent)):
        raise ValueError("Matched 5000-by-6 Boolean correctness matrices required")
    if type(replicates) is not int or replicates <= 0 or type(seed) is not int or seed < 0:
        raise ValueError("A positive bootstrap replicate count and nonnegative integer seed are required")

    rng = np.random.default_rng(seed)
    def summarize_group(a, b):
        intervals = _joint_intervals(a, b, rng, replicates)
        return {case: _summarize(a[:, column], b[:, column], intervals[column])
                for column, case in enumerate(CASES)}

    overall = summarize_group(candidate, incumbent)
    per_class = {}
    for index, name in enumerate(CLASSES):
        mask = labels == index
        per_class[name] = summarize_group(candidate[mask], incumbent[mask])
    return {
        "schema": "official_development_error_slices_v1",
        "bootstrap_seed": seed,
        "bootstrap_replicates": replicates,
        "familywise_alpha": ALPHA,
        "bonferroni_family_size": FAMILY_SIZE,
        "overall": overall,
        "per_class": per_class,
    }


def _direction(primary, repeat):
    a, b = primary["candidate_minus_incumbent"], repeat["candidate_minus_incumbent"]
    return {
        "primary_delta": a,
        "repeat_delta": b,
        "same_nonzero_direction": (a > 0 and b > 0) or (a < 0 and b < 0),
        "passes_familywise_in_both": primary["passes_familywise_0_05"] and repeat["passes_familywise_0_05"],
    }


def compare_directions(primary, repeat):
    """Compare two separately summarized seeds on the same reused dev split."""
    return {
        "same_development_images": True,
        "seed_results_pooled": False,
        "overall": {case: _direction(primary["overall"][case], repeat["overall"][case]) for case in CASES},
        "per_class": {
            name: {case: _direction(primary["per_class"][name][case], repeat["per_class"][name][case])
                   for case in CASES}
            for name in CLASSES
        },
    }
