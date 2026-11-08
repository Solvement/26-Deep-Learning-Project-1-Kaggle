"""Aggregate paired translation diagnostics without retaining predictions."""
from __future__ import annotations

import numpy as np

from research.development_error_slices import CLASSES, REPLICATES, SEED


RADII = {"reflect": (1, 2, 3, 4), "canvas": (2, 3, 4)}
DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def offsets(distance):
    return tuple((dx * distance, dy * distance) for dx, dy in DIRECTIONS)


def _predictions(values):
    values = np.asarray(values)
    if values.shape != (5000,) or values.dtype.kind not in "iu" or np.any((values < 0) | (values >= len(CLASSES))):
        raise ValueError("Exactly 5000 integer class predictions in [0, 9] required")
    return values


def _counts(correct, clean_correct, flipped):
    observations = correct.size
    opportunities = int(np.count_nonzero(clean_correct)) * correct.shape[1]
    lost = int(np.count_nonzero(clean_correct & ~correct))
    recovered = int(np.count_nonzero(~clean_correct & correct))
    hits = int(np.count_nonzero(correct))
    flips = int(np.count_nonzero(flipped))
    return {
        "images": len(correct),
        "directions": correct.shape[1],
        "observations": observations,
        "correct": hits,
        "accuracy": hits / observations,
        "clean_correct_opportunities": opportunities,
        "clean_correct_to_wrong": lost,
        "clean_correct_to_wrong_rate": lost / opportunities if opportunities else None,
        "clean_wrong_to_correct": recovered,
        "prediction_flips": flips,
        "prediction_flip_rate": flips / observations,
    }


def _summarize(predictions, labels, clean):
    correct = predictions == labels[:, None]
    clean_correct = clean[:, None] == labels[:, None]
    flipped = predictions != clean[:, None]
    per_class = {}
    for index, name in enumerate(CLASSES):
        mask = labels == index
        per_class[name] = _counts(correct[mask], clean_correct[mask], flipped[mask])
    return {
        **_counts(correct, clean_correct, flipped),
        "per_true_class": per_class,
    }


def _joint_intervals(contrasts, rng, replicates):
    names = tuple(contrasts)
    vectors = np.column_stack([contrasts[name][0] for name in names])
    scales = np.array([contrasts[name][1] for name in names])
    patterns, frequencies = np.unique(vectors, axis=0, return_counts=True)
    probabilities = frequencies / len(vectors)
    draws = np.empty((replicates, len(names)), dtype=np.int64)
    # Resample whole images, with every signed direction and border treatment
    # attached. Identical contribution vectors share a multinomial category.
    for start in range(0, replicates, 256):
        chunk = draws[start:start + 256]
        chunk[:] = rng.multinomial(len(vectors), probabilities, size=len(chunk)) @ patterns
    limits = np.quantile(draws / (len(vectors) * scales), (0.025, 0.975), axis=0)
    points = vectors.mean(axis=0) / scales
    return {
        name: {
            "delta": float(points[index]),
            "paired_bootstrap_ci95": limits[:, index].tolist(),
        }
        for index, name in enumerate(names)
    }


def summarize_shift_predictions(labels, clean, shifted, *, seed=SEED, replicates=REPLICATES):
    """Summarize one checkpoint and same-order development predictions only."""
    labels = _predictions(labels)
    if set(labels.tolist()) != set(range(len(CLASSES))):
        raise ValueError("Development labels must span all ten classes")
    if set(clean) != set(RADII) or set(shifted) != set(RADII):
        raise ValueError("Reflect and fixed-canvas modes require separate clean baselines")
    if type(replicates) is not int or replicates <= 0 or type(seed) is not int or seed < 0:
        raise ValueError("A positive bootstrap replicate count and nonnegative integer seed are required")

    by_border = {}
    contrast_vectors = {}
    counts = {}
    for border, distances in RADII.items():
        baseline = _predictions(clean[border])
        if set(shifted[border]) != {offset for distance in distances for offset in offsets(distance)}:
            raise ValueError(f"Exact four cardinal offsets per configured distance required for {border}")
        by_offset, by_distance = {}, {}
        counts[border] = {}
        for distance in distances:
            keys = offsets(distance)
            columns = [_predictions(shifted[border][key]) for key in keys]
            for (dx, dy), predictions in zip(keys, columns):
                by_offset[f"dx{dx:+d}_dy{dy:+d}"] = _summarize(predictions[:, None], labels, baseline)
            matrix = np.column_stack(columns)
            by_distance[str(distance)] = _summarize(matrix, labels, baseline)
            counts[border][distance] = np.count_nonzero(matrix == labels[:, None], axis=1)
        by_border[border] = {
            "clean": _summarize(baseline[:, None], labels, baseline),
            "by_offset": by_offset,
            "by_distance": by_distance,
        }
        hits = counts[border]
        contrast_vectors[border] = {
            "distance3_excess_error_vs_distance2_and4": (hits[2] + hits[4] - 2 * hits[3], 8),
            "distance4_minus_distance3_accuracy": (hits[4] - hits[3], 4),
        }

    contrast_vectors["canvas_minus_reflect"] = {
        name: (contrast_vectors["canvas"][name][0] - reflect_values, scale)
        for name, (reflect_values, scale) in contrast_vectors["reflect"].items()
    }
    joint = {f"{border}/{name}": value for border, block in contrast_vectors.items()
             for name, value in block.items()}
    intervals = _joint_intervals(joint, np.random.default_rng(seed), replicates)
    for border, block in by_border.items():
        block["paired_contrasts"] = {
            name: intervals[f"{border}/{name}"] for name in contrast_vectors[border]
        }
    return {
        "schema": "official_development_shift_stats_v1",
        "development_examples": len(labels),
        "bootstrap_seed": seed,
        "bootstrap_replicates": replicates,
        "bootstrap_unit": "original_image",
        "bootstrap_intervals": "pointwise_percentile_95",
        "by_border": by_border,
        "paired_border_controls": {
            name: intervals[f"canvas_minus_reflect/{name}"]
            for name in contrast_vectors["canvas_minus_reflect"]
        },
        "raw_predictions_persisted": False,
    }
