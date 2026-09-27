"""Deterministic cross-system parameter design."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def latin_hypercube(count: int, dimension: int, rng: np.random.Generator) -> np.ndarray:
    if count < 1 or dimension < 1:
        raise ValueError("Latin hypercube dimensions must be positive")
    result = np.empty((count, dimension), dtype=np.float64)
    for column in range(dimension):
        result[:, column] = (rng.permutation(count) + rng.random(count)) / count
    return result


def minimum_pairwise_distance(points: np.ndarray) -> float:
    if len(points) < 2:
        return float("inf")
    difference = points[:, None, :] - points[None, :, :]
    distance = np.sqrt(np.sum(difference**2, axis=-1))
    distance[np.diag_indices_from(distance)] = np.inf
    return float(np.min(distance))


def maximin_latin_hypercube(
    count: int,
    candidates: int,
    rng: np.random.Generator,
    margin: float,
) -> np.ndarray:
    if not 0.0 <= margin < 0.5:
        raise ValueError("interior margin must lie in [0,0.5)")
    best: np.ndarray | None = None
    best_score = -np.inf
    for _ in range(max(1, candidates)):
        design = latin_hypercube(count, 2, rng)
        design = margin + (1.0 - 2.0 * margin) * design
        score = minimum_pairwise_distance(design)
        if score > best_score:
            best = design
            best_score = score
    if best is None:
        raise RuntimeError("failed to build maximin Latin hypercube")
    return best


def maximin_edge_points(count: int, margin: float) -> np.ndarray:
    """Select deterministic points near all four edges by farthest-point sampling."""
    if count == 0:
        return np.empty((0, 2), dtype=np.float64)
    if not 0.0 < margin < 0.5:
        raise ValueError("edge margin must lie in (0,0.5)")
    coordinates = np.linspace(margin, 1.0 - margin, 97)
    pool = np.unique(
        np.concatenate(
            [
                np.column_stack((np.full_like(coordinates, margin), coordinates)),
                np.column_stack((np.full_like(coordinates, 1.0 - margin), coordinates)),
                np.column_stack((coordinates, np.full_like(coordinates, margin))),
                np.column_stack((coordinates, np.full_like(coordinates, 1.0 - margin))),
            ],
            axis=0,
        ),
        axis=0,
    )
    # Start in a corner and repeatedly choose the farthest candidate. Stable
    # argmax makes the design independent of platform sort order.
    selected = [int(np.argmin(np.sum(pool, axis=1)))]
    while len(selected) < count:
        difference = pool[:, None, :] - pool[np.asarray(selected)][None, :, :]
        nearest = np.min(np.sqrt(np.sum(difference**2, axis=-1)), axis=1)
        nearest[np.asarray(selected)] = -np.inf
        selected.append(int(np.argmax(nearest)))
    return pool[np.asarray(selected)]


def build_parameter_design(
    case_count: int, config: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Return unit points and labels; formal 50 rows are 40 interior + 10 edge."""
    if case_count < 1:
        raise ValueError("case_count must be positive")
    formal_interior = int(config["interior_count"])
    formal_edge = int(config["edge_count"])
    if case_count == formal_interior + formal_edge:
        edge_count = formal_edge
    else:
        edge_count = min(formal_edge, case_count // 5)
    interior_count = case_count - edge_count
    rng = np.random.default_rng(int(config["seed"]))
    interior = maximin_latin_hypercube(
        interior_count,
        int(config["maximin_candidates"]),
        rng,
        float(config["interior_margin"]),
    )
    edge = maximin_edge_points(edge_count, float(config["edge_margin"]))
    points = np.concatenate((interior, edge), axis=0)
    labels = np.asarray(
        ["interior"] * interior_count + ["edge"] * edge_count, dtype="<U16"
    )
    if points.shape != (case_count, 2) or np.any(points < 0.0) or np.any(points > 1.0):
        raise RuntimeError("invalid parameter design")
    return points.astype(np.float64), labels

