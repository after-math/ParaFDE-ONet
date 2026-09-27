"""Metrics shared by Frozen, Direct LM and Direct DE."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def normalized_parameter_rmse(
    estimate: np.ndarray, truth: np.ndarray, bounds: np.ndarray
) -> float:
    difference = (np.asarray(estimate) - np.asarray(truth)) / (
        bounds[:, 1] - bounds[:, 0]
    )
    return float(np.sqrt(np.mean(difference**2)))


def physical_parameter_rmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(estimate) - np.asarray(truth)) ** 2)))


def relative_l2(reference: np.ndarray, prediction: np.ndarray) -> float:
    numerator = np.linalg.norm(np.asarray(prediction) - np.asarray(reference))
    denominator = max(float(np.linalg.norm(reference)), 1.0e-15)
    return float(numerator / denominator)


def evaluate_estimate(
    adapter: Any,
    panel: Mapping[str, np.ndarray],
    case_index: int,
    estimate: np.ndarray,
    truth_internal_step: float,
    thresholds: Sequence[float],
    boundary_tolerance: float,
) -> dict[str, Any]:
    truth = np.asarray(panel["parameters"][case_index], dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    bounds = adapter.parameter_bounds
    normalized_rmse = normalized_parameter_rmse(estimate, truth, bounds)
    repeated_estimate = np.repeat(
        estimate.reshape(1, 2), panel["histories"].shape[0], axis=0
    )
    reconstructed = adapter.solve_batch(
        np.asarray(panel["histories"], dtype=np.float64),
        repeated_estimate,
        np.asarray(panel["history_grid"], dtype=np.float64),
        np.asarray(panel["output_times"], dtype=np.float64),
        float(truth_internal_step),
    )
    reference = np.asarray(panel["reference"][case_index], dtype=np.float64)
    unit_estimate = adapter.physical_to_unit_numpy(estimate)
    boundary = bool(
        np.any(unit_estimate <= boundary_tolerance)
        or np.any(unit_estimate >= 1.0 - boundary_tolerance)
    )
    result: dict[str, Any] = {
        "true_parameters": truth.tolist(),
        "estimated_parameters": estimate.tolist(),
        "parameter_normalized_rmse": normalized_rmse,
        "parameter_physical_rmse": physical_parameter_rmse(estimate, truth),
        "trajectory_relative_l2": relative_l2(reference, reconstructed),
        "boundary_estimate": int(boundary),
    }
    for index, name in enumerate(adapter.parameter_names):
        result[f"true_{name}"] = float(truth[index])
        result[f"estimated_{name}"] = float(estimate[index])
        result[f"absolute_error_{name}"] = float(abs(estimate[index] - truth[index]))
        result[f"normalized_absolute_error_{name}"] = float(
            abs(estimate[index] - truth[index])
            / (bounds[index, 1] - bounds[index, 0])
        )
    for threshold in thresholds:
        tag = str(float(threshold)).replace(".", "p")
        result[f"success_nrmse_le_{tag}"] = int(normalized_rmse <= float(threshold))
    return result
