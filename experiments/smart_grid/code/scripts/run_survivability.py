#!/usr/bin/env python3
"""Frozen-operator probabilistic survivability against direct DDE Monte Carlo."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import itertools
import json
import logging
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
CODE_DIR = PROJECT_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from equation import NODE_COUNT, SmartGridConfig, solve_batch


LOGGER = logging.getLogger("survivability")


def save_json(path: Path, values: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(values), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def save_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    temporary.replace(path)


def save_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to save empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(stream)
    LOGGER.addHandler(file_handler)


def risk_from_trajectories(
    trajectories: np.ndarray, edges: tuple[tuple[int, int], ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Return frequency and gauge-independent edge-angle excursion for each case."""
    values = np.asarray(trajectories)
    if values.ndim != 3 or values.shape[-1] != 2 * NODE_COUNT:
        raise ValueError("trajectories must have shape [N,Q,8]")
    frequency_risk = np.max(np.abs(values[..., NODE_COUNT:]), axis=(1, 2))
    edge_values = np.stack(
        [values[..., left] - values[..., right] for left, right in edges], axis=-1
    )
    angle_risk = np.max(np.abs(edge_values), axis=(1, 2))
    return frequency_risk.astype(np.float64), angle_risk.astype(np.float64)


def parameter_grid(equation: SmartGridConfig, levels: int) -> np.ndarray:
    if levels < 2:
        raise ValueError("parameter grid requires at least two levels per dimension")
    axes = [
        np.linspace(lower, upper, levels, dtype=np.float64)
        for lower, upper in equation.parameter_bounds
    ]
    return np.asarray(list(itertools.product(*axes)), dtype=np.float32)


def survivability_surface(
    frequency_risk: np.ndarray,
    angle_risk: np.ndarray,
    frequency_thresholds: np.ndarray,
    angle_thresholds: np.ndarray,
) -> np.ndarray:
    """Estimate S_T for every parameter and threshold pair over shared histories."""
    frequency = np.asarray(frequency_risk, dtype=np.float64)
    angle = np.asarray(angle_risk, dtype=np.float64)
    if frequency.ndim != 2 or angle.shape != frequency.shape:
        raise ValueError("risk arrays must both have shape [H,P]")
    safe = (
        frequency[:, :, None, None] <= frequency_thresholds[None, None, :, None]
    ) & (angle[:, :, None, None] <= angle_thresholds[None, None, None, :])
    return np.mean(safe, axis=0, dtype=np.float64)


def wilson_interval(
    probability: np.ndarray, sample_count: int, z_value: float = 1.959963984540054
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(probability, dtype=np.float64)
    if sample_count < 1 or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("invalid probability or sample count")
    denominator = 1.0 + z_value**2 / sample_count
    center = (values + z_value**2 / (2.0 * sample_count)) / denominator
    radius = z_value * np.sqrt(
        values * (1.0 - values) / sample_count + z_value**2 / (4.0 * sample_count**2)
    ) / denominator
    return np.maximum(0.0, center - radius), np.minimum(1.0, center + radius)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_ranks = _average_ranks(np.asarray(left).ravel())
    right_ranks = _average_ranks(np.asarray(right).ravel())
    if np.std(left_ranks) == 0.0 or np.std(right_ranks) == 0.0:
        return float("nan")
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def decode_cartesian_features(
    model: Any,
    history_features: Any,
    parameter_features: Any,
    trunk_features: Any,
) -> Any:
    """Decode H histories crossed with P parameters using cached branch features."""
    import torch

    normalized = model.latent_scale * torch.einsum(
        "hsl,pl,ql->hpqs", history_features, parameter_features, trunk_features
    )
    normalized = normalized + model.output_bias[None, None, None, :]
    return (
        normalized * model.state_std[None, None, None, :]
        + model.state_mean[None, None, None, :]
    )


def predict_operator_risks(
    model: Any,
    histories: np.ndarray,
    parameters: np.ndarray,
    output_times: np.ndarray,
    edges: tuple[tuple[int, int], ...],
    device: Any,
    history_batch_size: int,
    parameter_batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    history_count = histories.shape[0]
    parameter_count = parameters.shape[0]
    frequency_risk = np.empty((history_count, parameter_count), dtype=np.float64)
    angle_risk = np.empty_like(frequency_risk)
    histories_tensor = torch.as_tensor(histories, dtype=torch.float32, device=device)
    parameters_tensor = torch.as_tensor(parameters, dtype=torch.float32, device=device)
    times_tensor = torch.as_tensor(output_times, dtype=torch.float32, device=device)
    with torch.inference_mode():
        parameter_features = model.encode_parameters(parameters_tensor)
        trunk_features = model.encode_times(times_tensor, 1)[0]
        for history_start in range(0, history_count, history_batch_size):
            history_stop = min(history_start + history_batch_size, history_count)
            history_features = model.encode_histories(
                histories_tensor[history_start:history_stop]
            )
            for parameter_start in range(0, parameter_count, parameter_batch_size):
                parameter_stop = min(parameter_start + parameter_batch_size, parameter_count)
                prediction = decode_cartesian_features(
                    model,
                    history_features,
                    parameter_features[parameter_start:parameter_stop],
                    trunk_features,
                )
                local_frequency = prediction[..., NODE_COUNT:].abs().amax(dim=(2, 3))
                local_edges = torch.stack(
                    [
                        prediction[..., left] - prediction[..., right]
                        for left, right in edges
                    ],
                    dim=-1,
                )
                local_angle = local_edges.abs().amax(dim=(2, 3))
                frequency_risk[
                    history_start:history_stop, parameter_start:parameter_stop
                ] = local_frequency.cpu().numpy()
                angle_risk[
                    history_start:history_stop, parameter_start:parameter_stop
                ] = local_angle.cpu().numpy()
    return frequency_risk, angle_risk


def load_frozen_operator(checkpoint_path: Path, device: Any) -> tuple[Any, dict[str, Any]]:
    import torch
    from model import ParaFDEONet, count_parameters

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model_config = dict(checkpoint["model_config"])
    model_config.pop("network_format", None)
    model = ParaFDEONet(**model_config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    metadata = {
        "parameter_count": int(count_parameters(model)),
        "training_seed": int(checkpoint["training_seed"]),
        "iteration": int(checkpoint["iteration"]),
        "best_iteration": int(checkpoint["best_iteration"]),
        "best_validation_normalized_mse": float(
            checkpoint["best_validation_normalized_mse"]
        ),
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }
    return model, metadata


def _direct_risk_task(arguments: tuple[Any, ...]) -> tuple[int, np.ndarray, np.ndarray]:
    (
        start,
        histories,
        parameters,
        history_times,
        output_times,
        equation,
        internal_step,
    ) = arguments
    trajectories = solve_batch(
        histories,
        parameters,
        history_times,
        output_times,
        equation,
        internal_step,
    )
    frequency, angle = risk_from_trajectories(trajectories, equation.edges)
    return int(start), frequency, angle


def direct_dde_risks(
    histories: np.ndarray,
    parameters: np.ndarray,
    history_times: np.ndarray,
    output_times: np.ndarray,
    equation: SmartGridConfig,
    internal_step: float,
    processes: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    count = histories.shape[0]
    if parameters.shape[0] != count or processes < 1 or chunk_size < 1:
        raise ValueError("invalid direct-solver design")
    frequency = np.empty(count, dtype=np.float64)
    angle = np.empty(count, dtype=np.float64)
    tasks = [
        (
            start,
            histories[start : start + chunk_size],
            parameters[start : start + chunk_size],
            history_times,
            output_times,
            equation,
            internal_step,
        )
        for start in range(0, count, chunk_size)
    ]
    if processes == 1:
        for task_index, task in enumerate(tasks, start=1):
            start, local_frequency, local_angle = _direct_risk_task(task)
            stop = start + local_frequency.size
            frequency[start:stop] = local_frequency
            angle[start:stop] = local_angle
            if task_index % max(1, len(tasks) // 10) == 0 or task_index == len(tasks):
                LOGGER.info("direct DDE progress: %d/%d chunks", task_index, len(tasks))
        return frequency, angle
    workers = min(int(processes), len(tasks))
    completed = 0
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=mp.get_context("spawn")
    ) as executor:
        futures = [executor.submit(_direct_risk_task, task) for task in tasks]
        for future in as_completed(futures):
            start, local_frequency, local_angle = future.result()
            stop = start + local_frequency.size
            frequency[start:stop] = local_frequency
            angle[start:stop] = local_angle
            completed += 1
            if completed % max(1, len(tasks) // 10) == 0 or completed == len(tasks):
                LOGGER.info("direct DDE progress: %d/%d chunks", completed, len(tasks))
    return frequency, angle


def threshold_arrays(config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    settings = config["survivability"]
    frequency_hz = np.arange(
        float(settings["frequency_threshold_hz_min"]),
        float(settings["frequency_threshold_hz_max"]) + 0.5 * float(settings["frequency_threshold_hz_step"]),
        float(settings["frequency_threshold_hz_step"]),
        dtype=np.float64,
    )
    angle_degrees = np.arange(
        float(settings["edge_angle_threshold_degrees_min"]),
        float(settings["edge_angle_threshold_degrees_max"]) + 0.5 * float(settings["edge_angle_threshold_degrees_step"]),
        float(settings["edge_angle_threshold_degrees_step"]),
        dtype=np.float64,
    )
    return 2.0 * math.pi * frequency_hz, np.deg2rad(angle_degrees)


def summarize_timings(
    operator_seconds: list[float], direct_seconds: list[float], trajectory_count: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not operator_seconds or not direct_seconds:
        raise ValueError("both timing methods require observations")
    rows: list[dict[str, Any]] = []
    for method, values in (
        ("ParaFDEONet", operator_seconds),
        ("Direct DDE Monte Carlo", direct_seconds),
    ):
        for repeat, seconds in enumerate(values, start=1):
            rows.append(
                {
                    "method": method,
                    "repeat": repeat,
                    "trajectory_count": trajectory_count,
                    "seconds": float(seconds),
                    "trajectories_per_second": float(trajectory_count / seconds),
                }
            )
    operator = np.asarray(operator_seconds, dtype=np.float64)
    direct = np.asarray(direct_seconds, dtype=np.float64)
    metrics = {
        "timing_repeats": int(min(operator.size, direct.size)),
        "parafdeonet_mean_seconds": float(np.mean(operator)),
        "parafdeonet_sample_std_seconds": float(np.std(operator, ddof=1)) if operator.size > 1 else 0.0,
        "parafdeonet_mean_trajectories_per_second": float(trajectory_count / np.mean(operator)),
        "direct_dde_mean_seconds": float(np.mean(direct)),
        "direct_dde_sample_std_seconds": float(np.std(direct, ddof=1)) if direct.size > 1 else 0.0,
        "direct_dde_mean_trajectories_per_second": float(trajectory_count / np.mean(direct)),
        "speedup": float(np.mean(direct) / np.mean(operator)),
    }
    return metrics, rows


def analyze_results(
    output_dir: Path,
    config: Mapping[str, Any],
    history_indices: np.ndarray,
    parameters: np.ndarray,
    frequency_thresholds: np.ndarray,
    angle_thresholds: np.ndarray,
    operator_frequency: np.ndarray,
    operator_angle: np.ndarray,
    direct_frequency: np.ndarray,
    direct_angle: np.ndarray,
    operator_seconds: list[float],
    direct_seconds: list[float],
    model_metadata: Mapping[str, Any],
    convergence_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    history_count, parameter_count = direct_frequency.shape
    direct_surface = survivability_surface(
        direct_frequency, direct_angle, frequency_thresholds, angle_thresholds
    )
    operator_surface = survivability_surface(
        operator_frequency, operator_angle, frequency_thresholds, angle_thresholds
    )
    surface_error = operator_surface - direct_surface
    representative_frequency = 2.0 * math.pi * float(
        config["survivability"]["representative_frequency_threshold_hz"]
    )
    representative_angle = math.radians(
        float(config["survivability"]["representative_edge_angle_threshold_degrees"])
    )
    frequency_index = int(np.argmin(np.abs(frequency_thresholds - representative_frequency)))
    angle_index = int(np.argmin(np.abs(angle_thresholds - representative_angle)))
    if not math.isclose(
        float(frequency_thresholds[frequency_index]), representative_frequency, abs_tol=1e-10
    ) or not math.isclose(
        float(angle_thresholds[angle_index]), representative_angle, abs_tol=1e-10
    ):
        raise RuntimeError("representative screening limits are absent from threshold grids")
    direct_representative = direct_surface[:, frequency_index, angle_index]
    operator_representative = operator_surface[:, frequency_index, angle_index]
    wilson_lower, wilson_upper = wilson_interval(direct_representative, history_count)
    direct_safe = (direct_frequency <= representative_frequency) & (
        direct_angle <= representative_angle
    )
    operator_safe = (operator_frequency <= representative_frequency) & (
        operator_angle <= representative_angle
    )
    true_safe = int(np.sum(direct_safe))
    true_unsafe = int(direct_safe.size - true_safe)
    false_safe = int(np.sum(operator_safe & ~direct_safe))
    false_alarm = int(np.sum(~operator_safe & direct_safe))
    q95_direct_frequency = np.quantile(direct_frequency, 0.95, axis=0)
    q95_operator_frequency = np.quantile(operator_frequency, 0.95, axis=0)
    q95_direct_angle = np.quantile(direct_angle, 0.95, axis=0)
    q95_operator_angle = np.quantile(operator_angle, 0.95, axis=0)
    top_count = max(1, int(math.ceil(0.10 * parameter_count)))
    direct_high_risk = set(np.argsort(direct_representative, kind="mergesort")[:top_count])
    operator_high_risk = set(np.argsort(operator_representative, kind="mergesort")[:top_count])
    timing_metrics, timing_rows = summarize_timings(
        operator_seconds, direct_seconds, direct_safe.size
    )
    qomega_error = operator_frequency - direct_frequency
    qtheta_error = operator_angle - direct_angle
    metrics: dict[str, Any] = {
        "experiment_name": config["experiment_name"],
        "baseline": config["baseline"],
        "frozen_model": dict(model_metadata),
        "design": {
            "history_count": history_count,
            "parameter_count": parameter_count,
            "trajectory_count": int(history_count * parameter_count),
            "frequency_threshold_count": int(frequency_thresholds.size),
            "angle_threshold_count": int(angle_thresholds.size),
            "survivability_cell_count": int(parameter_count * frequency_thresholds.size * angle_thresholds.size),
        },
        "survivability_surface": {
            "mae": float(np.mean(np.abs(surface_error))),
            "rmse": float(np.sqrt(np.mean(surface_error**2))),
            "maximum_absolute_error": float(np.max(np.abs(surface_error))),
            "mean_surface_mae": float(
                np.mean(np.abs(np.mean(operator_surface, axis=0) - np.mean(direct_surface, axis=0)))
            ),
            "fraction_operator_estimates_inside_direct_wilson_95": float(
                np.mean(
                    (operator_surface >= wilson_interval(direct_surface, history_count)[0])
                    & (operator_surface <= wilson_interval(direct_surface, history_count)[1])
                )
            ),
        },
        "representative_screening": {
            "frequency_threshold_hz": float(representative_frequency / (2.0 * math.pi)),
            "frequency_threshold_rad_per_s": float(representative_frequency),
            "edge_angle_threshold_degrees": float(math.degrees(representative_angle)),
            "edge_angle_threshold_rad": float(representative_angle),
            "direct_overall_survivability": float(np.mean(direct_safe)),
            "parafdeonet_overall_survivability": float(np.mean(operator_safe)),
            "parameterwise_probability_mae": float(
                np.mean(np.abs(operator_representative - direct_representative))
            ),
            "parameterwise_probability_rmse": float(
                np.sqrt(np.mean((operator_representative - direct_representative) ** 2))
            ),
            "parameterwise_probability_maximum_absolute_error": float(
                np.max(np.abs(operator_representative - direct_representative))
            ),
            "parameterwise_spearman": spearman_correlation(
                direct_representative, operator_representative
            ),
            "lowest_survivability_10_percent_recall": float(
                len(direct_high_risk & operator_high_risk) / top_count
            ),
            "casewise_accuracy": float(np.mean(operator_safe == direct_safe)),
            "false_safe_count": false_safe,
            "false_safe_rate_given_direct_unsafe": float(false_safe / max(true_unsafe, 1)),
            "false_alarm_count": false_alarm,
            "false_alarm_rate_given_direct_safe": float(false_alarm / max(true_safe, 1)),
        },
        "continuous_risk": {
            "frequency_mae_rad_per_s": float(np.mean(np.abs(qomega_error))),
            "frequency_mae_hz": float(np.mean(np.abs(qomega_error)) / (2.0 * math.pi)),
            "frequency_p99_absolute_error_rad_per_s": float(np.quantile(np.abs(qomega_error), 0.99)),
            "frequency_p99_absolute_error_hz": float(np.quantile(np.abs(qomega_error), 0.99) / (2.0 * math.pi)),
            "frequency_q95_parameterwise_spearman": spearman_correlation(
                q95_direct_frequency, q95_operator_frequency
            ),
            "edge_angle_mae_rad": float(np.mean(np.abs(qtheta_error))),
            "edge_angle_mae_degrees": float(np.degrees(np.mean(np.abs(qtheta_error)))),
            "edge_angle_p99_absolute_error_rad": float(np.quantile(np.abs(qtheta_error), 0.99)),
            "edge_angle_p99_absolute_error_degrees": float(
                np.degrees(np.quantile(np.abs(qtheta_error), 0.99))
            ),
            "edge_angle_q95_parameterwise_spearman": spearman_correlation(
                q95_direct_angle, q95_operator_angle
            ),
        },
        "timing": timing_metrics,
        "direct_solver_convergence_audit": dict(convergence_metrics),
    }
    parameter_rows: list[dict[str, Any]] = []
    for index in range(parameter_count):
        parameter_rows.append(
            {
                "parameter_index": index,
                "coupling_K": float(parameters[index, 0]),
                "response_gamma": float(parameters[index, 1]),
                "damping_alpha": float(parameters[index, 2]),
                "direct_survivability": float(direct_representative[index]),
                "direct_wilson_95_lower": float(wilson_lower[index]),
                "direct_wilson_95_upper": float(wilson_upper[index]),
                "parafdeonet_survivability": float(operator_representative[index]),
                "survivability_error": float(
                    operator_representative[index] - direct_representative[index]
                ),
                "direct_frequency_q95_hz": float(q95_direct_frequency[index] / (2.0 * math.pi)),
                "parafdeonet_frequency_q95_hz": float(q95_operator_frequency[index] / (2.0 * math.pi)),
                "direct_edge_angle_q95_degrees": float(np.degrees(q95_direct_angle[index])),
                "parafdeonet_edge_angle_q95_degrees": float(np.degrees(q95_operator_angle[index])),
            }
        )
    case_rows: list[dict[str, Any]] = []
    for history_local, history_index in enumerate(history_indices):
        for parameter_index in range(parameter_count):
            case_rows.append(
                {
                    "test_history_index": int(history_index),
                    "parameter_index": parameter_index,
                    "coupling_K": float(parameters[parameter_index, 0]),
                    "response_gamma": float(parameters[parameter_index, 1]),
                    "damping_alpha": float(parameters[parameter_index, 2]),
                    "direct_frequency_risk_rad_per_s": float(direct_frequency[history_local, parameter_index]),
                    "parafdeonet_frequency_risk_rad_per_s": float(operator_frequency[history_local, parameter_index]),
                    "direct_edge_angle_risk_rad": float(direct_angle[history_local, parameter_index]),
                    "parafdeonet_edge_angle_risk_rad": float(operator_angle[history_local, parameter_index]),
                    "direct_safe_representative": int(direct_safe[history_local, parameter_index]),
                    "parafdeonet_safe_representative": int(operator_safe[history_local, parameter_index]),
                }
            )
    surface_rows: list[dict[str, Any]] = []
    mean_direct_surface = np.mean(direct_surface, axis=0)
    mean_operator_surface = np.mean(operator_surface, axis=0)
    for frequency_position, frequency_value in enumerate(frequency_thresholds):
        for angle_position, angle_value in enumerate(angle_thresholds):
            surface_rows.append(
                {
                    "frequency_threshold_hz": float(frequency_value / (2.0 * math.pi)),
                    "frequency_threshold_rad_per_s": float(frequency_value),
                    "edge_angle_threshold_degrees": float(np.degrees(angle_value)),
                    "edge_angle_threshold_rad": float(angle_value),
                    "direct_mean_survivability": float(
                        mean_direct_surface[frequency_position, angle_position]
                    ),
                    "parafdeonet_mean_survivability": float(
                        mean_operator_surface[frequency_position, angle_position]
                    ),
                    "absolute_error": float(
                        abs(
                            mean_operator_surface[frequency_position, angle_position]
                            - mean_direct_surface[frequency_position, angle_position]
                        )
                    ),
                }
            )
    save_json(output_dir / "metrics.json", metrics)
    save_csv(output_dir / "source_data_parameters.csv", parameter_rows)
    save_csv(output_dir / "source_data_cases.csv", case_rows)
    save_csv(output_dir / "source_data_surfaces.csv", surface_rows)
    save_csv(output_dir / "timing_runs.csv", timing_rows)
    save_npz(
        output_dir / "survivability_data.npz",
        frequency_thresholds_rad_per_s=frequency_thresholds,
        edge_angle_thresholds_rad=angle_thresholds,
        direct_survivability=direct_surface,
        parafdeonet_survivability=operator_surface,
        direct_mean_survivability=mean_direct_surface,
        parafdeonet_mean_survivability=mean_operator_surface,
        direct_representative_survivability=direct_representative,
        parafdeonet_representative_survivability=operator_representative,
        direct_representative_wilson_lower=wilson_lower,
        direct_representative_wilson_upper=wilson_upper,
        q95_direct_frequency_rad_per_s=q95_direct_frequency,
        q95_parafdeonet_frequency_rad_per_s=q95_operator_frequency,
        q95_direct_edge_angle_rad=q95_direct_angle,
        q95_parafdeonet_edge_angle_rad=q95_operator_angle,
        parameters=parameters,
    )
    return metrics


def write_results_markdown(output_dir: Path, metrics: Mapping[str, Any]) -> None:
    surface = metrics["survivability_surface"]
    representative = metrics["representative_screening"]
    risk = metrics["continuous_risk"]
    timing = metrics["timing"]
    audit = metrics["direct_solver_convergence_audit"]
    text = f"""# Probabilistic survivability experiment

## Design

The frozen 75,223,048-parameter ParaFDEONet at iteration 76,500 is evaluated on 512 held-out physical histories crossed with a 4 x 4 x 4 factorial grid in $(K, \\gamma, \\alpha)$, giving 32,768 trajectories. The numerical baseline is direct DDE Monte Carlo survivability following Hellmann et al. (2016), implemented with causal fixed-step RK4 at $\\Delta t=0.005$ s. Every comparison uses the same histories and parameter points.

Survivability is evaluated over a fixed grid of maximum nodal-frequency and maximum star-edge angle-difference limits. The representative screening slice is {representative['frequency_threshold_hz']:.2f} Hz and {representative['edge_angle_threshold_degrees']:.0f} degrees; it is a numerical screening boundary rather than a claimed grid-code standard.

## Main results

- Full parameter-threshold survivability MAE: {surface['mae']:.6f}, equal to {100.0 * surface['mae']:.4f} percentage points.
- Full survivability RMSE / maximum absolute error: {100.0 * surface['rmse']:.4f} / {100.0 * surface['maximum_absolute_error']:.4f} percentage points.
- Representative direct / ParaFDEONet overall survivability: {100.0 * representative['direct_overall_survivability']:.4f}% / {100.0 * representative['parafdeonet_overall_survivability']:.4f}%.
- Representative parameterwise probability MAE: {100.0 * representative['parameterwise_probability_mae']:.4f} percentage points.
- Representative casewise classification accuracy: {100.0 * representative['casewise_accuracy']:.4f}%; false-safe rate conditional on direct-DDE failure: {100.0 * representative['false_safe_rate_given_direct_unsafe']:.4f}%.
- Frequency-risk MAE: {risk['frequency_mae_hz']:.6e} Hz; P99 absolute error: {risk['frequency_p99_absolute_error_hz']:.6e} Hz.
- Edge-angle-risk MAE: {risk['edge_angle_mae_degrees']:.6e} degrees; P99 absolute error: {risk['edge_angle_p99_absolute_error_degrees']:.6e} degrees.
- Mean deployment time for 32,768 trajectories: ParaFDEONet {timing['parafdeonet_mean_seconds']:.6f} s, direct DDE Monte Carlo {timing['direct_dde_mean_seconds']:.6f} s.
- End-to-end deployment speedup: {timing['speedup']:.2f} x. Model and dataset loading are excluded for both methods; branch encoding and device transfer are included for ParaFDEONet.
- Refined-step DDE audit maximum frequency-risk / angle-risk difference: {audit['frequency_maximum_absolute_difference_rad_per_s']:.6e} rad/s / {audit['edge_angle_maximum_absolute_difference_rad']:.6e} rad.

## Scope

This experiment supports the claim that the frozen ParaFDEONet can reproduce direct-DDE Monte Carlo survivability estimates for this specified delayed four-node grid and history distribution at lower deployment cost. It does not establish calibrated reliability for a real power system, because the histories and parameter ranges are synthetic numerical designs and only one trained network seed is used.

## Baseline reference

Hellmann, F. et al. Survivability of deterministic dynamical systems. *Scientific Reports* **6**, 29654 (2016). https://doi.org/10.1038/srep29654
"""
    (output_dir / "RESULTS.md").write_text(text, encoding="utf-8")


def convergence_audit(
    pair_histories: np.ndarray,
    pair_parameters: np.ndarray,
    direct_frequency_flat: np.ndarray,
    direct_angle_flat: np.ndarray,
    history_times: np.ndarray,
    output_times: np.ndarray,
    equation: SmartGridConfig,
    config: Mapping[str, Any],
    processes: int,
    chunk_size: int,
) -> dict[str, Any]:
    baseline = config["baseline"]
    count = min(int(baseline["refined_audit_case_count"]), pair_histories.shape[0])
    indices = np.linspace(0, pair_histories.shape[0] - 1, count, dtype=np.int64)
    LOGGER.info("running refined-step audit on %d deterministic pairs", count)
    start = time.perf_counter()
    fine_frequency, fine_angle = direct_dde_risks(
        pair_histories[indices],
        pair_parameters[indices],
        history_times,
        output_times,
        equation,
        float(baseline["refined_audit_step_seconds"]),
        min(processes, count),
        min(chunk_size, count),
    )
    elapsed = time.perf_counter() - start
    coarse_frequency = direct_frequency_flat[indices]
    coarse_angle = direct_angle_flat[indices]
    representative_frequency = 2.0 * math.pi * float(
        config["survivability"]["representative_frequency_threshold_hz"]
    )
    representative_angle = math.radians(
        float(config["survivability"]["representative_edge_angle_threshold_degrees"])
    )
    coarse_safe = (coarse_frequency <= representative_frequency) & (
        coarse_angle <= representative_angle
    )
    fine_safe = (fine_frequency <= representative_frequency) & (
        fine_angle <= representative_angle
    )
    return {
        "case_count": count,
        "coarse_step_seconds": float(baseline["internal_step_seconds"]),
        "refined_step_seconds": float(baseline["refined_audit_step_seconds"]),
        "runtime_seconds": float(elapsed),
        "frequency_mae_rad_per_s": float(np.mean(np.abs(coarse_frequency - fine_frequency))),
        "frequency_maximum_absolute_difference_rad_per_s": float(
            np.max(np.abs(coarse_frequency - fine_frequency))
        ),
        "edge_angle_mae_rad": float(np.mean(np.abs(coarse_angle - fine_angle))),
        "edge_angle_maximum_absolute_difference_rad": float(
            np.max(np.abs(coarse_angle - fine_angle))
        ),
        "representative_classification_agreement": float(np.mean(coarse_safe == fine_safe)),
    }


def validate_configuration(
    config: Mapping[str, Any], model_metadata: Mapping[str, Any], histories: np.ndarray,
    parameters: np.ndarray, output_times: np.ndarray
) -> None:
    frozen = config["frozen_model"]
    design = config["design"]
    checks = {
        "parameter_count": (model_metadata["parameter_count"], frozen["expected_parameter_count"]),
        "training_seed": (model_metadata["training_seed"], frozen["expected_training_seed"]),
        "best_iteration": (model_metadata["best_iteration"], frozen["expected_best_iteration"]),
        "history_count": (histories.shape[0], design["history_count"]),
        "trajectory_count": (histories.shape[0] * parameters.shape[0], design["trajectory_count"]),
        "output_points": (output_times.size, design["output_points"]),
    }
    mismatches = [f"{name}: observed {actual}, expected {expected}" for name, (actual, expected) in checks.items() if int(actual) != int(expected)]
    if mismatches:
        raise RuntimeError("configuration mismatch: " + "; ".join(mismatches))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_DIR / "configs" / "survivability_experiment.json"
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--timing-repeats", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir or args.run_dir / "full" / "application_survivability"
    configure_logging(output_dir)
    runtime = config["runtime"]
    device_name = args.device or str(runtime["device"])
    processes = int(args.processes or runtime["direct_processes"])
    chunk_size = int(args.chunk_size or runtime["direct_chunk_size"])
    timing_repeats = int(args.timing_repeats or runtime["timing_repeats"])
    if timing_repeats < 1:
        raise ValueError("timing repeats must be positive")
    complete_marker = output_dir / "COMPLETE"
    if complete_marker.exists() and not args.resume:
        raise FileExistsError(f"completed output already exists: {output_dir}")
    save_json(output_dir / "status.json", {"status": "initializing", "updated_unix": time.time()})
    equation_config = json.loads((PROJECT_DIR / "configs" / "experiment.json").read_text(encoding="utf-8"))
    equation = SmartGridConfig.from_mapping(equation_config["equation"])
    dataset_path = args.run_dir / config["design"]["dataset_relative_to_run"]
    checkpoint_path = args.run_dir / config["frozen_model"]["checkpoint_relative_to_run"]
    if not dataset_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(f"missing dataset or checkpoint under {args.run_dir}")
    with np.load(dataset_path) as dataset:
        available_histories = int(dataset["histories"].shape[0])
        history_count = int(config["design"]["history_count"])
        if history_count > available_histories:
            raise ValueError("requested more histories than the fixed test split contains")
        history_indices = np.linspace(
            0, available_histories - 1, history_count, dtype=np.int64
        )
        if np.unique(history_indices).size != history_count:
            raise RuntimeError("history selection contains duplicates")
        histories = dataset["histories"][history_indices].astype(np.float32)
        history_times = dataset["history_times"].astype(np.float64)
        output_times = dataset["output_times"].astype(np.float64)
    parameters = parameter_grid(
        equation, int(config["design"]["parameter_levels_per_dimension"])
    )
    frequency_thresholds, angle_thresholds = threshold_arrays(config)
    save_npz(
        output_dir / "design.npz",
        selected_test_history_indices=history_indices,
        parameter_grid=parameters,
        frequency_thresholds_rad_per_s=frequency_thresholds,
        edge_angle_thresholds_rad=angle_thresholds,
        history_times=history_times,
        output_times=output_times,
    )
    LOGGER.info(
        "design: %d histories x %d parameters = %d trajectories",
        histories.shape[0], parameters.shape[0], histories.shape[0] * parameters.shape[0]
    )
    import torch

    if not torch.cuda.is_available() and device_name.startswith("cuda"):
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    torch.set_float32_matmul_precision("highest")
    model, model_metadata = load_frozen_operator(checkpoint_path, device)
    validate_configuration(config, model_metadata, histories, parameters, output_times)
    LOGGER.info(
        "loaded frozen checkpoint: %d parameters, best iteration %d",
        model_metadata["parameter_count"], model_metadata["best_iteration"]
    )
    resolved = dict(config)
    resolved["resolved_paths"] = {
        "run_dir": str(args.run_dir.resolve()),
        "dataset": str(dataset_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "output_dir": str(output_dir.resolve()),
    }
    resolved["resolved_runtime"] = {
        "device": device_name,
        "processes": processes,
        "chunk_size": chunk_size,
        "timing_repeats": timing_repeats,
    }
    resolved["model_metadata"] = model_metadata
    save_json(output_dir / "config_resolved.json", resolved)
    timing_progress_path = output_dir / "timing_progress.json"
    if timing_progress_path.exists() and args.resume:
        timing_progress = json.loads(timing_progress_path.read_text(encoding="utf-8"))
    else:
        timing_progress = {"parafdeonet_seconds": [], "direct_dde_seconds": []}
    operator_risk_path = output_dir / "parafdeonet_risks.npz"
    operator_seconds = [float(item) for item in timing_progress["parafdeonet_seconds"]]
    if len(operator_seconds) < timing_repeats:
        LOGGER.info("warming up cached ParaFDEONet deployment")
        predict_operator_risks(
            model, histories, parameters, output_times, equation.edges, device,
            int(runtime["operator_history_batch_size"]),
            int(runtime["operator_parameter_batch_size"]),
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        first_operator: tuple[np.ndarray, np.ndarray] | None = None
        for repeat in range(len(operator_seconds), timing_repeats):
            start = time.perf_counter()
            local_frequency, local_angle = predict_operator_risks(
                model, histories, parameters, output_times, equation.edges, device,
                int(runtime["operator_history_batch_size"]),
                int(runtime["operator_parameter_batch_size"]),
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            operator_seconds.append(float(elapsed))
            if first_operator is None:
                first_operator = (local_frequency, local_angle)
            LOGGER.info("ParaFDEONet timing repeat %d/%d: %.6f s", repeat + 1, timing_repeats, elapsed)
            timing_progress["parafdeonet_seconds"] = operator_seconds
            save_json(timing_progress_path, timing_progress)
        if not operator_risk_path.exists():
            if first_operator is None:
                raise RuntimeError("operator timing produced no risk values")
            save_npz(
                operator_risk_path,
                frequency_risk_rad_per_s=first_operator[0],
                edge_angle_risk_rad=first_operator[1],
            )
    with np.load(operator_risk_path) as values:
        operator_frequency = values["frequency_risk_rad_per_s"].astype(np.float64)
        operator_angle = values["edge_angle_risk_rad"].astype(np.float64)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    pair_histories = np.repeat(histories, parameters.shape[0], axis=0)
    pair_parameters = np.tile(parameters, (histories.shape[0], 1))
    direct_risk_path = output_dir / "direct_dde_risks.npz"
    direct_seconds = [float(item) for item in timing_progress["direct_dde_seconds"]]
    first_direct: tuple[np.ndarray, np.ndarray] | None = None
    for repeat in range(len(direct_seconds), timing_repeats):
        LOGGER.info("starting direct DDE timing repeat %d/%d", repeat + 1, timing_repeats)
        start = time.perf_counter()
        local_frequency_flat, local_angle_flat = direct_dde_risks(
            pair_histories,
            pair_parameters,
            history_times,
            output_times,
            equation,
            float(config["baseline"]["internal_step_seconds"]),
            processes,
            chunk_size,
        )
        elapsed = time.perf_counter() - start
        direct_seconds.append(float(elapsed))
        if first_direct is None:
            first_direct = (
                local_frequency_flat.reshape(histories.shape[0], parameters.shape[0]),
                local_angle_flat.reshape(histories.shape[0], parameters.shape[0]),
            )
        elif direct_risk_path.exists():
            with np.load(direct_risk_path) as saved:
                np.testing.assert_allclose(
                    local_frequency_flat.reshape(histories.shape[0], parameters.shape[0]),
                    saved["frequency_risk_rad_per_s"], atol=0.0, rtol=0.0
                )
                np.testing.assert_allclose(
                    local_angle_flat.reshape(histories.shape[0], parameters.shape[0]),
                    saved["edge_angle_risk_rad"], atol=0.0, rtol=0.0
                )
        LOGGER.info("direct DDE timing repeat %d/%d: %.6f s", repeat + 1, timing_repeats, elapsed)
        if not direct_risk_path.exists() and first_direct is not None:
            save_npz(
                direct_risk_path,
                frequency_risk_rad_per_s=first_direct[0],
                edge_angle_risk_rad=first_direct[1],
            )
        timing_progress["direct_dde_seconds"] = direct_seconds
        save_json(timing_progress_path, timing_progress)
    with np.load(direct_risk_path) as values:
        direct_frequency = values["frequency_risk_rad_per_s"].astype(np.float64)
        direct_angle = values["edge_angle_risk_rad"].astype(np.float64)
    audit_path = output_dir / "direct_solver_convergence_audit.json"
    if audit_path.exists() and args.resume:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    else:
        audit = convergence_audit(
            pair_histories,
            pair_parameters,
            direct_frequency.reshape(-1),
            direct_angle.reshape(-1),
            history_times,
            output_times,
            equation,
            config,
            processes,
            chunk_size,
        )
        save_json(audit_path, audit)
    metrics = analyze_results(
        output_dir,
        config,
        history_indices,
        parameters,
        frequency_thresholds,
        angle_thresholds,
        operator_frequency,
        operator_angle,
        direct_frequency,
        direct_angle,
        operator_seconds[:timing_repeats],
        direct_seconds[:timing_repeats],
        model_metadata,
        audit,
    )
    write_results_markdown(output_dir, metrics)
    from scripts.plot_survivability import create_figure

    figure_paths = create_figure(output_dir, config)
    save_json(
        output_dir / "status.json",
        {
            "status": "complete",
            "updated_unix": time.time(),
            "figure_paths": [str(path) for path in figure_paths],
            "metrics_path": str(output_dir / "metrics.json"),
        },
    )
    complete_marker.write_text("complete\n", encoding="utf-8")
    LOGGER.info("experiment complete: %s", output_dir)


if __name__ == "__main__":
    main()
