"""Direct causal-solver projected Levenberg--Marquardt baseline."""

from __future__ import annotations

import math
from pathlib import Path
import platform
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np

from common.io_utils import (
    read_json, sha256_file, sha256_mapping, write_csv, write_json,
)
from common.metrics import evaluate_estimate
from common.panel_archive import load_panel, select_panel_histories
from common.parameters import latin_hypercube
from systems import build_adapter


def _cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _observations(
    panel: Mapping[str, np.ndarray], case_index: int, sigma: float
) -> np.ndarray:
    indices = np.asarray(panel["observation_indices"], dtype=np.int64)
    reference = np.asarray(panel["reference"][case_index][:, indices, :])
    noise = np.asarray(panel["standard_normal_noise"][case_index])
    return (reference + float(sigma) * noise).astype(np.float64)


def _direct_residuals(
    adapter: Any,
    candidates: np.ndarray,
    panel: Mapping[str, np.ndarray],
    observations: np.ndarray,
    internal_step: float,
) -> tuple[np.ndarray, np.ndarray]:
    candidates = np.asarray(candidates, dtype=np.float64)
    histories = np.asarray(panel["histories"], dtype=np.float64)
    history_count = histories.shape[0]
    repeated_histories = np.tile(histories, (candidates.shape[0], 1, 1))
    repeated_parameters = np.repeat(candidates, history_count, axis=0)
    indices = np.asarray(panel["observation_indices"], dtype=np.int64)
    observation_times = np.asarray(panel["output_times"], dtype=np.float64)[indices]
    predicted = adapter.solve_batch(
        repeated_histories,
        repeated_parameters,
        np.asarray(panel["history_grid"], dtype=np.float64),
        observation_times,
        float(internal_step),
    ).reshape(
        candidates.shape[0],
        history_count,
        observation_times.size,
        adapter.state_dim,
    )
    residuals = (predicted.astype(np.float64) - observations[None]).reshape(
        candidates.shape[0], -1
    )
    if not np.isfinite(residuals).all():
        raise FloatingPointError("non-finite Direct LM residual")
    return residuals, predicted


def _initial_points(
    adapter: Any, count: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    unit = latin_hypercube(count, 2, rng)
    return adapter.unit_to_physical_numpy(unit)


def invert_lm(
    adapter: Any,
    panel: Mapping[str, np.ndarray],
    observations: np.ndarray,
    config: Mapping[str, Any],
    seed: int,
) -> tuple[np.ndarray, float, list[dict[str, Any]], dict[str, Any], np.ndarray]:
    restart_count = int(config["starts"])
    parameters = _initial_points(adapter, restart_count, seed)
    bounds = adapter.parameter_bounds
    damping = np.full(
        restart_count, float(config["initial_damping"]), dtype=np.float64
    )
    finite_step = float(config["finite_difference_step"])
    internal_step = float(config["internal_step"])
    minimum_iterations = int(config["min_iterations"])
    maximum_iterations = int(config["max_iterations"])
    tolerance = float(config["relative_objective_tolerance"])
    patience = int(config["patience"])
    trace: list[dict[str, Any]] = []
    stable_count = 0
    previous_best: float | None = None
    solver_batch_calls = 0
    trajectory_solves = 0
    history_count = int(panel["histories"].shape[0])

    started = time.perf_counter()
    stopped_early = False
    completed = 0
    for iteration in range(1, maximum_iterations + 1):
        residuals, _ = _direct_residuals(
            adapter, parameters, panel, observations, internal_step
        )
        solver_batch_calls += 1
        trajectory_solves += restart_count * history_count
        objectives = np.sum(residuals**2, axis=1)

        perturbed = np.repeat(parameters[None, :, :], 4, axis=0)
        for parameter_index in range(2):
            perturbed[2 * parameter_index, :, parameter_index] = np.minimum(
                bounds[parameter_index, 1],
                parameters[:, parameter_index] + finite_step,
            )
            perturbed[2 * parameter_index + 1, :, parameter_index] = np.maximum(
                bounds[parameter_index, 0],
                parameters[:, parameter_index] - finite_step,
            )
        perturbed_residuals, _ = _direct_residuals(
            adapter,
            perturbed.reshape(4 * restart_count, 2),
            panel,
            observations,
            internal_step,
        )
        solver_batch_calls += 1
        trajectory_solves += 4 * restart_count * history_count
        perturbed_residuals = perturbed_residuals.reshape(
            4, restart_count, residuals.shape[1]
        )
        jacobians = np.empty(
            (restart_count, residuals.shape[1], 2), dtype=np.float64
        )
        for parameter_index in range(2):
            denominator = (
                perturbed[2 * parameter_index, :, parameter_index]
                - perturbed[2 * parameter_index + 1, :, parameter_index]
            )
            if np.any(denominator <= 0.0):
                raise FloatingPointError("zero finite-difference denominator")
            jacobians[:, :, parameter_index] = (
                perturbed_residuals[2 * parameter_index]
                - perturbed_residuals[2 * parameter_index + 1]
            ) / denominator[:, None]

        proposed = parameters.copy()
        for restart in range(restart_count):
            jacobian = jacobians[restart]
            normal = jacobian.T @ jacobian
            scaling = np.maximum(np.diag(normal), 1.0e-12)
            system = normal + damping[restart] * np.diag(scaling)
            gradient = jacobian.T @ residuals[restart]
            try:
                delta = np.linalg.solve(system, -gradient)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(system, -gradient, rcond=None)[0]
            proposed[restart] = np.clip(
                parameters[restart] + delta, bounds[:, 0], bounds[:, 1]
            )
        proposed_residuals, _ = _direct_residuals(
            adapter, proposed, panel, observations, internal_step
        )
        solver_batch_calls += 1
        trajectory_solves += restart_count * history_count
        proposed_objectives = np.sum(proposed_residuals**2, axis=1)
        accepted = proposed_objectives < objectives
        parameters[accepted] = proposed[accepted]
        objectives[accepted] = proposed_objectives[accepted]
        damping[accepted] = np.maximum(
            float(config["minimum_damping"]),
            damping[accepted] / float(config["damping_decrease"]),
        )
        damping[~accepted] = np.minimum(
            float(config["maximum_damping"]),
            damping[~accepted] * float(config["damping_increase"]),
        )
        completed = iteration
        selected = int(np.argmin(objectives))
        best = float(objectives[selected])
        relative_change = float("inf")
        if previous_best is not None:
            relative_change = abs(best - previous_best) / max(abs(previous_best), 1e-12)
            if iteration >= minimum_iterations and relative_change < tolerance:
                stable_count += 1
            else:
                stable_count = 0
        previous_best = best
        trace.append(
            {
                "iteration": int(iteration),
                "selected_restart": int(selected),
                "selected_objective_sum_squares": float(best),
                "selected_parameter_0": float(parameters[selected, 0]),
                "selected_parameter_1": float(parameters[selected, 1]),
                "mean_damping": float(np.mean(damping)),
                "relative_objective_change": float(relative_change),
                "consecutive_stable_iterations": int(stable_count),
                "elapsed_seconds": float(time.perf_counter() - started),
            }
        )
        if stable_count >= patience:
            stopped_early = True
            break

    final_residuals, _ = _direct_residuals(
        adapter, parameters, panel, observations, internal_step
    )
    solver_batch_calls += 1
    trajectory_solves += restart_count * history_count
    objectives = np.sum(final_residuals**2, axis=1)
    selected = int(np.argmin(objectives))
    elapsed = time.perf_counter() - started
    scalar_count = int(final_residuals.shape[1])
    timing = {
        "warm_online_seconds": float(elapsed),
        "lm_iterations_completed": int(completed),
        "lm_stopped_early": int(stopped_early),
        "lm_stop_reason": (
            "relative_objective_stable" if stopped_early else "maximum_iterations"
        ),
        "solver_batch_calls": int(solver_batch_calls),
        "direct_trajectory_solves": int(trajectory_solves),
        "lm_final_damping_mean": float(np.mean(damping)),
    }
    return (
        parameters[selected].copy(),
        float(objectives[selected] / max(scalar_count, 1)),
        trace,
        timing,
        parameters,
    )


def _result_path(panel_dir: Path, case_index: int, sigma: float) -> Path:
    noise_tag = str(float(sigma)).replace(".", "p")
    return panel_dir / "jobs" / "lm" / f"case_{case_index:03d}" / f"noise_{noise_tag}" / "result.json"


def run_lm_panel(
    panel_index: int,
    system_key: str,
    system_config: Mapping[str, Any],
    common_config: Mapping[str, Any],
    benchmark_root_text: str,
    source_root_text: str | None,
    archive_dir_text: str,
    full_dir_text: str,
    lm_config: Mapping[str, Any],
    histories_per_request: int,
    resume: bool,
) -> list[dict[str, Any]]:
    print(f"[lm] starting panel {panel_index:03d}", flush=True)
    adapter = build_adapter(
        system_key,
        system_config,
        Path(benchmark_root_text),
        Path(source_root_text) if source_root_text else None,
    )
    archive_dir = Path(archive_dir_text)
    full_dir = Path(full_dir_text)
    manifest = read_json(archive_dir / "manifest.json")
    panel = select_panel_histories(
        load_panel(archive_dir, panel_index), histories_per_request
    )
    panel_dir = full_dir / f"panel_{panel_index:03d}"
    panel_dir.mkdir(parents=True, exist_ok=True)
    thresholds = common_config["reporting"]["parameter_error_thresholds"]
    boundary_tolerance = float(
        common_config["reporting"]["boundary_tolerance_normalized"]
    )
    panel_sha = sha256_file(archive_dir / f"panel_{panel_index:03d}.npz")
    inverse_config_sha256 = sha256_mapping(lm_config)
    rows: list[dict[str, Any]] = []
    request_index = 0
    for case_index in range(int(manifest["cases_per_panel"])):
        initialization_seed = (
            int(common_config["data"]["seed"])
            + panel_index * 100_003
            + case_index * 101
        )
        for sigma in manifest["noise_standard_deviations"]:
            result_path = _result_path(panel_dir, case_index, float(sigma))
            if resume and result_path.is_file():
                result = read_json(result_path)
                expected = {
                    "protocol_version": common_config["protocol_version"],
                    "system_key": system_key,
                    "method_key": "lm",
                    "panel_index": int(panel_index),
                    "case_index": int(case_index),
                    "noise_standard_deviation": float(sigma),
                    "panel_archive_sha256": panel_sha,
                    "inverse_config_sha256": inverse_config_sha256,
                    "history_count": int(histories_per_request),
                }
                if any(result.get(key) != value for key, value in expected.items()):
                    raise RuntimeError(f"incompatible completed LM result: {result_path}")
                rows.append(result)
                request_index += 1
                continue
            result_path.parent.mkdir(parents=True, exist_ok=True)
            observations = _observations(panel, case_index, float(sigma))
            try:
                estimate, objective, trace, timing, candidates = invert_lm(
                    adapter, panel, observations, lm_config, initialization_seed
                )
                metrics = evaluate_estimate(
                    adapter,
                    panel,
                    case_index,
                    estimate,
                    float(manifest["truth_internal_step"]),
                    thresholds,
                    boundary_tolerance,
                )
                status = "completed"
                failure = None
            except Exception as error:
                status = "failed"
                failure = {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                }
                trace = []
                candidates = np.empty((0, 2), dtype=np.float64)
                objective = float("nan")
                timing = {
                    "warm_online_seconds": float("nan"),
                    "lm_iterations_completed": 0,
                    "lm_stopped_early": 0,
                    "lm_stop_reason": "numerical_failure",
                }
                metrics = {
                    "true_parameters": np.asarray(
                        panel["parameters"][case_index], dtype=np.float64
                    ).tolist(),
                    "estimated_parameters": [float("nan"), float("nan")],
                    "parameter_normalized_rmse": float("nan"),
                    "parameter_physical_rmse": float("nan"),
                    "trajectory_relative_l2": float("nan"),
                    "boundary_estimate": 0,
                }
            result = {
                "protocol_version": common_config["protocol_version"],
                "system_key": system_key,
                "method_key": "lm",
                "status": status,
                "failure": failure,
                "panel_index": int(panel_index),
                "case_index": int(case_index),
                "request_index_within_panel": int(request_index),
                "noise_standard_deviation": float(sigma),
                "history_count": int(panel["histories"].shape[0]),
                "archive_history_count": int(manifest["histories_per_panel"]),
                "history_indices_used": np.asarray(
                    panel["history_indices_used"], dtype=np.int64
                ).tolist(),
                "observation_count_per_history": int(manifest["observation_count"]),
                "scalar_observation_count": int(
                    panel["histories"].shape[0]
                    * manifest["observation_count"]
                    * manifest["state_dim"]
                ),
                "selected_observation_mse": float(objective),
                "panel_archive_sha256": panel_sha,
                "inverse_config_sha256": inverse_config_sha256,
                "compute_device": "cpu",
                "cpu_model": _cpu_name(),
                "lm_initialization_seed": int(initialization_seed),
                **timing,
                **metrics,
            }
            write_csv(result_path.parent / "lm_trace.csv", trace)
            write_csv(
                result_path.parent / "final_candidate_parameters.csv",
                [
                    {
                        "restart": int(index),
                        "parameter_0": float(row[0]),
                        "parameter_1": float(row[1]),
                    }
                    for index, row in enumerate(candidates)
                ],
            )
            write_json(result_path, result)
            rows.append(result)
            request_index += 1
    write_csv(panel_dir / "lm_per_case_results.csv", rows)
    print(
        f"[lm] completed panel {panel_index:03d} ({len(rows)} requests)",
        flush=True,
    )
    return rows
