#!/usr/bin/env python3
"""Additive High-75 Direct-LM entry point with one simple stopping rule."""

from __future__ import annotations

import copy
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import pipeline_high75_eight_history_40obs as _high75

_pipeline = _high75._pipeline
_HIGH75_VALIDATE = _high75.validate_config
_HIGH75_AGGREGATE = _high75.aggregate_stage

MINIMUM_ITERATIONS = 5
RELATIVE_OBJECTIVE_TOLERANCE = 1.0e-8
EARLY_STOPPING_PATIENCE = 3


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate the original protocol and record the additive LM rule."""
    lm = config["projected_lm"]
    lm["minimum_iterations"] = MINIMUM_ITERATIONS
    lm["relative_objective_tolerance"] = RELATIVE_OBJECTIVE_TOLERANCE
    lm["early_stopping_patience"] = EARLY_STOPPING_PATIENCE
    _HIGH75_VALIDATE(config)


def run_lm_job_early_stop(
    scenario_key: str,
    case_index: int,
    experiment_seed: int,
    config: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    checkpoint_identity: Mapping[str, Any],
    archive_path: Path,
    output_dir: Path,
    smoke: bool,
    resume: bool,
) -> dict[str, Any]:
    """Run the original projected LM and stop after three stable objectives."""
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = _pipeline.job_signature(
        "projected_lm", scenario_key, case_index, experiment_seed,
        config, checkpoint_identity, smoke,
    )
    result_path = output_dir / "result.json"
    if resume and result_path.exists():
        result = _pipeline.read_json(result_path)
        if result.get("signature") != signature:
            raise RuntimeError("completed early-stop LM result signature mismatch")
        return result

    online_started = time.perf_counter()
    _pipeline.setup_logger(output_dir / "job.log")
    equation = _pipeline.DelayedSEIConfig.from_mapping(
        checkpoint["resolved_config"]["equation"]
    )
    case = _pipeline.load_case(archive_path, case_index)
    observed = _pipeline.condition_observations(
        case, _pipeline.condition_by_key(config, scenario_key)
    )
    lm = config["projected_lm"]
    restart_count = int(
        config["smoke"]["pinndde_random_initializations"]
        if smoke else lm["random_initializations"]
    )
    maximum_iterations = int(
        config["smoke"]["lm_iterations"] if smoke else lm["max_iterations"]
    )
    minimum_iterations = min(int(lm["minimum_iterations"]), maximum_iterations)
    tolerance = float(lm["relative_objective_tolerance"])
    patience = int(lm["early_stopping_patience"])
    bounds = np.asarray(equation.parameter_bounds, dtype=np.float64)
    parameters = _pipeline.lhs_physical(
        restart_count,
        _pipeline.shared_initialization_seed(config, experiment_seed, case_index),
        bounds,
    )
    _pipeline.write_csv(
        output_dir / "initial_lhs_points.csv",
        [{"restart_index": index, "transmission_b": float(row[0]),
          "convexity_a": float(row[1])}
         for index, row in enumerate(parameters)],
    )
    damping = np.full(restart_count, float(lm["initial_damping"]), dtype=np.float64)
    internal_step = float(lm["internal_step"])
    finite_step = float(lm["finite_difference_step"])
    trace: list[dict[str, Any]] = []
    solver_batch_calls = 0
    trajectory_solves = 0
    completed = 0
    stable_count = 0
    previous_best_objective: float | None = None
    stopped_early = False
    stop_reason = "maximum_iterations"
    started = time.perf_counter()
    budget_exhausted = False

    while completed < maximum_iterations:
        residuals, _ = _pipeline.direct_solver_residuals(
            parameters, case, observed, equation, internal_step
        )
        solver_batch_calls += 1
        trajectory_solves += restart_count * int(observed["history_count"])
        objectives = np.sum(residuals**2, axis=1)

        perturbed = np.repeat(parameters[None, :, :], 4, axis=0)
        for parameter_index in range(2):
            perturbed[2 * parameter_index, :, parameter_index] = np.minimum(
                bounds[parameter_index, 1], parameters[:, parameter_index] + finite_step
            )
            perturbed[2 * parameter_index + 1, :, parameter_index] = np.maximum(
                bounds[parameter_index, 0], parameters[:, parameter_index] - finite_step
            )
        perturbed_residuals, _ = _pipeline.direct_solver_residuals(
            perturbed.reshape(4 * restart_count, 2), case, observed, equation, internal_step
        )
        solver_batch_calls += 1
        trajectory_solves += 4 * restart_count * int(observed["history_count"])
        perturbed_residuals = perturbed_residuals.reshape(
            4, restart_count, residuals.shape[1]
        )
        jacobians = np.empty((restart_count, residuals.shape[1], 2), dtype=np.float64)
        for parameter_index in range(2):
            denominator = (
                perturbed[2 * parameter_index, :, parameter_index]
                - perturbed[2 * parameter_index + 1, :, parameter_index]
            )
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
        candidate_residuals, _ = _pipeline.direct_solver_residuals(
            proposed, case, observed, equation, internal_step
        )
        solver_batch_calls += 1
        trajectory_solves += restart_count * int(observed["history_count"])
        candidate_objectives = np.sum(candidate_residuals**2, axis=1)
        accepted = candidate_objectives < objectives
        parameters[accepted] = proposed[accepted]
        objectives[accepted] = candidate_objectives[accepted]
        damping[accepted] = np.maximum(
            float(lm["minimum_damping"]),
            damping[accepted] / float(lm["damping_decrease"]),
        )
        damping[~accepted] = np.minimum(
            float(lm["maximum_damping"]),
            damping[~accepted] * float(lm["damping_increase"]),
        )
        completed += 1

        selected = int(np.argmin(objectives))
        best_objective = float(objectives[selected])
        relative_change = float("inf")
        if previous_best_objective is not None:
            relative_change = abs(best_objective - previous_best_objective) / max(
                abs(previous_best_objective), 1.0e-12
            )
            if completed >= minimum_iterations and relative_change < tolerance:
                stable_count += 1
            else:
                stable_count = 0
        previous_best_objective = best_objective
        if completed % int(lm["record_interval"]) == 0:
            trace.append({
                "phase": "lm_early_stop", "iteration": completed,
                "elapsed_seconds": time.perf_counter() - started,
                "selected_restart": selected,
                "selected_objective": best_objective,
                "selected_transmission_b": float(parameters[selected, 0]),
                "selected_convexity_a": float(parameters[selected, 1]),
                "mean_damping": float(np.mean(damping)),
                "relative_objective_change": relative_change,
                "stable_iteration_count": stable_count,
            })
        if stable_count >= patience:
            stopped_early = True
            stop_reason = "relative_objective_stable"
            break
        if time.perf_counter() - started >= float(config["maximum_seconds_per_job"]):
            budget_exhausted = True
            stop_reason = "time_budget_exhausted"
            break

    residuals, _ = _pipeline.direct_solver_residuals(
        parameters, case, observed, equation, internal_step
    )
    solver_batch_calls += 1
    trajectory_solves += restart_count * int(observed["history_count"])
    objectives = np.sum(residuals**2, axis=1)
    optimization_seconds = time.perf_counter() - started
    _pipeline.write_csv(output_dir / "trace.csv", trace)
    np.save(output_dir / "candidate_parameters.npy", parameters)
    result = _pipeline.finalize_result(
        "projected_lm", scenario_key, case_index, experiment_seed, signature,
        "budget_exhausted" if budget_exhausted else "completed",
        parameters, objectives, case["true_parameters"], bounds, case, equation,
        config, optimization_seconds, output_dir,
        {
            "solver_batch_calls": solver_batch_calls,
            "direct_trajectory_solves": trajectory_solves,
            "lm_final_damping_mean": float(np.mean(damping)),
            "lm_iterations_completed": completed,
            "lm_stopped_early": int(stopped_early),
            "lm_stop_reason": stop_reason,
            "lm_relative_objective_tolerance": tolerance,
            "lm_early_stopping_patience": patience,
            "inverse_internal_step": internal_step,
            "finite_difference_step": finite_step,
            "scalar_observation_count": int(observed["scalar_observation_count"]),
            "online_end_to_end_seconds": time.perf_counter() - online_started,
        },
    )
    _pipeline.write_json(result_path, result)
    return result


def aggregate_stage(stage_dir: Path, config: Mapping[str, Any], experiment_seed: int,
                    case_count: int, smoke: bool) -> dict[str, Any]:
    summary = _HIGH75_AGGREGATE(stage_dir, config, experiment_seed, case_count, smoke)
    summary["lm_implementation"] = (
        "Original direct causal RK4 projected LM, stopped only when the relative "
        "best objective change stays below 1e-8 for three consecutive iterations "
        "after at least five iterations; maximum 100 iterations."
    )
    _pipeline.write_json(stage_dir / "comparison.json", summary)
    return summary


_pipeline.validate_config = validate_config
_pipeline.run_lm_job = run_lm_job_early_stop
_pipeline.aggregate_stage = aggregate_stage


def main(argv: Sequence[str] | None = None) -> int:
    return _pipeline.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
