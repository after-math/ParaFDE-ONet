#!/usr/bin/env python3
"""High-75 inverse pilot with adaptive projected-Adam stopping and no L-BFGS-B.

This is an additive entry point.  It imports the existing High-75 protocol but
does not modify any of its source files.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import pipeline_high75_eight_history_40obs as _high75

_pipeline = _high75._pipeline
_HIGH75_VALIDATE = _high75.validate_config
_HIGH75_AGGREGATE = _high75.aggregate_stage


EARLY_STOP_DEFAULTS = {
    "minimum_steps": 100,
    "maximum_steps": 500,
    "check_interval": 20,
    "patience_checks": 5,
    "parameter_tolerance": 1.0e-5,
    "relative_objective_tolerance": 1.0e-7,
}


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate the original protocol and inject immutable pilot settings."""
    inverse = config["operator_projected_grid"]
    inverse["early_stopping"] = copy.deepcopy(EARLY_STOP_DEFAULTS)
    inverse["adam_steps"] = EARLY_STOP_DEFAULTS["maximum_steps"]
    inverse["lbfgsb_max_iterations"] = 0
    _HIGH75_VALIDATE(config)
    early = inverse["early_stopping"]
    if int(early["minimum_steps"]) < int(early["check_interval"]):
        raise ValueError("minimum_steps must cover at least one check interval")
    if int(early["maximum_steps"]) < int(early["minimum_steps"]):
        raise ValueError("maximum_steps must not be below minimum_steps")
    if int(early["patience_checks"]) < 1:
        raise ValueError("patience_checks must be positive")
    if float(early["parameter_tolerance"]) <= 0.0:
        raise ValueError("parameter_tolerance must be positive")
    if float(early["relative_objective_tolerance"]) <= 0.0:
        raise ValueError("relative_objective_tolerance must be positive")


def run_operator_job_early_stop(
    scenario_key: str,
    case_index: int,
    experiment_seed: int,
    config: Mapping[str, Any],
    checkpoint_path: Path,
    checkpoint_identity: Mapping[str, Any],
    archive_path: Path,
    device_name: str,
    output_dir: Path,
    smoke: bool,
    resume: bool,
) -> dict[str, Any]:
    """Run grid screening and projected Adam with paired stopping tests."""
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = _pipeline.job_signature(
        "operator_projected_grid", scenario_key, case_index, experiment_seed,
        config, checkpoint_identity, smoke,
    )
    result_path = output_dir / "result.json"
    if resume and result_path.exists():
        result = _pipeline.read_json(result_path)
        if result.get("signature") != signature:
            raise RuntimeError("completed early-stop operator result signature mismatch")
        return result

    online_started = time.perf_counter()
    _pipeline.setup_logger(output_dir / "job.log")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, checkpoint = _pipeline.load_operator_checkpoint(checkpoint_path, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    equation = _pipeline.DelayedSEIConfig.from_mapping(
        checkpoint["resolved_config"]["equation"]
    )
    case = _pipeline.load_case(archive_path, case_index)
    observed = _pipeline.condition_observations(
        case, _pipeline.condition_by_key(config, scenario_key)
    )
    history = torch.as_tensor(observed["histories"], dtype=torch.float32, device=device)
    history_grid = torch.as_tensor(case["history_grid"], dtype=torch.float32, device=device)
    observation_times = torch.as_tensor(observed["times"], dtype=torch.float32, device=device)
    observations = torch.as_tensor(observed["values"], dtype=torch.float32, device=device)

    inverse = config["operator_projected_grid"]
    smoke_config = config["smoke"]
    first_points = int(smoke_config["grid_transmission_points"] if smoke else inverse["grid_transmission_points"])
    second_points = int(smoke_config["grid_convexity_points"] if smoke else inverse["grid_convexity_points"])
    selected_count = int(smoke_config["selected_grid_starts"] if smoke else inverse["selected_grid_starts"])
    batch_size = int(smoke_config["grid_evaluation_batch_size"] if smoke else inverse["grid_evaluation_batch_size"])
    collocation_count = int(smoke_config["operator_collocation_points"] if smoke else inverse["collocation_points"])
    early = inverse["early_stopping"]
    maximum_steps = int(smoke_config["operator_adam_steps"] if smoke else early["maximum_steps"])
    minimum_steps = min(int(early["minimum_steps"]), maximum_steps)
    check_interval = int(early["check_interval"])
    patience_checks = int(early["patience_checks"])
    parameter_tolerance = float(early["parameter_tolerance"])
    objective_tolerance = float(early["relative_objective_tolerance"])

    started = time.perf_counter()
    normalized = _pipeline.screen_parameter_grid(
        model, history, observation_times, observations, observed["states"], equation,
        first_points, second_points, selected_count, batch_size,
        output_dir / "grid_screening.csv",
    ).requires_grad_(True)
    collocation_rng = np.random.default_rng(
        experiment_seed + int(config["randomness"]["collocation_offset"]) + case_index * 100
    )
    collocation_times = torch.as_tensor(
        np.sort(collocation_rng.uniform(0.0, float(case["output_times"][-1]),
                                       size=(selected_count, collocation_count)), axis=1),
        dtype=torch.float32, device=device,
    )
    optimizer = torch.optim.Adam([normalized], lr=float(inverse["parameter_learning_rate"]))
    trace: list[dict[str, Any]] = []
    previous_selected_parameters: torch.Tensor | None = None
    previous_selected_objective: float | None = None
    consecutive_stable_checks = 0
    stopped_early = False
    stop_reason = "maximum_steps"
    completed_steps = 0
    deadline = time.perf_counter() + float(config["maximum_seconds_per_job"])
    budget_exhausted = False

    for step in range(1, maximum_steps + 1):
        fraction = (step - 1) / max(maximum_steps - 1, 1)
        learning_rate = float(inverse["minimum_learning_rate"]) + 0.5 * (
            float(inverse["parameter_learning_rate"]) - float(inverse["minimum_learning_rate"])
        ) * (1.0 + math.cos(math.pi * fraction))
        optimizer.param_groups[0]["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        objective, _, _ = _pipeline.operator_objective(
            model, normalized, history, history_grid, observation_times, observations,
            observed["states"], collocation_times, equation, float(inverse["physics_weight"]),
        )
        objective.sum().backward()
        optimizer.step()
        with torch.no_grad():
            normalized.clamp_(-1.0, 1.0)
        completed_steps = step

        if step % check_interval == 0 or step == maximum_steps:
            # Physics residual evaluation differentiates the operator output with
            # respect to time, so autograd must remain enabled during this call.
            checked_objective, checked_physical, checked_components = _pipeline.operator_objective(
                model, normalized, history, history_grid, observation_times, observations,
                observed["states"], collocation_times, equation,
                float(inverse["physics_weight"]),
            )
            with torch.no_grad():
                selected = int(torch.argmin(checked_objective).cpu())
                current_parameters = normalized[selected].detach().clone()
                current_objective = float(checked_objective[selected].detach().cpu())
                parameter_change = float("inf")
                relative_objective_change = float("inf")
                if previous_selected_parameters is not None and previous_selected_objective is not None:
                    parameter_change = float(
                        torch.max(torch.abs(current_parameters - previous_selected_parameters)).cpu()
                    )
                    relative_objective_change = abs(current_objective - previous_selected_objective) / max(
                        abs(previous_selected_objective), 1.0e-12
                    )
                    if (step >= minimum_steps and parameter_change < parameter_tolerance
                            and relative_objective_change < objective_tolerance):
                        consecutive_stable_checks += 1
                    else:
                        consecutive_stable_checks = 0
                previous_selected_parameters = current_parameters
                previous_selected_objective = current_objective
                trace.append({
                    "phase": "adam_early_stop", "step": step,
                    "elapsed_seconds": time.perf_counter() - started,
                    "selected_restart": selected,
                    "selected_objective": current_objective,
                    "selected_data_loss": float(checked_components["data_loss"][selected].cpu()),
                    "selected_physics_loss": float(checked_components["physics_loss"][selected].cpu()),
                    "selected_transmission_b": float(checked_physical[selected, 0].cpu()),
                    "selected_convexity_a": float(checked_physical[selected, 1].cpu()),
                    "parameter_change_inf": parameter_change,
                    "relative_objective_change": relative_objective_change,
                    "stable_check_count": consecutive_stable_checks,
                    "learning_rate": learning_rate,
                })
            if consecutive_stable_checks >= patience_checks:
                stopped_early = True
                stop_reason = "parameter_and_objective_stable"
                break
        if time.perf_counter() >= deadline:
            budget_exhausted = True
            stop_reason = "time_budget_exhausted"
            break

    objective, physical, components = _pipeline.operator_objective(
        model, normalized, history, history_grid, observation_times, observations,
        observed["states"], collocation_times, equation, float(inverse["physics_weight"]),
    )
    optimization_seconds = time.perf_counter() - started
    _pipeline.write_csv(output_dir / "trace.csv", trace)
    np.save(output_dir / "candidate_parameters.npy", physical.detach().cpu().numpy())
    selected_index = int(torch.argmin(objective.detach()).cpu())
    result = _pipeline.finalize_result(
        "operator_projected_grid", scenario_key, case_index, experiment_seed, signature,
        "budget_exhausted" if budget_exhausted else "completed",
        physical.detach().cpu().numpy(), objective.detach().cpu().numpy(),
        case["true_parameters"], np.asarray(equation.parameter_bounds), case, equation,
        config, optimization_seconds, output_dir,
        {
            "selected_data_loss": float(components["data_loss"][selected_index].detach().cpu()),
            "selected_physics_loss": float(components["physics_loss"][selected_index].detach().cpu()),
            "grid_candidate_count": first_points * second_points,
            "checkpoint_model_type": checkpoint_identity["model_type"],
            "scalar_observation_count": int(observed["scalar_observation_count"]),
            "online_end_to_end_seconds": time.perf_counter() - online_started,
            "adam_steps_completed": completed_steps,
            "adam_stopped_early": int(stopped_early),
            "adam_stop_reason": stop_reason,
            "adam_parameter_tolerance": parameter_tolerance,
            "adam_relative_objective_tolerance": objective_tolerance,
            "adam_patience_checks": patience_checks,
            "lbfgsb_used": 0,
        },
    )
    _pipeline.write_json(result_path, result)
    return result


def aggregate_stage(stage_dir: Path, config: Mapping[str, Any], experiment_seed: int,
                    case_count: int, smoke: bool) -> dict[str, Any]:
    summary = _HIGH75_AGGREGATE(stage_dir, config, experiment_seed, case_count, smoke)
    summary["operator_implementation"] = (
        "Normalized 41x61 Cartesian screen, top-10 projected Adam with paired "
        "parameter/objective early stopping, maximum 500 steps, and no L-BFGS-B."
    )
    _pipeline.write_json(stage_dir / "comparison.json", summary)
    return summary


_pipeline.validate_config = validate_config
_pipeline.run_operator_job = run_operator_job_early_stop
_pipeline.aggregate_stage = aggregate_stage


def main(argv: Sequence[str] | None = None) -> int:
    return _pipeline.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
