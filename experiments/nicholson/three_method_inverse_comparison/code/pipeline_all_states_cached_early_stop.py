#!/usr/bin/env python3
"""Paired all-state Nicholson inverse comparison with a lightweight frozen operator.

This is an additive experiment entry point.  It imports the original pipeline for
case generation, numerical solvers, evaluation metrics and file formats, but does
not modify the original source files.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import pipeline as _pipeline


METHODS = ("operator_projected_grid", "projected_lm")
BASE_METHODS = ("operator_projected_grid", "projected_lm", "pinndde")
FORMAL_NOISE_LEVELS = (0.005, 0.01, 0.02, 0.05)

_ORIGINAL_VALIDATE_CONFIG = _pipeline.validate_config
_MODEL_CACHE: dict[tuple[str, str], tuple[Any, dict[str, Any]]] = {}


def condition_definitions(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return only the four all-state, one-history noise conditions."""
    noise = config["noise_escalation"]
    mode = noise["observation_modes"]["all_states"]
    definitions = []
    for standard_deviation in FORMAL_NOISE_LEVELS:
        definitions.append(
            {
                "experiment": "noise_escalation",
                "condition": (
                    f"all_states_{_pipeline.noise_tag(standard_deviation)}"
                ),
                "display_name": (
                    f"{mode['display_name']}, sigma={standard_deviation:g}"
                ),
                "history_count": 1,
                "observation_count": 20,
                "observed_state_indices": [0, 1, 2, 3],
                "noise_standard_deviation": float(standard_deviation),
                "observation_mode": "all_states",
            }
        )
    return definitions


def condition_by_key(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    matches = [item for item in condition_definitions(config) if item["condition"] == key]
    if len(matches) != 1:
        raise KeyError(f"unknown or ambiguous condition: {key}")
    return matches[0]


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate the base protocol, then resolve the independent variant."""
    # The original validator intentionally requires its original three-method
    # registry.  Restore that registry only during base-file validation, then put
    # the independent two-method registry back immediately.
    current_methods = _pipeline.METHODS
    _pipeline.METHODS = BASE_METHODS
    try:
        _ORIGINAL_VALIDATE_CONFIG(config)
    finally:
        _pipeline.METHODS = current_methods
    if not isinstance(config, dict):
        raise TypeError("the resolved experiment configuration must be mutable")
    config["experiment_name"] = (
        "nicholson_patch4d_all_states_cached_frozen_vs_early_stop_lm"
    )
    config["methods"] = list(METHODS)
    operator = config["operator_projected_grid"]
    operator.update(
        {
            "grid_beta_points": 41,
            "grid_tau0_points": 61,
            "selected_grid_starts": 10,
            "adam_steps": 500,
            "physics_weight": 0.0,
            "lbfgsb_max_iterations": 0,
            "minimum_adam_steps": 100,
            "early_stop_check_interval": 20,
            "early_stop_parameter_tolerance": 1.0e-5,
            "early_stop_relative_objective_tolerance": 1.0e-7,
            "early_stop_patience": 5,
            "cache_history_branch": True,
            "cache_observation_trunk": True,
            "persistent_worker_model": True,
            "objective": "all_four_state_observation_mse_only",
        }
    )
    lm = config["projected_lm"]
    lm.update(
        {
            "random_initializations": 5,
            "max_iterations": 100,
            "minimum_iterations": 5,
            "early_stop_relative_objective_tolerance": 1.0e-8,
            "early_stop_patience": 3,
            "objective": "all_four_state_observation_mse_only",
        }
    )
    config["comparison_protocol"] = {
        "paired_methods_in_one_run": True,
        "truth_used_only_for_final_evaluation": True,
        "shared_histories_parameters_times_and_noise": True,
        "formal_conditions": [item["condition"] for item in condition_definitions(config)],
        "formal_jobs_per_seed": 80,
    }


def _load_model_once(
    checkpoint_path: Path, device: torch.device
) -> tuple[Any, dict[str, Any], bool, float]:
    """Load one model once per persistent worker process and device."""
    key = (str(checkpoint_path.resolve()), str(device))
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached[0], cached[1], True, 0.0
    started = time.perf_counter()
    model, full_checkpoint = _pipeline.load_operator_checkpoint(
        checkpoint_path, device
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.model_type != "separate_parameter_shared":
        raise RuntimeError(
            "cached online decomposition requires separate_parameter_shared"
        )
    checkpoint = _pipeline.lightweight_checkpoint(full_checkpoint)
    del full_checkpoint
    gc.collect()
    load_seconds = time.perf_counter() - started
    _MODEL_CACHE[key] = (model, checkpoint)
    return model, checkpoint, False, load_seconds


def _build_fixed_features(
    model: Any, history: torch.Tensor, observation_times: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Cache exactly the history-branch product and observation-time trunk."""
    _pipeline.synchronize(history.device)
    history_started = time.perf_counter()
    with torch.no_grad():
        encoded = [
            1.0
            + branch(history[:, component, :]).reshape(
                history.shape[0], 4, model.latent_dim
            )
            for component, branch in enumerate(model.history_branches)
        ]
        history_features = encoded[0]
        for values in encoded[1:]:
            history_features = history_features * values
        history_features = history_features.detach()
    _pipeline.synchronize(history.device)
    history_seconds = time.perf_counter() - history_started

    trunk_started = time.perf_counter()
    with torch.no_grad():
        trunk_features = model.time_trunk(
            model.time_features(observation_times.reshape(1, -1))
        )[0].detach()
    _pipeline.synchronize(history.device)
    trunk_seconds = time.perf_counter() - trunk_started
    return history_features, trunk_features, history_seconds, trunk_seconds


def _cached_data_objective(
    model: Any,
    normalized_parameters: torch.Tensor,
    history_features: torch.Tensor,
    trunk_features: torch.Tensor,
    observations: torch.Tensor,
    observed_states: Sequence[int],
    bounds: Sequence[Sequence[float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the exact model output while recomputing only Parameter Branch."""
    physical = _pipeline.normalized_to_physical(normalized_parameters, bounds)
    parameter_features = model.parameter_branch(model.normalize_parameters(physical))
    fusion = history_features.unsqueeze(0) * (
        1.0 + parameter_features[:, None, None, :]
    )
    prediction = (
        torch.einsum("chsp,qp->chqs", fusion, trunk_features)
        * float(model.latent_scale)
        + model.output_bias.reshape(1, 1, 1, 4)
    )
    states = torch.as_tensor(
        observed_states, dtype=torch.long, device=normalized_parameters.device
    )
    prediction = prediction.index_select(3, states)
    target = observations.index_select(2, states).unsqueeze(0)
    loss = torch.mean((prediction - target) ** 2, dim=(1, 2, 3))
    if not torch.all(torch.isfinite(loss)):
        raise FloatingPointError("non-finite cached operator observation objective")
    return loss, physical


def _screen_grid(
    model: Any,
    history_features: torch.Tensor,
    trunk_features: torch.Tensor,
    observations: torch.Tensor,
    observed_states: Sequence[int],
    bounds: Sequence[Sequence[float]],
    first_points: int,
    second_points: int,
    selected_count: int,
    batch_size: int,
    output_path: Path,
) -> torch.Tensor:
    grid = _pipeline.parameter_grid(first_points, second_points, observations.device)
    pieces = []
    with torch.no_grad():
        for start in range(0, grid.shape[0], batch_size):
            objective, _ = _cached_data_objective(
                model,
                grid[start : start + batch_size],
                history_features,
                trunk_features,
                observations,
                observed_states,
                bounds,
            )
            pieces.append(objective.cpu())
    objectives = torch.cat(pieces).numpy().astype(np.float64)
    normalized = grid.cpu().numpy().astype(np.float64)
    physical = _pipeline.normalized_to_physical(
        grid.cpu(), bounds
    ).numpy().astype(np.float64)
    order = np.argsort(objectives, kind="stable")
    selected_indices = order[:selected_count]
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, order.size + 1)
    selected_set = set(int(value) for value in selected_indices)
    _pipeline.write_csv(
        output_path,
        [
            {
                "grid_index": index,
                "objective_rank": int(ranks[index]),
                "selected": int(index in selected_set),
                "normalized_beta": float(normalized[index, 0]),
                "normalized_tau0": float(normalized[index, 1]),
                "beta": float(physical[index, 0]),
                "tau0": float(physical[index, 1]),
                "observation_mse": float(objectives[index]),
            }
            for index in range(grid.shape[0])
        ],
    )
    return torch.as_tensor(
        normalized[selected_indices], dtype=torch.float32, device=observations.device
    )


def run_operator_job(
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
    """Run cached grid plus projected Adam; never call L-BFGS-B."""
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = _pipeline.job_signature(
        "operator_projected_grid",
        scenario_key,
        case_index,
        experiment_seed,
        config,
        checkpoint_identity,
        smoke,
    )
    result_path = output_dir / "result.json"
    if resume and result_path.exists():
        result = _pipeline.read_json(result_path)
        if result.get("signature") != signature:
            raise RuntimeError("completed operator result signature mismatch")
        return result

    online_started = time.perf_counter()
    _pipeline.setup_logger(output_dir / "job.log")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, checkpoint, cache_hit, model_load_seconds = _load_model_once(
        checkpoint_path, device
    )
    equation = _pipeline.NicholsonConfig.from_mapping(
        checkpoint["resolved_config"]["equation"]
    )
    case = _pipeline.load_case(archive_path, case_index)
    observed = _pipeline.condition_observations(
        case, condition_by_key(config, scenario_key)
    )
    history = torch.as_tensor(observed["histories"], dtype=torch.float32, device=device)
    observation_times = torch.as_tensor(
        observed["times"], dtype=torch.float32, device=device
    )
    observations = torch.as_tensor(
        observed["values"], dtype=torch.float32, device=device
    )
    inverse = config["operator_projected_grid"]
    smoke_config = config["smoke"]
    first_points = int(
        smoke_config["grid_beta_points"] if smoke else inverse["grid_beta_points"]
    )
    second_points = int(
        smoke_config["grid_tau0_points"] if smoke else inverse["grid_tau0_points"]
    )
    selected_count = int(
        smoke_config["selected_grid_starts"]
        if smoke
        else inverse["selected_grid_starts"]
    )
    batch_size = int(
        smoke_config["grid_evaluation_batch_size"]
        if smoke
        else inverse["grid_evaluation_batch_size"]
    )
    maximum_steps = int(
        smoke_config["operator_adam_steps"] if smoke else inverse["adam_steps"]
    )
    minimum_steps = maximum_steps if smoke else int(inverse["minimum_adam_steps"])
    check_interval = (
        max(1, maximum_steps)
        if smoke
        else int(inverse["early_stop_check_interval"])
    )
    parameter_tolerance = float(inverse["early_stop_parameter_tolerance"])
    objective_tolerance = float(
        inverse["early_stop_relative_objective_tolerance"]
    )
    patience = int(inverse["early_stop_patience"])

    optimization_started = time.perf_counter()
    (
        history_features,
        trunk_features,
        history_cache_seconds,
        trunk_cache_seconds,
    ) = _build_fixed_features(model, history, observation_times)
    normalized = _screen_grid(
        model,
        history_features,
        trunk_features,
        observations,
        observed["states"],
        equation.parameter_bounds,
        first_points,
        second_points,
        selected_count,
        batch_size,
        output_dir / "grid_screening.csv",
    ).requires_grad_(True)
    optimizer = torch.optim.Adam(
        [normalized], lr=float(inverse["parameter_learning_rate"])
    )
    trace: list[dict[str, Any]] = []
    progress_path = output_dir / "progress.pt"
    completed = 0
    elapsed_before = 0.0
    stable_checks = 0
    previous_checked_parameters: torch.Tensor | None = None
    previous_checked_objective: float | None = None
    if resume and progress_path.exists():
        progress = torch.load(progress_path, map_location=device, weights_only=False)
        if progress.get("signature") != signature:
            raise RuntimeError("operator progress signature mismatch")
        normalized.data.copy_(progress["normalized_parameters"])
        optimizer.load_state_dict(progress["optimizer_state_dict"])
        completed = int(progress["completed_steps"])
        elapsed_before = float(progress["elapsed_seconds"])
        trace = [dict(row) for row in progress["trace"]]
        stable_checks = int(progress.get("stable_checks", 0))
        saved_parameters = progress.get("previous_checked_parameters")
        if saved_parameters is not None:
            previous_checked_parameters = saved_parameters.to(device)
        saved_objective = progress.get("previous_checked_objective")
        if saved_objective is not None:
            previous_checked_objective = float(saved_objective)

    run_started = time.perf_counter()
    deadline = run_started + max(
        0.0, float(config["maximum_seconds_per_job"]) - elapsed_before
    )
    early_stopped = False
    budget_exhausted = False
    for step in range(completed + 1, maximum_steps + 1):
        fraction = (step - 1) / max(maximum_steps - 1, 1)
        learning_rate = float(inverse["minimum_learning_rate"]) + 0.5 * (
            float(inverse["parameter_learning_rate"])
            - float(inverse["minimum_learning_rate"])
        ) * (1.0 + math.cos(math.pi * fraction))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        objective, _ = _cached_data_objective(
            model,
            normalized,
            history_features,
            trunk_features,
            observations,
            observed["states"],
            equation.parameter_bounds,
        )
        objective.sum().backward()
        optimizer.step()
        with torch.no_grad():
            normalized.clamp_(-1.0, 1.0)

        should_check = step % check_interval == 0 or step == maximum_steps
        if should_check:
            with torch.no_grad():
                checked_objective, checked_physical = _cached_data_objective(
                    model,
                    normalized,
                    history_features,
                    trunk_features,
                    observations,
                    observed["states"],
                    equation.parameter_bounds,
                )
            selected = int(torch.argmin(checked_objective).cpu())
            selected_normalized = normalized.detach()[selected].clone()
            selected_objective = float(checked_objective[selected].cpu())
            parameter_change = float("inf")
            relative_objective_change = float("inf")
            if previous_checked_parameters is not None:
                parameter_change = float(
                    torch.max(
                        torch.abs(selected_normalized - previous_checked_parameters)
                    ).cpu()
                )
            if previous_checked_objective is not None:
                relative_objective_change = abs(
                    selected_objective - previous_checked_objective
                ) / max(abs(previous_checked_objective), 1.0e-12)
            simultaneous = (
                step >= minimum_steps
                and parameter_change < parameter_tolerance
                and relative_objective_change < objective_tolerance
            )
            stable_checks = stable_checks + 1 if simultaneous else 0
            trace.append(
                {
                    "phase": "projected_adam",
                    "step": step,
                    "elapsed_seconds": elapsed_before
                    + time.perf_counter()
                    - run_started,
                    "learning_rate": learning_rate,
                    "selected_restart": selected,
                    "selected_objective": selected_objective,
                    "selected_beta": float(checked_physical[selected, 0].cpu()),
                    "selected_tau0": float(checked_physical[selected, 1].cpu()),
                    "selected_normalized_parameter_change_inf": parameter_change,
                    "selected_relative_objective_change": relative_objective_change,
                    "simultaneous_stability_check": int(simultaneous),
                    "consecutive_stable_checks": stable_checks,
                }
            )
            previous_checked_parameters = selected_normalized
            previous_checked_objective = selected_objective
            completed = step
            elapsed = elapsed_before + time.perf_counter() - run_started
            _pipeline.atomic_torch_save(
                progress_path,
                {
                    "signature": signature,
                    "normalized_parameters": normalized.detach(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "completed_steps": step,
                    "elapsed_seconds": elapsed,
                    "trace": trace,
                    "stable_checks": stable_checks,
                    "previous_checked_parameters": previous_checked_parameters,
                    "previous_checked_objective": previous_checked_objective,
                },
            )
            if stable_checks >= patience:
                early_stopped = True
                break
        completed = step
        if time.perf_counter() >= deadline:
            budget_exhausted = True
            break

    with torch.no_grad():
        objective, physical = _cached_data_objective(
            model,
            normalized,
            history_features,
            trunk_features,
            observations,
            observed["states"],
            equation.parameter_bounds,
        )
    _pipeline.synchronize(device)
    optimization_seconds = elapsed_before + time.perf_counter() - optimization_started
    _pipeline.write_csv(output_dir / "trace.csv", trace)
    np.save(output_dir / "candidate_parameters.npy", physical.cpu().numpy())
    selected_index = int(torch.argmin(objective).cpu())
    online_seconds = time.perf_counter() - online_started
    accelerator_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor() or platform.machine()
    )
    status = (
        "budget_exhausted"
        if budget_exhausted
        else "early_stopped"
        if early_stopped
        else "completed"
    )
    return _pipeline.finalize_result(
        "operator_projected_grid",
        scenario_key,
        case_index,
        experiment_seed,
        signature,
        status,
        physical.cpu().numpy(),
        objective.cpu().numpy(),
        case["true_parameters"],
        np.asarray(equation.parameter_bounds),
        case,
        equation,
        config,
        optimization_seconds,
        output_dir,
        {
            "selected_data_loss": float(objective[selected_index].cpu()),
            "selected_physics_loss": 0.0,
            "objective_definition": "mean squared error over all four observed states",
            "grid_candidate_count": first_points * second_points,
            "checkpoint_model_type": checkpoint_identity["model_type"],
            "scalar_observation_count": int(observed["scalar_observation_count"]),
            "adam_steps_completed": int(completed),
            "adam_early_stopped": int(early_stopped),
            "lbfgsb_used": 0,
            "model_cache_hit": int(cache_hit),
            "model_residency_status": "warm" if cache_hit else "cold",
            "checkpoint_load_seconds": float(model_load_seconds),
            "history_branch_cache_seconds": float(history_cache_seconds),
            "observation_trunk_cache_seconds": float(trunk_cache_seconds),
            "resident_equivalent_online_seconds": float(
                max(0.0, online_seconds - model_load_seconds)
            ),
            "cold_start_online_seconds": None if cache_hit else float(online_seconds),
            "warm_start_online_seconds": float(online_seconds) if cache_hit else None,
            "compute_device": str(device),
            "accelerator_name": accelerator_name,
            "online_end_to_end_seconds": float(online_seconds),
        },
    )


def _cpu_model_name() -> str:
    if Path("/proc/cpuinfo").is_file():
        try:
            for line in Path("/proc/cpuinfo").read_text(
                encoding="utf-8", errors="ignore"
            ).splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def run_lm_job(
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
    """Run the original five-start projected LM with one conservative stop rule."""
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = _pipeline.job_signature(
        "projected_lm",
        scenario_key,
        case_index,
        experiment_seed,
        config,
        checkpoint_identity,
        smoke,
    )
    result_path = output_dir / "result.json"
    if resume and result_path.exists():
        result = _pipeline.read_json(result_path)
        if result.get("signature") != signature:
            raise RuntimeError("completed LM result signature mismatch")
        return result

    online_started = time.perf_counter()
    _pipeline.setup_logger(output_dir / "job.log")
    equation = _pipeline.NicholsonConfig.from_mapping(
        checkpoint["resolved_config"]["equation"]
    )
    case = _pipeline.load_case(archive_path, case_index)
    observed = _pipeline.condition_observations(
        case, condition_by_key(config, scenario_key)
    )
    lm = config["projected_lm"]
    restart_count = int(
        config["smoke"]["pinndde_random_initializations"]
        if smoke
        else lm["random_initializations"]
    )
    maximum_iterations = int(
        config["smoke"]["lm_iterations"] if smoke else lm["max_iterations"]
    )
    minimum_iterations = maximum_iterations if smoke else int(lm["minimum_iterations"])
    relative_tolerance = float(lm["early_stop_relative_objective_tolerance"])
    patience = int(lm["early_stop_patience"])
    bounds = np.asarray(equation.parameter_bounds, dtype=np.float64)
    parameters = _pipeline.lhs_physical(
        restart_count,
        _pipeline.shared_initialization_seed(config, experiment_seed, case_index),
        bounds,
    )
    _pipeline.write_csv(
        output_dir / "initial_lhs_points.csv",
        [
            {
                "restart_index": index,
                "beta": float(row[0]),
                "tau0": float(row[1]),
            }
            for index, row in enumerate(parameters)
        ],
    )
    damping = np.full(restart_count, float(lm["initial_damping"]), dtype=np.float64)
    completed = 0
    elapsed_before = 0.0
    trace: list[dict[str, Any]] = []
    progress_path = output_dir / "progress.pt"
    solver_batch_calls = 0
    trajectory_solves = 0
    stable_iterations = 0
    previous_selected_objective: float | None = None
    if resume and progress_path.exists():
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        if progress.get("signature") != signature:
            raise RuntimeError("LM progress signature mismatch")
        parameters = np.asarray(progress["parameters"], dtype=np.float64)
        damping = np.asarray(progress["damping"], dtype=np.float64)
        completed = int(progress["completed_iterations"])
        elapsed_before = float(progress["elapsed_seconds"])
        trace = [dict(row) for row in progress["trace"]]
        solver_batch_calls = int(progress["solver_batch_calls"])
        trajectory_solves = int(progress["trajectory_solves"])
        stable_iterations = int(progress.get("stable_iterations", 0))
        saved_objective = progress.get("previous_selected_objective")
        if saved_objective is not None:
            previous_selected_objective = float(saved_objective)

    internal_step = float(lm["internal_step"])
    finite_step = float(lm["finite_difference_step"])
    started = time.perf_counter()
    early_stopped = False
    budget_exhausted = False
    while completed < maximum_iterations:
        residuals, _ = _pipeline.direct_solver_residuals(
            parameters, case, observed, equation, internal_step
        )
        solver_batch_calls += 1
        trajectory_solves += restart_count * int(observed["history_count"])
        objectives = np.sum(residuals**2, axis=1)
        if previous_selected_objective is None:
            previous_selected_objective = float(np.min(objectives))

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
        perturbed_residuals, _ = _pipeline.direct_solver_residuals(
            perturbed.reshape(4 * restart_count, 2),
            case,
            observed,
            equation,
            internal_step,
        )
        solver_batch_calls += 1
        trajectory_solves += 4 * restart_count * int(observed["history_count"])
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
        updated_objectives = np.where(accepted, candidate_objectives, objectives)
        damping[accepted] = np.maximum(
            float(lm["minimum_damping"]),
            damping[accepted] / float(lm["damping_decrease"]),
        )
        damping[~accepted] = np.minimum(
            float(lm["maximum_damping"]),
            damping[~accepted] * float(lm["damping_increase"]),
        )
        completed += 1
        selected = int(np.argmin(updated_objectives))
        selected_objective = float(updated_objectives[selected])
        relative_change = abs(
            selected_objective - previous_selected_objective
        ) / max(abs(previous_selected_objective), 1.0e-12)
        stable = completed >= minimum_iterations and relative_change < relative_tolerance
        stable_iterations = stable_iterations + 1 if stable else 0
        previous_selected_objective = selected_objective
        elapsed = elapsed_before + time.perf_counter() - started
        trace.append(
            {
                "phase": "projected_lm",
                "iteration": completed,
                "elapsed_seconds": elapsed,
                "selected_restart": selected,
                "selected_objective": selected_objective,
                "selected_beta": float(parameters[selected, 0]),
                "selected_tau0": float(parameters[selected, 1]),
                "accepted_restart_count": int(np.sum(accepted)),
                "mean_damping": float(np.mean(damping)),
                "selected_relative_objective_change": relative_change,
                "consecutive_stable_iterations": stable_iterations,
            }
        )
        should_save = (
            completed % int(lm["checkpoint_interval"]) == 0
            or completed == maximum_iterations
            or stable_iterations >= patience
        )
        if should_save:
            _pipeline.atomic_torch_save(
                progress_path,
                {
                    "signature": signature,
                    "parameters": parameters,
                    "damping": damping,
                    "completed_iterations": completed,
                    "elapsed_seconds": elapsed,
                    "trace": trace,
                    "solver_batch_calls": solver_batch_calls,
                    "trajectory_solves": trajectory_solves,
                    "stable_iterations": stable_iterations,
                    "previous_selected_objective": previous_selected_objective,
                },
            )
        if stable_iterations >= patience:
            early_stopped = True
            break
        if elapsed >= float(config["maximum_seconds_per_job"]):
            budget_exhausted = True
            break

    residuals, _ = _pipeline.direct_solver_residuals(
        parameters, case, observed, equation, internal_step
    )
    solver_batch_calls += 1
    trajectory_solves += restart_count * int(observed["history_count"])
    objectives = np.sum(residuals**2, axis=1)
    optimization_seconds = elapsed_before + time.perf_counter() - started
    _pipeline.write_csv(output_dir / "trace.csv", trace)
    np.save(output_dir / "candidate_parameters.npy", parameters)
    online_seconds = time.perf_counter() - online_started
    status = (
        "budget_exhausted"
        if budget_exhausted
        else "early_stopped"
        if early_stopped
        else "completed"
    )
    return _pipeline.finalize_result(
        "projected_lm",
        scenario_key,
        case_index,
        experiment_seed,
        signature,
        status,
        parameters,
        objectives,
        case["true_parameters"],
        bounds,
        case,
        equation,
        config,
        optimization_seconds,
        output_dir,
        {
            "solver_batch_calls": solver_batch_calls,
            "direct_trajectory_solves": trajectory_solves,
            "lm_final_damping_mean": float(np.mean(damping)),
            "inverse_internal_step": internal_step,
            "finite_difference_step": finite_step,
            "lm_iterations_completed": int(completed),
            "lm_early_stopped": int(early_stopped),
            "objective_definition": "mean squared error over all four observed states",
            "scalar_observation_count": int(observed["scalar_observation_count"]),
            "compute_device": "cpu",
            "cpu_model": _cpu_model_name(),
            "online_end_to_end_seconds": float(online_seconds),
        },
    )


def queue_worker(
    device_name: str,
    jobs: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    checkpoint_path_text: str,
    checkpoint_identity: Mapping[str, Any],
    checkpoint_science: Mapping[str, Any],
    archive_path_text: str,
    stage_dir_text: str,
    smoke: bool,
    resume: bool,
) -> list[dict[str, Any]]:
    """Run one queue; the global model cache survives every Frozen job."""
    checkpoint_path = Path(checkpoint_path_text)
    archive_path = Path(archive_path_text)
    stage_dir = Path(stage_dir_text)
    results = []
    for job in jobs:
        output_dir = _pipeline.job_output_dir(stage_dir, job)
        common = (
            str(job["condition"]),
            int(job["case_index"]),
            int(job["experiment_seed"]),
            config,
        )
        if job["method"] == "operator_projected_grid":
            result = run_operator_job(
                *common,
                checkpoint_path,
                checkpoint_identity,
                archive_path,
                device_name,
                output_dir,
                smoke,
                resume,
            )
        elif job["method"] == "projected_lm":
            result = run_lm_job(
                *common,
                checkpoint_science,
                checkpoint_identity,
                archive_path,
                output_dir,
                smoke,
                resume,
            )
        else:
            raise ValueError(f"unsupported method: {job['method']}")
        results.append(result)
        gc.collect()
        if device_name.startswith("cuda"):
            torch.cuda.empty_cache()
    return results


def build_method_balanced_queues(
    jobs: Sequence[Mapping[str, Any]], worker_count: int
) -> list[list[Mapping[str, Any]]]:
    """Give every worker Frozen jobs first, then LM jobs, round-robin."""
    queues: list[list[Mapping[str, Any]]] = [[] for _ in range(worker_count)]
    for method in METHODS:
        method_jobs = [job for job in jobs if job["method"] == method]
        for index, job in enumerate(method_jobs):
            queues[index % worker_count].append(job)
    return queues


def _mean_or_none(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def aggregate_stage(
    stage_dir: Path,
    config: Mapping[str, Any],
    experiment_seed: int,
    case_count: int,
    smoke: bool,
) -> dict[str, Any]:
    """Aggregate only the paired 4 conditions x 10 cases x 2 methods."""
    rows: list[dict[str, Any]] = []
    conditions = condition_definitions(config)
    for condition in conditions:
        for case_index in range(case_count):
            for method in METHODS:
                path = (
                    stage_dir
                    / "jobs"
                    / str(condition["experiment"])
                    / str(condition["condition"])
                    / f"case_{case_index:03d}"
                    / method
                    / "result.json"
                )
                if not path.exists():
                    raise RuntimeError(f"missing completed job result: {path}")
                rows.append(_pipeline.read_json(path))
    _pipeline.write_csv(stage_dir / "per_case_results.csv", rows)

    summary_rows = []
    for condition in conditions:
        for method in METHODS:
            group = [
                row
                for row in rows
                if row["condition"] == condition["condition"]
                and row["method_key"] == method
            ]
            row = {
                "experiment_seed": int(experiment_seed),
                "experiment": condition["experiment"],
                "condition": condition["condition"],
                "history_count": 1,
                "noise_standard_deviation": condition[
                    "noise_standard_deviation"
                ],
                "method_key": method,
                "method": _pipeline.METHOD_LABELS[method],
                **_pipeline.summarize_group(group),
            }
            if method == "operator_projected_grid":
                resident_values = np.asarray(
                    [
                        float(item["resident_equivalent_online_seconds"])
                        for item in group
                    ],
                    dtype=np.float64,
                )
                row.update(
                    {
                        "resident_equivalent_online_seconds_mean": float(
                            np.mean(resident_values)
                        ),
                        "resident_equivalent_online_seconds_std": float(
                            np.std(resident_values, ddof=1)
                        )
                        if len(resident_values) > 1
                        else 0.0,
                        "cold_start_case_count": int(
                            sum(int(item["model_cache_hit"]) == 0 for item in group)
                        ),
                        "warm_start_case_count": int(
                            sum(int(item["model_cache_hit"]) == 1 for item in group)
                        ),
                    }
                )
            summary_rows.append(row)
    _pipeline.write_csv(stage_dir / "comparison.csv", summary_rows)

    frozen_rows = [row for row in rows if row["method_key"] == METHODS[0]]
    lm_rows = [row for row in rows if row["method_key"] == METHODS[1]]
    cold = [
        float(row["online_end_to_end_seconds"])
        for row in frozen_rows
        if int(row["model_cache_hit"]) == 0
    ]
    warm = [
        float(row["online_end_to_end_seconds"])
        for row in frozen_rows
        if int(row["model_cache_hit"]) == 1
    ]
    resident = [
        float(row["resident_equivalent_online_seconds"]) for row in frozen_rows
    ]
    timing_rows = [
        {
            "method_key": METHODS[0],
            "timing_definition": "cold_start_actual",
            "case_count": len(cold),
            "mean_seconds": _mean_or_none(cold),
            "median_seconds": float(np.median(cold)) if cold else None,
        },
        {
            "method_key": METHODS[0],
            "timing_definition": "warm_start_actual",
            "case_count": len(warm),
            "mean_seconds": _mean_or_none(warm),
            "median_seconds": float(np.median(warm)) if warm else None,
        },
        {
            "method_key": METHODS[0],
            "timing_definition": "resident_equivalent_all_cases",
            "case_count": len(resident),
            "mean_seconds": _mean_or_none(resident),
            "median_seconds": float(np.median(resident)) if resident else None,
        },
        {
            "method_key": METHODS[1],
            "timing_definition": "cpu_online_end_to_end",
            "case_count": len(lm_rows),
            "mean_seconds": _mean_or_none(
                [float(row["online_end_to_end_seconds"]) for row in lm_rows]
            ),
            "median_seconds": float(
                np.median([float(row["online_end_to_end_seconds"]) for row in lm_rows])
            ),
        },
    ]
    _pipeline.write_csv(stage_dir / "timing_breakdown.csv", timing_rows)
    summary = {
        "experiment_seed": int(experiment_seed),
        "smoke": bool(smoke),
        "case_count": int(case_count),
        "condition_count": len(conditions),
        "job_count": len(rows),
        "rows": summary_rows,
        "timing_rows": timing_rows,
        "selection_rule": (
            "Both methods select only by four-state observation MSE; truth is "
            "used only after optimization for evaluation."
        ),
        "shared_data_rule": (
            "Frozen and LM share the same histories, true parameters, 20 "
            "observation times and standard-normal noise arrays in one run."
        ),
        "operator_implementation": (
            "41x61 normalized grid, cached History Branch and observation Trunk, "
            "Top-10 projected Adam with 100/20/1e-5/1e-7/patience-5 early "
            "stopping and a 500-step cap; no physics loss and no L-BFGS-B."
        ),
        "lm_implementation": (
            "Original five LHS starts, RK4 step 0.01, centered finite difference "
            "step 0.001 and original damping; relative-objective early stopping "
            "with tolerance 1e-8 and patience 3, capped at 100 iterations."
        ),
    }
    _pipeline.write_json(stage_dir / "comparison.json", summary)
    return summary


def run_seed_stage(
    stage_dir: Path,
    config: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    checkpoint_identity: Mapping[str, Any],
    checkpoint_path: Path,
    experiment_seed: int,
    devices: Sequence[str],
    processes: int,
    smoke: bool,
    resume: bool,
) -> dict[str, Any]:
    # A smoke pool and a formal pool are separate deployments.  Clearing here also
    # preserves that cold-start boundary when --processes 1 executes in-process.
    _MODEL_CACHE.clear()
    gc.collect()
    stage_dir.mkdir(parents=True, exist_ok=True)
    case_count = int(config["smoke"]["case_count"] if smoke else config["case_count"])
    archive = _pipeline.prepare_shared_cases(
        stage_dir / "shared_cases",
        config,
        checkpoint,
        checkpoint_identity,
        experiment_seed,
        smoke,
        resume,
    )
    jobs = [
        {
            "method": method,
            "experiment": condition["experiment"],
            "condition": condition["condition"],
            "case_index": case_index,
            "experiment_seed": int(experiment_seed),
        }
        for condition in condition_definitions(config)
        for case_index in range(case_count)
        for method in METHODS
    ]
    worker_count = min(len(jobs), len(devices), int(processes))
    queues = build_method_balanced_queues(jobs, worker_count)
    _pipeline.LOGGER.info(
        "Seed %d worker queue lengths: %s", experiment_seed, [len(q) for q in queues]
    )
    science = _pipeline.lightweight_checkpoint(checkpoint)
    arguments = [
        (
            devices[index],
            queue,
            config,
            str(checkpoint_path),
            checkpoint_identity,
            science,
            str(archive),
            str(stage_dir),
            smoke,
            resume,
        )
        for index, queue in enumerate(queues)
        if queue
    ]
    if len(arguments) == 1:
        queue_worker(*arguments[0])
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=len(arguments), mp_context=context
        ) as executor:
            futures = [executor.submit(queue_worker, *values) for values in arguments]
            for future in as_completed(futures):
                future.result()
    return aggregate_stage(stage_dir, config, experiment_seed, case_count, smoke)


# The original main function remains responsible for strict checkpoint checks,
# output-directory safety, shared runtime metadata, smoke/full stages and resume.
_pipeline.METHODS = METHODS
_pipeline.condition_definitions = condition_definitions
_pipeline.condition_by_key = condition_by_key
_pipeline.validate_config = validate_config
_pipeline.run_seed_stage = run_seed_stage
_pipeline.aggregate_stage = aggregate_stage


def main(argv: Sequence[str] | None = None) -> int:
    return _pipeline.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
