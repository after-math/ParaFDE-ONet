"""One-event inverse-calibration and 101x101 operating-condition pilot."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch

from cstr_forward.equation import CSTRConfig, constant_histories, solve_batch
from cstr_forward.model import CSTRParaFDEONet, load_operator_checkpoint


LOGGER = logging.getLogger(__name__)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, values: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(dict(values), indent=2), encoding="utf-8")
    temporary.replace(path)


def save_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot save an empty CSV")
    fieldnames = list(rows[0])
    fieldnames.extend(
        key for row in rows[1:] for key in row if key not in fieldnames
    )
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def build_fixed_operator_features(
    model: CSTRParaFDEONet,
    histories: np.ndarray | torch.Tensor,
    times: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Cache the history branches and time trunk exactly as in FDE_parameter.tex."""
    history = torch.as_tensor(histories, dtype=torch.float32, device=device)
    if history.shape[0] != 1:
        raise ValueError("the industrial pilot cache currently expects one fixed history")
    synchronize(device)
    history_started = time.perf_counter()
    history_features = model.encode_histories(history).detach()
    synchronize(device)
    history_seconds = time.perf_counter() - history_started

    trunk_started = time.perf_counter()
    trunk_features = model.time_trunk(
        model.time_features(times.reshape(1, -1))
    )[0].detach()
    synchronize(device)
    trunk_seconds = time.perf_counter() - trunk_started
    return history_features, trunk_features, {
        "history_cache_seconds": history_seconds,
        "time_trunk_cache_seconds": trunk_seconds,
    }


def encode_condition_features(
    model: CSTRParaFDEONet,
    conditions: torch.Tensor,
    detach: bool = False,
) -> torch.Tensor:
    """Encode the parameter/control input using the dedicated condition branch."""
    result = model.encode_conditions(conditions)
    return result.detach() if detach else result


def predict_from_cached_features(
    model: CSTRParaFDEONet,
    history_features: torch.Tensor,
    condition_features: torch.Tensor,
    trunk_features: torch.Tensor,
) -> torch.Tensor:
    """Fuse cached factors without reevaluating a frozen subnetwork."""
    if history_features.shape[0] != 1:
        raise ValueError("cached pilot prediction expects one fixed history")
    if condition_features.ndim != 2 or condition_features.shape[1] != model.latent_dim:
        raise ValueError("shared condition features must have shape [B,P]")
    fused = history_features[0].unsqueeze(0) * condition_features.unsqueeze(1)
    normalized_output = (
        torch.einsum("bsp,qp->bqs", fused, trunk_features) * model.latent_scale
        + model.normalized_output_bias
    )
    return normalized_output * model.output_std + model.output_mean


def cached_operator_prediction(
    model: CSTRParaFDEONet,
    history_features: torch.Tensor,
    trunk_features: torch.Tensor,
    conditions: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the exact operator while recomputing only its condition branch."""
    return predict_from_cached_features(
        model,
        history_features,
        encode_condition_features(model, conditions),
        trunk_features,
    )


def control_grid(equation: CSTRConfig, points: int) -> np.ndarray:
    bounds = equation.bounds_array
    dilution = np.linspace(bounds[2, 0], bounds[2, 1], points)
    coolant = np.linspace(bounds[3, 0], bounds[3, 1], points)
    dd, tt = np.meshgrid(dilution, coolant, indexing="ij")
    return np.column_stack((dd.ravel(), tt.ravel())).astype(np.float32)


def trajectory_features(trajectories: np.ndarray) -> dict[str, np.ndarray]:
    late_count = max(2, trajectories.shape[1] // 5)
    late = trajectories[:, -late_count:, :]
    maximum_temperature = trajectories[:, :, 1].max(axis=1)
    conversion = 1.0 - late[:, :, 0].mean(axis=1)
    temperature_std = late[:, :, 1].std(axis=1)
    return {
        "maximum_temperature": maximum_temperature,
        "conversion": conversion,
        "temperature_std": temperature_std,
    }


def feasible_mask(features: Mapping[str, np.ndarray], config: Mapping[str, Any], margin: bool) -> np.ndarray:
    prefix = "decision_" if margin else ""
    return (
        (features["maximum_temperature"] <= float(config[f"{prefix}temperature_limit"]))
        & (features["conversion"] >= float(config[f"{prefix}conversion_minimum"]))
        & (features["temperature_std"] <= float(config[f"{prefix}temperature_std_limit"]))
    )


def choose_control(
    controls: np.ndarray,
    features: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    margin: bool,
) -> tuple[int, np.ndarray]:
    feasible = feasible_mask(features, config, margin)
    productivity = controls[:, 0] * features["conversion"]
    if not feasible.any():
        raise RuntimeError("no feasible control exists on the configured grid")
    index = int(np.argmax(np.where(feasible, productivity, -np.inf)))
    return index, productivity


def direct_screen(
    histories: np.ndarray,
    parameters: np.ndarray,
    controls: np.ndarray,
    config: Mapping[str, Any],
    equation: CSTRConfig,
    step: float | None = None,
) -> tuple[np.ndarray, float]:
    batch_size = int(config["direct_batch_size"])
    conditions = np.column_stack(
        (
            np.full(controls.shape[0], parameters[0]),
            np.full(controls.shape[0], parameters[1]),
            controls,
        )
    ).astype(np.float64)
    predictions = []
    started = time.perf_counter()
    for start in range(0, controls.shape[0], batch_size):
        stop = min(start + batch_size, controls.shape[0])
        predictions.append(
            solve_batch(
                np.repeat(histories, stop - start, axis=0),
                conditions[start:stop],
                float(config["screen_horizon"]),
                int(config["screen_output_points"]),
                float(config["reference_step"] if step is None else step),
                equation,
            )
        )
    return np.concatenate(predictions, axis=0), time.perf_counter() - started


@torch.no_grad()
def operator_screen(
    model: CSTRParaFDEONet,
    histories: np.ndarray,
    parameters: np.ndarray,
    controls: np.ndarray,
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, float, dict[str, float]]:
    conditions = np.column_stack(
        (
            np.full(controls.shape[0], parameters[0]),
            np.full(controls.shape[0], parameters[1]),
            controls,
        )
    ).astype(np.float32)
    times = torch.linspace(
        0.0,
        float(config["screen_horizon"]),
        int(config["screen_output_points"]),
        device=device,
    )
    batch_size = int(config["operator_batch_size"])
    predictions = []
    synchronize(device)
    started = time.perf_counter()
    history_features, trunk_features, cache_times = build_fixed_operator_features(
        model, histories, times, device
    )
    for start in range(0, controls.shape[0], batch_size):
        stop = min(start + batch_size, controls.shape[0])
        q = torch.as_tensor(conditions[start:stop], dtype=torch.float32, device=device)
        predictions.append(
            cached_operator_prediction(
                model, history_features, trunk_features, q
            ).cpu().numpy()
        )
    synchronize(device)
    return np.concatenate(predictions, axis=0), time.perf_counter() - started, cache_times


def normalized_parameter_error(estimate: np.ndarray, truth: np.ndarray, equation: CSTRConfig) -> float:
    spans = equation.inverse_spans
    return float(np.sqrt(np.mean(((estimate - truth) / spans) ** 2)))


def inverse_para_lm(
    model: CSTRParaFDEONet,
    history: np.ndarray,
    control: np.ndarray,
    observations: np.ndarray,
    config: Mapping[str, Any],
    equation: CSTRConfig,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Invert the frozen operator with a projected, damped Gauss--Newton method.

    The history branches and time trunk are evaluated once.  At each LM step only
    the shared condition branch is reevaluated, and its two parameter-Jacobian
    columns are obtained by exact JVPs rather than finite differences.
    """
    bounds = equation.bounds_array[:2]
    points = int(config["parameter_grid_points"])
    k_values = np.linspace(bounds[0, 0], bounds[0, 1], points)
    kappa_values = np.linspace(bounds[1, 0], bounds[1, 1], points)
    kk, pp = np.meshgrid(k_values, kappa_values, indexing="ij")
    parameter_grid = np.column_stack((kk.ravel(), pp.ravel())).astype(np.float32)
    conditions = np.column_stack(
        (
            parameter_grid,
            np.full(parameter_grid.shape[0], control[0]),
            np.full(parameter_grid.shape[0], control[1]),
        )
    ).astype(np.float32)
    times = torch.linspace(
        0.0,
        float(config["observation_horizon"]),
        int(config["observation_points"]),
        device=device,
    )
    target = torch.as_tensor(observations, dtype=torch.float32, device=device)
    objective_scale = (
        model.output_std.detach()
        if bool(config.get("normalize_inverse_objective", True))
        else torch.ones_like(model.output_std)
    )

    synchronize(device)
    started = time.perf_counter()
    history_features, trunk_features, cache_times = build_fixed_operator_features(
        model, history, times, device
    )
    grid_conditions = torch.as_tensor(conditions, dtype=torch.float32, device=device)
    synchronize(device)
    grid_cache_started = time.perf_counter()
    with torch.no_grad():
        grid_condition_features = encode_condition_features(
            model, grid_conditions, detach=True
        )
    synchronize(device)
    grid_condition_cache_seconds = time.perf_counter() - grid_cache_started

    synchronize(device)
    warm_started = time.perf_counter()
    grid_started = time.perf_counter()
    grid_losses: list[torch.Tensor] = []
    batch_size = int(config["operator_batch_size"])
    with torch.no_grad():
        for start in range(0, conditions.shape[0], batch_size):
            stop = min(start + batch_size, conditions.shape[0])
            prediction = predict_from_cached_features(
                model,
                history_features,
                grid_condition_features[start:stop],
                trunk_features,
            )
            grid_losses.append(
                torch.mean(
                    ((prediction - target) / objective_scale) ** 2,
                    dim=(1, 2),
                ).cpu()
            )
    synchronize(device)
    grid_screening_seconds = time.perf_counter() - grid_started
    grid_loss_array = torch.cat(grid_losses).numpy()
    restart_count = min(
        int(config["para_lm_starts"]),
        parameter_grid.shape[0],
    )
    top = np.argsort(grid_loss_array)[:restart_count]
    normalized_initial = 2.0 * (
        parameter_grid[top] - bounds[:, 0]
    ) / (bounds[:, 1] - bounds[:, 0]) - 1.0
    z = torch.as_tensor(normalized_initial, dtype=torch.float32, device=device)
    lower = torch.as_tensor(bounds[:, 0], dtype=torch.float32, device=device)
    span = torch.as_tensor(bounds[:, 1] - bounds[:, 0], dtype=torch.float32, device=device)
    fixed = torch.as_tensor(control, dtype=torch.float32, device=device).expand(
        restart_count, -1
    )
    target_repeated = target.expand(restart_count, -1, -1)

    def residual_function(normalized_parameters: torch.Tensor) -> torch.Tensor:
        physical = lower + 0.5 * (normalized_parameters + 1.0) * span
        q = torch.cat((physical, fixed), dim=1)
        prediction = cached_operator_prediction(
            model, history_features, trunk_features, q
        )
        return ((prediction - target_repeated) / objective_scale).reshape(
            restart_count, -1
        )

    maximum_iterations = int(config["para_lm_max_iterations"])
    minimum_iterations = int(config["para_lm_minimum_iterations"])
    relative_tolerance = float(config["para_lm_relative_objective_tolerance"])
    parameter_tolerance = float(config["para_lm_parameter_tolerance"])
    patience = int(config["para_lm_early_stop_patience"])
    damping = torch.full(
        (restart_count,),
        float(config["para_lm_initial_damping"]),
        dtype=torch.float64,
        device=device,
    )
    damping_increase = float(config["para_lm_damping_increase"])
    damping_decrease = float(config["para_lm_damping_decrease"])
    minimum_damping = float(config["para_lm_minimum_damping"])
    maximum_damping = float(config["para_lm_maximum_damping"])
    identity = torch.eye(2, dtype=torch.float64, device=device).expand(
        restart_count, -1, -1
    )
    previous_best_z: torch.Tensor | None = None
    previous_best_loss: float | None = None
    stable_count = 0
    early_stopped = False
    completed = 0
    accepted_steps = 0

    for iteration in range(1, maximum_iterations + 1):
        current_z = z.detach().requires_grad_(True)
        current_residual = residual_function(current_z)
        jacobian_columns = []
        for parameter_index in range(2):
            tangent = torch.zeros_like(current_z)
            tangent[:, parameter_index] = 1.0
            _, column = torch.autograd.functional.jvp(
                residual_function,
                (current_z,),
                (tangent,),
                create_graph=False,
                strict=False,
            )
            jacobian_columns.append(column.detach())
        residual64 = current_residual.detach().to(torch.float64)
        jacobian64 = torch.stack(jacobian_columns, dim=2).to(torch.float64)
        residual_count = residual64.shape[1]
        objectives = torch.mean(residual64.square(), dim=1)
        normal = torch.einsum("bmi,bmj->bij", jacobian64, jacobian64) / residual_count
        gradient = torch.einsum("bmi,bm->bi", jacobian64, residual64) / residual_count
        scaling = torch.diagonal(normal, dim1=1, dim2=2).clamp_min(1.0e-12)
        system = (
            normal
            + damping[:, None, None] * torch.diag_embed(scaling)
            + 1.0e-12 * identity
        )
        delta = torch.linalg.solve(system, -gradient.unsqueeze(2)).squeeze(2)
        proposed_z = torch.clamp(
            current_z.detach() + delta.to(torch.float32), -1.0, 1.0
        )
        with torch.no_grad():
            proposed_residual = residual_function(proposed_z)
            proposed_objectives = torch.mean(
                proposed_residual.to(torch.float64).square(), dim=1
            )
            accepted = proposed_objectives < objectives
            accepted_steps += int(accepted.sum().cpu())
            z = torch.where(accepted[:, None], proposed_z, current_z.detach())
            updated_objectives = torch.where(
                accepted, proposed_objectives, objectives
            )
            damping = torch.where(
                accepted,
                damping / damping_decrease,
                damping * damping_increase,
            ).clamp(minimum_damping, maximum_damping)
            selected = int(torch.argmin(updated_objectives))
            best_z = z[selected].detach().clone()
            best_loss = float(updated_objectives[selected].cpu())
        completed = iteration

        parameter_change = float("inf")
        relative_loss_change = float("inf")
        if previous_best_z is not None:
            parameter_change = float(
                torch.max(torch.abs(best_z - previous_best_z)).cpu()
            )
        if previous_best_loss is not None:
            relative_loss_change = abs(best_loss - previous_best_loss) / max(
                abs(previous_best_loss), 1.0e-12
            )
        converged = (
            iteration >= minimum_iterations
            and parameter_change < parameter_tolerance
            and relative_loss_change < relative_tolerance
        )
        stable_count = stable_count + 1 if converged else 0
        previous_best_z = best_z
        previous_best_loss = best_loss
        if stable_count >= patience:
            early_stopped = True
            break

    with torch.no_grad():
        final_residual = residual_function(z)
        final_losses = torch.mean(final_residual.square(), dim=1)
        best = int(torch.argmin(final_losses))
        parameters = lower + 0.5 * (z + 1.0) * span
        estimate = parameters[best].cpu().numpy()
        final_loss = float(final_losses[best].cpu())
    synchronize(device)
    warm_online_seconds = time.perf_counter() - warm_started
    elapsed = time.perf_counter() - started
    condition_branch_calls = 2 + 4 * completed
    evaluations = parameter_grid.shape[0] + restart_count * (4 * completed + 1)
    return estimate, {
        "time_seconds": elapsed,
        "warm_online_seconds": warm_online_seconds,
        "optimizer": "projected_damped_gauss_newton",
        "forward_model_calls": int(condition_branch_calls),
        "condition_branch_calls": int(condition_branch_calls),
        "forward_condition_evaluations": int(evaluations),
        "grid_screening_seconds": grid_screening_seconds,
        "grid_best_loss": float(grid_loss_array[top[0]]),
        "final_loss": final_loss,
        "para_lm_starts": int(restart_count),
        "lm_iterations_completed": int(completed),
        "lm_early_stopped": bool(early_stopped),
        "lm_accepted_restart_steps": int(accepted_steps),
        "lm_final_damping_mean": float(damping.mean().cpu()),
        "jacobian_jvp_calls": int(2 * completed),
        "history_cache_seconds": cache_times["history_cache_seconds"],
        "time_trunk_cache_seconds": cache_times["time_trunk_cache_seconds"],
        "grid_condition_branch_cache_seconds": grid_condition_cache_seconds,
        "cached_parameter_branch_only": True,
    }


def inverse_direct_lm(
    history: np.ndarray,
    control: np.ndarray,
    observations: np.ndarray,
    config: Mapping[str, Any],
    equation: CSTRConfig,
    residual_scale: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    bounds = equation.bounds_array[:2]
    rng = np.random.default_rng(int(config["seed"]) + 17)
    unit = np.empty((int(config["lm_starts"]), 2), dtype=np.float64)
    for column in range(2):
        order = rng.permutation(unit.shape[0])
        unit[:, column] = (order + rng.random(unit.shape[0])) / unit.shape[0]
    parameters = bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])
    restart_count = parameters.shape[0]
    damping = np.full(restart_count, float(config["lm_initial_damping"]))
    finite_step = float(config["lm_finite_difference_step"])
    minimum_iterations = int(config["lm_minimum_iterations"])
    maximum_iterations = int(config["lm_max_iterations"])
    tolerance = float(config["lm_relative_objective_tolerance"])
    patience = int(config["lm_early_stop_patience"])
    solver_batch_calls = 0
    trajectory_solves = 0
    stable_count = 0
    previous_best: float | None = None
    stopped_early = False
    completed = 0
    scale = np.ones(2, dtype=np.float64)
    if residual_scale is not None:
        scale = np.asarray(residual_scale, dtype=np.float64)
        if scale.shape != (2,) or np.any(scale <= 0.0):
            raise ValueError("residual_scale must contain two positive values")

    def residuals(candidates: np.ndarray) -> np.ndarray:
        conditions = np.column_stack(
            (
                candidates,
                np.full(candidates.shape[0], control[0]),
                np.full(candidates.shape[0], control[1]),
            )
        ).astype(np.float64)
        prediction = solve_batch(
            np.repeat(history, candidates.shape[0], axis=0),
            conditions,
            float(config["observation_horizon"]),
            int(config["observation_points"]),
            float(config["reference_step"]),
            equation,
        )
        result = (
            (prediction.astype(np.float64) - observations[None])
            / scale[None, None, :]
        ).reshape(
            candidates.shape[0], -1
        )
        if not np.isfinite(result).all():
            raise FloatingPointError("non-finite Direct LM residual")
        return result

    started = time.perf_counter()
    for iteration in range(1, maximum_iterations + 1):
        current_residuals = residuals(parameters)
        solver_batch_calls += 1
        trajectory_solves += restart_count
        objectives = np.sum(current_residuals**2, axis=1)

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
        perturbed_residuals = residuals(perturbed.reshape(4 * restart_count, 2))
        solver_batch_calls += 1
        trajectory_solves += 4 * restart_count
        perturbed_residuals = perturbed_residuals.reshape(
            4, restart_count, current_residuals.shape[1]
        )
        jacobians = np.empty(
            (restart_count, current_residuals.shape[1], 2), dtype=np.float64
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
            gradient = jacobian.T @ current_residuals[restart]
            try:
                delta = np.linalg.solve(system, -gradient)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(system, -gradient, rcond=None)[0]
            proposed[restart] = np.clip(
                parameters[restart] + delta, bounds[:, 0], bounds[:, 1]
            )
        proposed_residuals = residuals(proposed)
        solver_batch_calls += 1
        trajectory_solves += restart_count
        proposed_objectives = np.sum(proposed_residuals**2, axis=1)
        accepted = proposed_objectives < objectives
        parameters[accepted] = proposed[accepted]
        objectives[accepted] = proposed_objectives[accepted]
        damping[accepted] = np.maximum(
            float(config["lm_minimum_damping"]),
            damping[accepted] / float(config["lm_damping_decrease"]),
        )
        damping[~accepted] = np.minimum(
            float(config["lm_maximum_damping"]),
            damping[~accepted] * float(config["lm_damping_increase"]),
        )
        completed = iteration
        best = float(np.min(objectives))
        if previous_best is not None:
            relative_change = abs(best - previous_best) / max(abs(previous_best), 1.0e-12)
            stable_count = (
                stable_count + 1
                if iteration >= minimum_iterations and relative_change < tolerance
                else 0
            )
        previous_best = best
        if stable_count >= patience:
            stopped_early = True
            break

    final_residuals = residuals(parameters)
    solver_batch_calls += 1
    trajectory_solves += restart_count
    objectives = np.sum(final_residuals**2, axis=1)
    selected = int(np.argmin(objectives))
    elapsed = time.perf_counter() - started
    return parameters[selected].copy(), {
        "time_seconds": elapsed,
        "warm_online_seconds": elapsed,
        "forward_model_calls": int(solver_batch_calls),
        "forward_condition_evaluations": int(trajectory_solves),
        "solver_batch_calls": int(solver_batch_calls),
        "direct_trajectory_solves": int(trajectory_solves),
        "lm_iterations_completed": int(completed),
        "lm_early_stopped": bool(stopped_early),
        "lm_final_damping_mean": float(np.mean(damping)),
        "final_loss": float(objectives[selected] / final_residuals.shape[1]),
    }


def interpolate_history(history: np.ndarray, sensors: int) -> np.ndarray:
    old = np.linspace(-1.0, 0.0, history.shape[-1])
    new = np.linspace(-1.0, 0.0, sensors)
    result = np.empty((history.shape[0], 2, sensors), dtype=np.float32)
    for component in range(2):
        result[0, component] = np.interp(new, old, history[0, component])
    return result


def verify_recommendation(
    history: np.ndarray,
    true_parameters: np.ndarray,
    control: np.ndarray,
    config: Mapping[str, Any],
    equation: CSTRConfig,
) -> dict[str, Any]:
    step = float(config["verification_step"])
    sensors = int(round(equation.delay / step)) + 1
    fine_history = interpolate_history(history, sensors)
    try:
        trajectory = solve_batch(
            fine_history,
            np.asarray([[*true_parameters, *control]], dtype=np.float64),
            float(config["screen_horizon"]),
            int(config["screen_output_points"]),
            step,
            equation,
        )
    except (FloatingPointError, OverflowError):
        runaway_marker = float(config.get("runaway_temperature_marker", 1.0e6))
        return {
            "D": float(control[0]),
            "Tc": float(control[1]),
            "true_maximum_temperature": runaway_marker,
            "true_conversion": 0.0,
            "true_temperature_std": runaway_marker,
            "true_safe": False,
            "true_productivity": 0.0,
            "valid_productivity": 0.0,
            "solver_runaway": True,
        }
    features = trajectory_features(trajectory)
    safe = bool(feasible_mask(features, config, margin=False)[0])
    productivity = float(control[0] * features["conversion"][0])
    return {
        "D": float(control[0]),
        "Tc": float(control[1]),
        "true_maximum_temperature": float(features["maximum_temperature"][0]),
        "true_conversion": float(features["conversion"][0]),
        "true_temperature_std": float(features["temperature_std"][0]),
        "true_safe": safe,
        "true_productivity": productivity,
        "valid_productivity": productivity if safe else 0.0,
        "solver_runaway": False,
    }


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(args.output_dir / "pilot.log"), logging.StreamHandler()],
    )
    config = load_json(args.pilot_config)
    training_config = load_json(args.experiment_config)
    equation = CSTRConfig.from_mapping(training_config["equation"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_operator_checkpoint(args.checkpoint, device)
    model.eval()
    for weight in model.parameters():
        weight.requires_grad_(False)
    controls = control_grid(equation, int(config["control_grid_points"]))
    old_parameters = np.asarray(config["old_parameters"], dtype=np.float64)
    true_parameters = np.asarray(config["new_parameters"], dtype=np.float64)

    initial_history = constant_histories(1, 101)
    old_trajectories, old_screen_time = direct_screen(
        initial_history, old_parameters, controls, config, equation
    )
    old_features = trajectory_features(old_trajectories)
    old_index, old_productivity = choose_control(
        controls, old_features, config, margin=True
    )
    old_control = controls[old_index]
    LOGGER.info("old control D=%.6f Tc=%.6f", *old_control)

    old_burn = solve_batch(
        initial_history,
        np.asarray([[*old_parameters, *old_control]], dtype=np.float64),
        30.0,
        3001,
        float(config["reference_step"]),
        equation,
        return_internal=True,
    )
    old_history = old_burn[:, -101:, :].transpose(0, 2, 1)
    observation_internal = solve_batch(
        old_history,
        np.asarray([[*true_parameters, *old_control]], dtype=np.float64),
        float(config["observation_horizon"]),
        int(round(float(config["observation_horizon"]) / float(config["reference_step"]))) + 1,
        float(config["reference_step"]),
        equation,
        return_internal=True,
    )
    observation_indices = np.linspace(
        0,
        observation_internal.shape[1] - 1,
        int(config["observation_points"]),
    ).round().astype(int)
    clean_observations = observation_internal[0, observation_indices]
    rng = np.random.default_rng(int(config["seed"]))
    noise_std = float(config["noise_fraction"]) * model.output_std.detach().cpu().numpy()
    noisy_observations = clean_observations + rng.normal(
        scale=noise_std, size=clean_observations.shape
    )
    planning_history = observation_internal[:, -101:, :].transpose(0, 2, 1)

    para_estimate, para_inverse = inverse_para_lm(
        model, old_history, old_control, noisy_observations, config, equation, device
    )
    direct_estimate, direct_inverse = inverse_direct_lm(
        old_history,
        old_control,
        noisy_observations,
        config,
        equation,
        model.output_std.detach().cpu().numpy()
        if bool(config.get("normalize_inverse_objective", True))
        else None,
    )
    para_inverse["normalized_parameter_error"] = normalized_parameter_error(
        para_estimate, true_parameters, equation
    )
    direct_inverse["normalized_parameter_error"] = normalized_parameter_error(
        direct_estimate, true_parameters, equation
    )
    LOGGER.info("estimates para=%s direct=%s true=%s", para_estimate, direct_estimate, true_parameters)

    para_trajectories, para_screen_time, para_screen_cache = operator_screen(
        model, planning_history, para_estimate, controls, config, device
    )
    direct_trajectories, direct_screen_time = direct_screen(
        planning_history, direct_estimate, controls, config, equation
    )
    oracle_trajectories, oracle_screen_time = direct_screen(
        planning_history, true_parameters, controls, config, equation
    )
    para_features = trajectory_features(para_trajectories)
    direct_features = trajectory_features(direct_trajectories)
    oracle_features = trajectory_features(oracle_trajectories)
    para_index, _ = choose_control(controls, para_features, config, margin=True)
    direct_index, _ = choose_control(controls, direct_features, config, margin=True)
    oracle_index, _ = choose_control(controls, oracle_features, config, margin=False)

    recommendations = {
        "Static": verify_recommendation(
            planning_history, true_parameters, old_control, config, equation
        ),
        "Direct physics": verify_recommendation(
            planning_history, true_parameters, controls[direct_index], config, equation
        ),
        "ParaFDEONet": verify_recommendation(
            planning_history, true_parameters, controls[para_index], config, equation
        ),
        "Oracle": verify_recommendation(
            planning_history, true_parameters, controls[oracle_index], config, equation
        ),
    }
    oracle_productivity = recommendations["Oracle"]["valid_productivity"]
    for values in recommendations.values():
        values["production_loss"] = float(
            1.0 - values["valid_productivity"] / max(oracle_productivity, 1e-12)
        )

    true_safe = feasible_mask(oracle_features, config, margin=False)
    para_safe = feasible_mask(para_features, config, margin=False)
    direct_safe = feasible_mask(direct_features, config, margin=False)
    para_difference = para_trajectories - oracle_trajectories
    para_relative = np.linalg.norm(para_difference.reshape(len(controls), -1), axis=1) / np.maximum(
        np.linalg.norm(oracle_trajectories.reshape(len(controls), -1), axis=1), 1e-12
    )
    map_metrics = {
        "para_safe_accuracy": float(np.mean(para_safe == true_safe)),
        "para_false_safe_count": int(np.sum(para_safe & ~true_safe)),
        "para_false_safe_rate_all": float(np.mean(para_safe & ~true_safe)),
        "direct_safe_accuracy": float(np.mean(direct_safe == true_safe)),
        "direct_false_safe_count": int(np.sum(direct_safe & ~true_safe)),
        "para_trajectory_relative_l2_mean": float(para_relative.mean()),
        "para_trajectory_relative_l2_q95": float(np.quantile(para_relative, 0.95)),
    }
    inversion_rows = [
        {
            "method": "ParaFDEONet",
            "estimated_k": float(para_estimate[0]),
            "estimated_kappa": float(para_estimate[1]),
            **para_inverse,
        },
        {
            "method": "Direct LM",
            "estimated_k": float(direct_estimate[0]),
            "estimated_kappa": float(direct_estimate[1]),
            **direct_inverse,
        },
    ]
    recommendation_rows = [
        {"pipeline": name, **values} for name, values in recommendations.items()
    ]
    summary = {
        "checkpoint_iteration": int(checkpoint["iteration"]),
        "old_parameters": old_parameters.tolist(),
        "true_new_parameters": true_parameters.tolist(),
        "old_control": old_control.tolist(),
        "old_screen_time_seconds": old_screen_time,
        "noise_fraction": float(config["noise_fraction"]),
        "noise_std": noise_std.tolist(),
        "inverse": {"ParaFDEONet": para_inverse, "Direct LM": direct_inverse},
        "screen_time_seconds": {
            "ParaFDEONet": para_screen_time,
            "Direct physics": direct_screen_time,
            "Oracle": oracle_screen_time,
        },
        "para_screen_cache_seconds": para_screen_cache,
        "cycle_time_seconds": {
            "ParaFDEONet": para_inverse["time_seconds"] + para_screen_time,
            "Direct physics": direct_inverse["time_seconds"] + direct_screen_time,
        },
        "screen_condition_evaluations": int(controls.shape[0]),
        "map_metrics": map_metrics,
        "recommendations": recommendations,
    }
    save_json(args.output_dir / "pilot_summary.json", summary)
    save_csv(args.output_dir / "parameter_inversion.csv", inversion_rows)
    save_csv(args.output_dir / "recommendations.csv", recommendation_rows)
    np.savez_compressed(
        args.output_dir / "observation_data.npz",
        old_history=old_history,
        planning_history=planning_history,
        clean_observations=clean_observations,
        noisy_observations=noisy_observations,
        observation_times=np.linspace(
            0.0,
            float(config["observation_horizon"]),
            int(config["observation_points"]),
            dtype=np.float32,
        ),
        old_control=old_control,
    )
    np.savez_compressed(
        args.output_dir / "screen_maps.npz",
        controls=controls,
        para_features=np.column_stack(tuple(para_features.values())),
        direct_features=np.column_stack(tuple(direct_features.values())),
        oracle_features=np.column_stack(tuple(oracle_features.values())),
        true_safe=true_safe,
        para_safe=para_safe,
        direct_safe=direct_safe,
    )
    LOGGER.info("PILOT_COMPLETED summary=%s", summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", type=Path, default=Path("configs/experiment.json"))
    parser.add_argument("--pilot-config", type=Path, default=Path("configs/pilot.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    run(build_parser().parse_args())
