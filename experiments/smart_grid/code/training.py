"""Joint data, initial, sensitivity and physics training for ParaFDEONet."""

from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch

from data import DatasetSplit
from equation import NODE_COUNT, PARAMETER_DIM, PARAMETER_NAMES, STATE_DIM, SmartGridConfig, rhs_torch
from model import MODEL_TYPE, OPERATOR_NETWORK_FORMAT, ParaFDEONet, build_operator, count_parameters
from wandb_tracker import WandBTracker


LOGGER = logging.getLogger(__name__)


def save_json(path: Path, values: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(values), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def save_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot save an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def learning_rate_at(iteration: int, config: Mapping[str, Any]) -> float:
    total = int(config["iterations"])
    warmup = int(config["warmup_iterations"])
    decay_end = int(config.get("learning_rate_decay_iterations", total))
    start = float(config["warmup_learning_rate"])
    peak = float(config["learning_rate"])
    minimum = float(config["minimum_learning_rate"])
    if iteration <= warmup:
        return start + (peak - start) * iteration / max(warmup, 1)
    if iteration >= decay_end:
        return minimum
    progress = (iteration - warmup) / max(decay_end - warmup, 1)
    return minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * progress))


def physics_weight_at(iteration: int, config: Mapping[str, Any]) -> float:
    pretrain = int(config["data_pretrain_iterations"])
    ramp = int(config["physics_ramp_iterations"])
    target = float(config["physics_weight"])
    if iteration <= pretrain:
        return 0.0
    return target * min(1.0, (iteration - pretrain) / max(ramp, 1))


def interpolate_history_torch(
    histories: torch.Tensor,
    query_times: torch.Tensor,
    maximum_history: float,
) -> torch.Tensor:
    """Linearly interpolate physical histories at batched query times."""
    if histories.ndim != 3 or histories.shape[1] != STATE_DIM:
        raise ValueError("histories must have shape [B,8,M]")
    if query_times.ndim != 2 or query_times.shape[0] != histories.shape[0]:
        raise ValueError("query_times must have shape [B,Q]")
    sensors = histories.shape[2]
    positions = ((query_times + maximum_history) / maximum_history).clamp(0.0, 1.0)
    positions = positions * (sensors - 1)
    lower = positions.floor().long().clamp(0, sensors - 1)
    upper = (lower + 1).clamp(0, sensors - 1)
    weight = positions - lower.to(positions.dtype)
    transposed = histories.transpose(1, 2)
    lower_values = torch.gather(
        transposed, 1, lower.unsqueeze(-1).expand(-1, -1, STATE_DIM)
    )
    upper_values = torch.gather(
        transposed, 1, upper.unsqueeze(-1).expand(-1, -1, STATE_DIM)
    )
    return lower_values + weight.unsqueeze(-1) * (upper_values - lower_values)


def operator_physics_residual(
    model: ParaFDEONet,
    histories: torch.Tensor,
    parameters: torch.Tensor,
    times: torch.Tensor,
    equation: SmartGridConfig,
    residual_rms: torch.Tensor,
) -> torch.Tensor:
    """Return the eight physical residuals divided by training-only RMS scales."""
    if not times.requires_grad:
        raise ValueError("physics times must require gradients")
    history_features = model.encode_histories(histories)
    parameter_features = model.encode_parameters(parameters)
    current = model.decode_features(
        history_features,
        parameter_features,
        model.encode_times(times, histories.shape[0]),
    )
    derivatives = []
    for component in range(STATE_DIM):
        derivative = torch.autograd.grad(
            current[..., component],
            times,
            grad_outputs=torch.ones_like(current[..., component]),
            create_graph=True,
            retain_graph=True,
        )[0]
        derivatives.append(derivative)
    time_derivative = torch.stack(derivatives, dim=-1)
    delayed_times = times - equation.fixed_delay
    history_values = interpolate_history_torch(histories, delayed_times, equation.maximum_history)
    predicted_values = model.decode_features(
        history_features,
        parameter_features,
        model.encode_times(delayed_times.clamp_min(0.0), histories.shape[0]),
    )
    delayed = torch.where((delayed_times <= 0.0).unsqueeze(-1), history_values, predicted_values)
    physical_residual = time_derivative - rhs_torch(current, delayed, parameters, equation)
    return physical_residual / residual_rms[None, None, :]


def normalized_model_parameter_sensitivities(
    model: ParaFDEONet,
    histories: torch.Tensor,
    parameters: torch.Tensor,
    times: torch.Tensor,
    history_features: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiate physical predictions and scale columns by parameter widths."""
    if parameters.ndim != 2 or parameters.shape[1] != PARAMETER_DIM:
        raise ValueError("parameters must have shape [B,3]")

    cached_history = (
        model.encode_histories(histories) if history_features is None else history_features
    )
    trunk_features = model.encode_times(times, histories.shape[0])

    def evaluate(parameter_values: torch.Tensor) -> torch.Tensor:
        parameter_features = model.encode_parameters(parameter_values)
        return model.decode_features(cached_history, parameter_features, trunk_features)

    columns = []
    for column in range(PARAMETER_DIM):
        tangent = torch.zeros_like(parameters)
        tangent[:, column] = 1.0
        _, derivative = torch.func.jvp(evaluate, (parameters,), (tangent,), strict=True)
        columns.append(model.physical_parameter_span[column] * derivative)
    return torch.stack(columns, dim=-1)


def evaluate_parameter_sensitivities(
    model: ParaFDEONet,
    split: DatasetSplit,
    device: torch.device,
    batch_size: int,
    case_count: int,
    point_count: int,
) -> dict[str, float]:
    """Compare interval-scaled model Jacobians with finite-difference labels."""
    labels = split.normalized_parameter_sensitivities
    if labels is None:
        raise RuntimeError("parameter-Jacobian reporting requires test sensitivity labels")
    case_count = min(int(case_count), split.histories.shape[0])
    point_count = min(int(point_count), split.output_times.size)
    if case_count < 1 or point_count < 1:
        raise ValueError("Jacobian evaluation counts must be positive")
    case_indices = np.linspace(0, split.histories.shape[0] - 1, case_count, dtype=np.int64)
    time_indices = np.linspace(0, split.output_times.size - 1, point_count, dtype=np.int64)
    times = torch.as_tensor(split.output_times[time_indices], dtype=torch.float32, device=device)
    predictions = []
    model.eval()
    for start in range(0, case_count, batch_size):
        selected = case_indices[start : start + batch_size]
        histories = torch.as_tensor(split.histories[selected], dtype=torch.float32, device=device)
        parameters = torch.as_tensor(split.parameters[selected], dtype=torch.float32, device=device)
        values = normalized_model_parameter_sensitivities(model, histories, parameters, times)
        predictions.append(values.detach().cpu().numpy())
    prediction = np.concatenate(predictions, axis=0).astype(np.float64)
    target = labels[case_indices][:, time_indices].astype(np.float64)
    error = (prediction - target) / split.state_std[None, None, :, None]
    metrics = {
        "case_count": int(case_count),
        "time_point_count": int(point_count),
        "normalized_mse": float(np.mean(error**2)),
    }
    for column, name in enumerate(PARAMETER_NAMES):
        column_error = error[..., column]
        column_target = target[..., column] / split.state_std[None, None, :]
        metrics[f"{name}_normalized_mse"] = float(np.mean(column_error**2))
        metrics[f"{name}_relative_l2"] = float(
            np.linalg.norm(column_error.ravel())
            / max(np.linalg.norm(column_target.ravel()), 1e-12)
        )
    return metrics


@torch.no_grad()
def evaluate_operator(
    model: ParaFDEONet,
    split: DatasetSplit,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, float], np.ndarray, list[dict[str, float]]]:
    """Evaluate physical and gauge-independent errors on one fixed split."""
    model.eval()
    times = torch.as_tensor(split.output_times, dtype=torch.float32, device=device)
    predictions = []
    for start in range(0, split.histories.shape[0], batch_size):
        histories = torch.as_tensor(split.histories[start : start + batch_size], device=device)
        parameters = torch.as_tensor(split.parameters[start : start + batch_size], device=device)
        predictions.append(model(histories, parameters, times).cpu().numpy())
    prediction = np.concatenate(predictions, axis=0).astype(np.float64)
    truth = split.solutions.astype(np.float64)
    difference = prediction - truth
    standardized = difference / split.state_std[None, None, :]
    case_mse = np.mean(difference**2, axis=(1, 2))
    case_normalized_mse = np.mean(standardized**2, axis=(1, 2))
    case_relative = np.linalg.norm(difference.reshape(difference.shape[0], -1), axis=1) / np.maximum(
        np.linalg.norm(truth.reshape(truth.shape[0], -1), axis=1), 1e-12
    )
    predicted_angle_difference = prediction[..., 1:NODE_COUNT] - prediction[..., :1]
    true_angle_difference = truth[..., 1:NODE_COUNT] - truth[..., :1]
    angle_error = predicted_angle_difference - true_angle_difference
    case_angle_relative = np.linalg.norm(angle_error.reshape(angle_error.shape[0], -1), axis=1) / np.maximum(
        np.linalg.norm(true_angle_difference.reshape(true_angle_difference.shape[0], -1), axis=1), 1e-12
    )
    frequency_error = difference[..., NODE_COUNT:]
    true_frequency = truth[..., NODE_COUNT:]
    case_frequency_relative = np.linalg.norm(frequency_error.reshape(frequency_error.shape[0], -1), axis=1) / np.maximum(
        np.linalg.norm(true_frequency.reshape(true_frequency.shape[0], -1), axis=1), 1e-12
    )
    state_mse = np.mean(difference**2, axis=1)
    state_relative = np.linalg.norm(difference, axis=1) / np.maximum(np.linalg.norm(truth, axis=1), 1e-12)
    rows = []
    for index in range(truth.shape[0]):
        row: dict[str, float | int] = {
            "case_index": index,
            "coupling_K": float(split.parameters[index, 0]),
            "response_gamma": float(split.parameters[index, 1]),
            "damping_alpha": float(split.parameters[index, 2]),
            "mse": float(case_mse[index]),
            "normalized_mse": float(case_normalized_mse[index]),
            "relative_l2": float(case_relative[index]),
            "relative_angle_difference_l2": float(case_angle_relative[index]),
            "frequency_relative_l2": float(case_frequency_relative[index]),
        }
        for component in range(STATE_DIM):
            row[f"state_{component + 1}_mse"] = float(state_mse[index, component])
            row[f"state_{component + 1}_rmse"] = float(np.sqrt(state_mse[index, component]))
            row[f"state_{component + 1}_relative_l2"] = float(state_relative[index, component])
        rows.append(row)
    metrics = {
        "mse": float(np.mean(case_mse)),
        "normalized_mse": float(np.mean(case_normalized_mse)),
        "relative_l2_mean": float(np.mean(case_relative)),
        "relative_l2_median": float(np.median(case_relative)),
        "relative_angle_difference_l2_mean": float(np.mean(case_angle_relative)),
        "frequency_relative_l2_mean": float(np.mean(case_frequency_relative)),
    }
    for component in range(STATE_DIM):
        metrics[f"state_{component + 1}_mse"] = float(np.mean(state_mse[:, component]))
        metrics[f"state_{component + 1}_rmse"] = float(np.sqrt(np.mean(state_mse[:, component])))
        metrics[f"state_{component + 1}_relative_l2_mean"] = float(np.mean(state_relative[:, component]))
    return metrics, prediction.astype(np.float32), rows


def _atomic_torch_save(path: Path, values: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(values), temporary)
    temporary.replace(path)


def _checkpoint_payload(
    model: ParaFDEONet,
    optimizer: torch.optim.Optimizer | None,
    iteration: int,
    best_iteration: int,
    best_validation: float,
    training_seed: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "network_format": OPERATOR_NETWORK_FORMAT,
        "model_type": MODEL_TYPE,
        "training_seed": int(training_seed),
        "iteration": int(iteration),
        "best_iteration": int(best_iteration),
        "best_validation_normalized_mse": float(best_validation),
        "model_config": model.model_config,
        "model_state": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    return payload


def train_operator(
    train: DatasetSplit,
    validation: DatasetSplit,
    test: DatasetSplit,
    resolved_config: Mapping[str, Any],
    output_dir: Path,
    device: torch.device,
    training_seed: int,
    stage_name: str,
    resume: bool,
) -> dict[str, Any]:
    """Train one ParaFDEONet seed, select by validation and test once."""
    operator_config = resolved_config["operator"]
    data_config = resolved_config["data"]
    equation = SmartGridConfig.from_mapping(resolved_config["equation"])
    torch.manual_seed(int(training_seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(training_seed))
    model = build_operator(
        operator_config,
        data_config,
        train.state_mean.tolist(),
        train.state_std.tolist(),
        int(training_seed),
    ).to(device)
    parameter_count = count_parameters(model)
    if stage_name == "full":
        expected = int(resolved_config["expected_formal_parameter_counts"]["parafdeonet"])
        if parameter_count != expected:
            raise RuntimeError(f"formal parameter count {parameter_count} does not match {expected}")
    LOGGER.info(
        "Starting %s seed %d on %s with %d trainable parameters",
        stage_name, training_seed, device, parameter_count,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(operator_config["learning_rate"]),
        weight_decay=float(operator_config["weight_decay"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker = WandBTracker(
        resolved_config["wandb"], output_dir / "wandb", resolved_config,
        f"smart-grid8d-parafdeonet-seed-{training_seed}-{stage_name}", LOGGER,
    )
    best_path = output_dir / "best_model.pt"
    last_path = output_dir / "last_model.pt"
    history_path = output_dir / "training_history.csv"
    start_iteration = 1
    best_iteration = 0
    best_validation = math.inf
    history_rows: list[dict[str, Any]] = []
    previous_runtime_seconds = 0.0
    if resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint.get("network_format") != OPERATOR_NETWORK_FORMAT:
            raise RuntimeError("checkpoint network format does not match")
        if int(checkpoint.get("training_seed")) != int(training_seed):
            raise RuntimeError("checkpoint training seed does not match")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_iteration = int(checkpoint["iteration"]) + 1
        best_iteration = int(checkpoint["best_iteration"])
        best_validation = float(checkpoint["best_validation_normalized_mse"])
        if history_path.exists():
            with history_path.open("r", encoding="utf-8", newline="") as handle:
                history_rows = list(csv.DictReader(handle))
        metrics_path = output_dir / "metrics.json"
        if metrics_path.exists():
            previous_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            previous_runtime_seconds = float(
                previous_metrics.get("training_runtime_seconds", 0.0)
            )

    state_std = torch.as_tensor(train.state_std, dtype=torch.float32, device=device)
    residual_rms = torch.as_tensor(train.residual_rms, dtype=torch.float32, device=device)
    output_times = torch.as_tensor(train.output_times, dtype=torch.float32, device=device)
    total_iterations = int(operator_config["iterations"])
    validation_every = int(operator_config["validation_every"])
    supervised_batch = int(operator_config["supervised_pair_batch"])
    physics_batch = int(operator_config["physics_pool_batch"])
    sensitivity_points = min(int(operator_config["sensitivity_points"]), train.output_times.size)
    residual_points = int(operator_config["residual_points"])
    loss_sums = {name: 0.0 for name in ("total", "data", "initial", "K", "gamma", "alpha", "physics")}
    interval_count = 0
    started = time.perf_counter()
    success = False
    try:
        for iteration in range(start_iteration, total_iterations + 1):
            model.train()
            rng = np.random.default_rng(int(training_seed) * 1_000_003 + iteration)
            indices = rng.integers(0, train.histories.shape[0], size=supervised_batch)
            histories = torch.as_tensor(train.histories[indices], dtype=torch.float32, device=device)
            parameters = torch.as_tensor(train.parameters[indices], dtype=torch.float32, device=device)
            targets = torch.as_tensor(train.solutions[indices], dtype=torch.float32, device=device)
            history_features = model.encode_histories(histories)
            parameter_features = model.encode_parameters(parameters)
            output_trunk_features = model.encode_times(output_times, histories.shape[0])
            predictions = model.decode_features(
                history_features, parameter_features, output_trunk_features
            )
            data_loss = torch.mean(((predictions - targets) / state_std[None, None, :]) ** 2)
            initial_loss = torch.mean(((predictions[:, 0, :] - targets[:, 0, :]) / state_std[None, :]) ** 2)

            sensitivity_indices = np.sort(
                rng.choice(train.output_times.size, size=sensitivity_points, replace=False)
            )
            sensitivity_times = output_times[sensitivity_indices]
            sensitivity_prediction = normalized_model_parameter_sensitivities(
                model, histories, parameters, sensitivity_times, history_features
            )
            if train.normalized_parameter_sensitivities is None:
                raise RuntimeError("training split has no sensitivity labels")
            sensitivity_target = torch.as_tensor(
                train.normalized_parameter_sensitivities[indices][:, sensitivity_indices],
                dtype=torch.float32,
                device=device,
            )
            sensitivity_error = (
                sensitivity_prediction - sensitivity_target
            ) / state_std[None, None, :, None]
            sensitivity_losses = torch.mean(sensitivity_error**2, dim=(0, 1, 2))

            current_physics_weight = physics_weight_at(iteration, operator_config)
            if current_physics_weight > 0.0:
                history_indices = rng.integers(0, train.histories.shape[0], size=physics_batch)
                parameter_indices = rng.integers(0, train.parameters.shape[0], size=physics_batch)
                physics_histories = torch.as_tensor(
                    train.histories[history_indices], dtype=torch.float32, device=device
                )
                physics_parameters = torch.as_tensor(
                    train.parameters[parameter_indices], dtype=torch.float32, device=device
                )
                physics_times = torch.as_tensor(
                    rng.uniform(0.0, float(data_config["horizon"]), size=(physics_batch, residual_points)),
                    dtype=torch.float32,
                    device=device,
                ).requires_grad_(True)
                residual = operator_physics_residual(
                    model, physics_histories, physics_parameters, physics_times,
                    equation, residual_rms,
                )
                physics_loss = torch.mean(residual**2)
            else:
                physics_loss = torch.zeros((), dtype=torch.float32, device=device)

            total_loss = (
                float(operator_config["data_weight"]) * data_loss
                + float(operator_config["initial_weight"]) * initial_loss
                + float(operator_config["coupling_sensitivity_weight"]) * sensitivity_losses[0]
                + float(operator_config["response_sensitivity_weight"]) * sensitivity_losses[1]
                + float(operator_config["damping_sensitivity_weight"]) * sensitivity_losses[2]
                + current_physics_weight * physics_loss
            )
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"non-finite training loss at iteration {iteration}")
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            logical_epoch = math.ceil(iteration / int(operator_config["steps_per_epoch"]))
            gradient_interval = int(resolved_config["wandb"].get("gradient_interval_epochs", 1))
            if (
                bool(resolved_config["wandb"].get("log_gradients", False))
                and iteration % int(operator_config["steps_per_epoch"]) == 0
                and logical_epoch % gradient_interval == 0
            ):
                tracker.log_gradients(model, logical_epoch)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(operator_config["gradient_clip"])
            )
            current_learning_rate = learning_rate_at(iteration, operator_config)
            for group in optimizer.param_groups:
                group["lr"] = current_learning_rate
            optimizer.step()

            values = {
                "total": total_loss,
                "data": data_loss,
                "initial": initial_loss,
                "K": sensitivity_losses[0],
                "gamma": sensitivity_losses[1],
                "alpha": sensitivity_losses[2],
                "physics": physics_loss,
            }
            for name, value in values.items():
                loss_sums[name] += float(value.detach().cpu())
            interval_count += 1

            if iteration % validation_every == 0 or iteration == total_iterations:
                validation_metrics, _, _ = evaluate_operator(
                    model, validation, device, int(operator_config["evaluation_batch_size"])
                )
                row = {
                    "iteration": iteration,
                    "train_total_loss": loss_sums["total"] / interval_count,
                    "train_data_loss": loss_sums["data"] / interval_count,
                    "train_initial_loss": loss_sums["initial"] / interval_count,
                    "train_K_sensitivity_loss": loss_sums["K"] / interval_count,
                    "train_gamma_sensitivity_loss": loss_sums["gamma"] / interval_count,
                    "train_alpha_sensitivity_loss": loss_sums["alpha"] / interval_count,
                    "train_physics_loss": loss_sums["physics"] / interval_count,
                    "physics_weight": current_physics_weight,
                    "learning_rate": current_learning_rate,
                    "gradient_norm": float(gradient_norm.detach().cpu()),
                    "validation_mse": validation_metrics["mse"],
                    "validation_normalized_mse": validation_metrics["normalized_mse"],
                    "validation_relative_l2": validation_metrics["relative_l2_mean"],
                    "validation_relative_angle_difference_l2": validation_metrics["relative_angle_difference_l2_mean"],
                    "validation_frequency_relative_l2": validation_metrics["frequency_relative_l2_mean"],
                }
                history_rows.append(row)
                save_csv(history_path, history_rows)
                LOGGER.info(
                    "iteration %d/%d | train %.4e | validation normalized MSE %.4e | lr %.3e",
                    iteration,
                    total_iterations,
                    row["train_total_loss"],
                    row["validation_normalized_mse"],
                    row["learning_rate"],
                )
                tracker.log(
                    {f"train/{key}": float(value) for key, value in row.items() if key != "iteration"},
                    logical_epoch,
                )
                if validation_metrics["normalized_mse"] < best_validation:
                    best_validation = float(validation_metrics["normalized_mse"])
                    best_iteration = iteration
                    _atomic_torch_save(
                        best_path,
                        _checkpoint_payload(
                            model, None, iteration, best_iteration,
                            best_validation, training_seed,
                        ),
                    )
                _atomic_torch_save(
                    last_path,
                    _checkpoint_payload(
                        model, optimizer, iteration, best_iteration,
                        best_validation, training_seed,
                    ),
                )
                loss_sums = {name: 0.0 for name in loss_sums}
                interval_count = 0

        if not best_path.exists():
            raise RuntimeError("training completed without a best checkpoint")
        best = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best["model_state"])
        test_metrics, test_predictions, test_rows = evaluate_operator(
            model, test, device, int(operator_config["evaluation_batch_size"])
        )
        np.save(output_dir / "test_predictions.npy", test_predictions)
        save_csv(output_dir / "forward_per_case_metrics.csv", test_rows)
        sensitivity_metrics = None
        reporting_config = resolved_config["reporting"]
        if bool(reporting_config.get("report_parameter_jacobian_metrics", False)):
            sensitivity_metrics = evaluate_parameter_sensitivities(
                model,
                test,
                device,
                min(int(operator_config["evaluation_batch_size"]), 16),
                int(reporting_config["parameter_jacobian_evaluation_cases"]),
                int(reporting_config["parameter_jacobian_evaluation_points"]),
            )
        metrics = {
            "model_type": MODEL_TYPE,
            "training_seed": int(training_seed),
            "parameter_count": int(parameter_count),
            "best_iteration": int(best_iteration),
            "best_validation_normalized_mse": float(best_validation),
            "training_runtime_seconds": float(
                previous_runtime_seconds + time.perf_counter() - started
            ),
            "learned_parameters": list(PARAMETER_NAMES),
            "test": test_metrics,
            "test_parameter_sensitivity": sensitivity_metrics,
        }
        save_json(output_dir / "metrics.json", metrics)
        LOGGER.info(
            "Finished seed %d | best iteration %d | test normalized MSE %.4e",
            training_seed, best_iteration, test_metrics["normalized_mse"],
        )
        success = True
        return metrics
    finally:
        tracker.finish(success)
