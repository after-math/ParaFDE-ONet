"""Formal data, interface, normalized-sensitivity and physics training."""

from __future__ import annotations

import csv
import json
import logging
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping

import numpy as np
import torch

from .data import DatasetSplit, normalization_statistics, resolve_sizes
from .equation import INVERSE_DIM, STATE_DIM, CSTRConfig, rhs_torch
from .model import (
    NETWORK_FORMAT,
    CSTRParaFDEONet,
    build_operator,
    count_trainable_parameters,
    load_operator_checkpoint,
)


LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def physics_weight_at(iteration: int, config: Mapping[str, Any]) -> float:
    pretrain = int(config["data_pretrain_iterations"])
    ramp = int(config["physics_ramp_iterations"])
    final = float(config["physics_weight"])
    if iteration <= pretrain:
        return 0.0
    if iteration < pretrain + ramp:
        return final * (iteration - pretrain) / max(ramp, 1)
    return final


def learning_rate_at(iteration: int, total: int, config: Mapping[str, Any]) -> float:
    peak = float(config["learning_rate"])
    start = float(config["warmup_learning_rate"])
    minimum = float(config["minimum_learning_rate"])
    warmup = int(config["warmup_iterations"])
    if iteration <= warmup:
        return start + (peak - start) * iteration / max(warmup, 1)
    progress = (iteration - warmup) / max(total - warmup, 1)
    return minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * progress))


def interpolate_history_torch(
    histories: torch.Tensor, query_times: torch.Tensor, delay: float
) -> torch.Tensor:
    sensors = histories.shape[-1]
    clipped = query_times.clamp(-delay, 0.0)
    positions = (clipped + delay) * (sensors - 1) / delay
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
    model: CSTRParaFDEONet,
    histories: torch.Tensor,
    conditions: torch.Tensor,
    times: torch.Tensor,
    equation: CSTRConfig,
) -> torch.Tensor:
    if not times.requires_grad:
        raise ValueError("physics times must require gradients")
    current = model(histories, conditions, times)
    derivatives = []
    for component in range(STATE_DIM):
        derivatives.append(
            torch.autograd.grad(
                current[..., component],
                times,
                grad_outputs=torch.ones_like(current[..., component]),
                create_graph=True,
                retain_graph=True,
            )[0]
        )
    time_derivative = torch.stack(derivatives, dim=-1)
    delayed_times = times - equation.delay
    delayed_history = interpolate_history_torch(histories, delayed_times, equation.delay)
    delayed_operator = model(histories, conditions, delayed_times.clamp_min(0.0))
    delayed = torch.where(
        (delayed_times <= 0.0).unsqueeze(-1), delayed_history, delayed_operator
    )
    residual = time_derivative - rhs_torch(current, delayed, conditions, equation)
    return residual / model.output_std


def normalized_model_sensitivities(
    model: CSTRParaFDEONet,
    histories: torch.Tensor,
    conditions: torch.Tensor,
    times: torch.Tensor,
) -> torch.Tensor:
    if conditions.ndim != 2 or conditions.shape[1] != 4:
        raise ValueError("conditions must have shape [B,4]")

    def evaluate(values: torch.Tensor) -> torch.Tensor:
        return model(histories, values, times)

    derivatives = []
    for condition_index in range(INVERSE_DIM):
        tangent = torch.zeros_like(conditions)
        tangent[:, condition_index] = 1.0
        _, derivative = torch.func.jvp(
            evaluate, (conditions,), (tangent,), strict=True
        )
        derivatives.append(model.condition_span[condition_index] * derivative)
    return torch.stack(derivatives, dim=-1)


@torch.no_grad()
def evaluate_operator(
    model: CSTRParaFDEONet,
    split: DatasetSplit,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, float], np.ndarray, list[dict[str, float]]]:
    model.eval()
    times = torch.as_tensor(split.output_times, dtype=torch.float32, device=device)
    predictions: list[np.ndarray] = []
    for start in range(0, split.histories.shape[0], batch_size):
        histories = torch.as_tensor(
            split.histories[start : start + batch_size], dtype=torch.float32, device=device
        )
        conditions = torch.as_tensor(
            split.conditions[start : start + batch_size], dtype=torch.float32, device=device
        )
        predictions.append(model(histories, conditions, times).cpu().numpy())
    prediction = np.concatenate(predictions, axis=0).astype(np.float64)
    truth = split.solutions.astype(np.float64)
    difference = prediction - truth
    case_mse = np.mean(difference**2, axis=(1, 2))
    case_relative = np.linalg.norm(difference.reshape(difference.shape[0], -1), axis=1) / np.maximum(
        np.linalg.norm(truth.reshape(truth.shape[0], -1), axis=1), 1e-12
    )
    state_mse = np.mean(difference**2, axis=1)
    state_relative = np.linalg.norm(difference, axis=1) / np.maximum(
        np.linalg.norm(truth, axis=1), 1e-12
    )
    zero_mse = float(np.mean(truth**2))
    model_mse = float(np.mean(case_mse))
    rows = []
    for index in range(case_mse.size):
        rows.append(
            {
                "case_index": int(index),
                "mse": float(case_mse[index]),
                "relative_l2": float(case_relative[index]),
                "c_mse": float(state_mse[index, 0]),
                "T_mse": float(state_mse[index, 1]),
                "c_relative_l2": float(state_relative[index, 0]),
                "T_relative_l2": float(state_relative[index, 1]),
            }
        )
    metrics = {
        "mse": model_mse,
        "relative_l2_mean": float(case_relative.mean()),
        "relative_l2_median": float(np.median(case_relative)),
        "c_mse": float(state_mse[:, 0].mean()),
        "T_mse": float(state_mse[:, 1].mean()),
        "c_relative_l2_mean": float(state_relative[:, 0].mean()),
        "T_relative_l2_mean": float(state_relative[:, 1].mean()),
        "zero_predictor_mse": zero_mse,
        "zero_predictor_improvement": 1.0 - model_mse / max(zero_mse, 1e-12),
    }
    return metrics, prediction.astype(np.float32), rows


def save_json(path: Path, values: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(dict(values), indent=2), encoding="utf-8")
    temporary.replace(path)


def save_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot save empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def optimizer_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def train_operator(
    splits: Mapping[str, DatasetSplit],
    config: Mapping[str, Any],
    output_dir: Path,
    device: torch.device,
    smoke: bool,
    resume: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_config, operator_config = resolve_sizes(config, smoke)
    training_seed = int(config["randomness"]["training_seed"])
    set_seed(training_seed)
    normalization = normalization_statistics(splits["train"])
    model = build_operator(
        data_config,
        operator_config,
        config["equation"],
        normalization,
        device,
        training_seed,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(operator_config["learning_rate"]),
        weight_decay=float(operator_config["weight_decay"]),
    )
    total_iterations = int(operator_config["iterations"])
    start_iteration = 1
    best_validation = float("inf")
    best_iteration = 0
    history_rows: list[dict[str, float]] = []
    last_path = output_dir / "last_model.pt"
    best_path = output_dir / "best_model.pt"
    if resume:
        if not last_path.exists():
            raise FileNotFoundError("resume requested but last_model.pt is absent")
        payload = torch.load(last_path, map_location=device, weights_only=False)
        if payload.get("network_format") != NETWORK_FORMAT:
            raise RuntimeError("resume checkpoint has incompatible format")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        optimizer_to_device(optimizer, device)
        start_iteration = int(payload["iteration"]) + 1
        best_validation = float(payload["best_validation"])
        best_iteration = int(payload["best_iteration"])
        history_rows = list(payload["history"])

    train = splits["train"]
    if train.normalized_sensitivities is None:
        raise ValueError("training sensitivity labels are absent")
    expected = (*train.solutions.shape, INVERSE_DIM)
    if train.normalized_sensitivities.shape != expected:
        raise ValueError(f"sensitivity shape mismatch: {train.normalized_sensitivities.shape} != {expected}")
    full_times = torch.as_tensor(train.output_times, dtype=torch.float32, device=device)
    equation = CSTRConfig.from_mapping(config["equation"])
    rng = np.random.default_rng(training_seed + 1000)
    batch_size = int(operator_config["supervised_pair_batch"])
    sensitivity_points = min(int(operator_config["sensitivity_points"]), full_times.numel())
    pool_batch = int(operator_config["physics_pool_batch"])
    residual_points = int(operator_config["residual_points"])
    validation_every = int(operator_config["validation_every"])
    evaluation_batch = int(operator_config["evaluation_batch_size"])
    start_time = time.perf_counter()

    for iteration in range(start_iteration, total_iterations + 1):
        lr = learning_rate_at(iteration, total_iterations, operator_config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        indices = rng.choice(train.histories.shape[0], size=batch_size, replace=False)
        histories = torch.as_tensor(train.histories[indices], dtype=torch.float32, device=device)
        conditions = torch.as_tensor(train.conditions[indices], dtype=torch.float32, device=device)
        targets = torch.as_tensor(train.solutions[indices], dtype=torch.float32, device=device)
        prediction = model(histories, conditions, full_times)
        data_loss = torch.mean(((prediction - targets) / model.output_std) ** 2)
        interface_prediction = model(histories, conditions, full_times[:1])[:, 0, :]
        interface_loss = torch.mean(
            ((interface_prediction - histories[:, :, -1]) / model.output_std) ** 2
        )

        sensitivity_indices = np.sort(
            rng.choice(full_times.numel(), size=sensitivity_points, replace=False)
        )
        sensitivity_index_tensor = torch.as_tensor(
            sensitivity_indices, dtype=torch.long, device=device
        )
        sensitivity_times = full_times[sensitivity_index_tensor]
        sensitivity_targets = torch.as_tensor(
            train.normalized_sensitivities[indices][:, sensitivity_indices],
            dtype=torch.float32,
            device=device,
        )
        sensitivity_prediction = normalized_model_sensitivities(
            model, histories, conditions, sensitivity_times
        )
        sensitivity_difference = (
            sensitivity_prediction - sensitivity_targets
        ) / model.output_std.view(1, 1, STATE_DIM, 1)
        k_sensitivity_loss = torch.mean(sensitivity_difference[..., 0] ** 2)
        kappa_sensitivity_loss = torch.mean(sensitivity_difference[..., 1] ** 2)

        active_physics_weight = physics_weight_at(iteration, operator_config)
        if active_physics_weight > 0.0:
            history_rows_indices = rng.choice(train.histories.shape[0], size=pool_batch, replace=False)
            condition_rows_indices = rng.choice(train.conditions.shape[0], size=pool_batch, replace=False)
            physics_histories_np = np.repeat(
                train.histories[history_rows_indices], pool_batch, axis=0
            )
            physics_conditions_np = np.tile(
                train.conditions[condition_rows_indices], (pool_batch, 1)
            )
            physics_histories = torch.as_tensor(
                physics_histories_np, dtype=torch.float32, device=device
            )
            physics_conditions = torch.as_tensor(
                physics_conditions_np, dtype=torch.float32, device=device
            )
            physics_times = (
                torch.rand(
                    physics_histories.shape[0], residual_points, device=device
                )
                * float(data_config["prediction_horizon"])
            ).requires_grad_(True)
            residual = operator_physics_residual(
                model, physics_histories, physics_conditions, physics_times, equation
            )
            physics_loss = torch.mean(residual**2)
        else:
            physics_loss = torch.zeros((), device=device)

        total_loss = (
            float(operator_config["data_weight"]) * data_loss
            + float(operator_config["interface_weight"]) * interface_loss
            + float(operator_config["k_sensitivity_weight"]) * k_sensitivity_loss
            + float(operator_config["kappa_sensitivity_weight"]) * kappa_sensitivity_loss
            + active_physics_weight * physics_loss
        )
        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"non-finite training loss at iteration {iteration}")
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(operator_config["gradient_clip"])
        )
        optimizer.step()

        if iteration == 1 or iteration % validation_every == 0 or iteration == total_iterations:
            validation_metrics, _, _ = evaluate_operator(
                model, splits["validation"], device, evaluation_batch
            )
            row = {
                "iteration": float(iteration),
                "total_loss": float(total_loss.detach().cpu()),
                "data_loss": float(data_loss.detach().cpu()),
                "interface_loss": float(interface_loss.detach().cpu()),
                "k_sensitivity_loss": float(k_sensitivity_loss.detach().cpu()),
                "kappa_sensitivity_loss": float(kappa_sensitivity_loss.detach().cpu()),
                "physics_loss": float(physics_loss.detach().cpu()),
                "physics_weight": float(active_physics_weight),
                "learning_rate": float(lr),
                "gradient_norm": float(torch.as_tensor(gradient_norm).detach().cpu()),
                "validation_relative_l2": validation_metrics["relative_l2_mean"],
                "validation_mse": validation_metrics["mse"],
            }
            history_rows.append(row)
            improved = validation_metrics["mse"] < best_validation
            if improved:
                best_validation = validation_metrics["mse"]
                best_iteration = iteration
            payload = {
                "network_format": NETWORK_FORMAT,
                "model_config": model.model_config(),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "iteration": iteration,
                "best_validation": best_validation,
                "checkpoint_selection_metric": "validation_mse",
                "best_iteration": best_iteration,
                "training_seed": training_seed,
                "history": history_rows,
                "config": dict(config),
            }
            save_checkpoint(last_path, payload)
            if improved:
                save_checkpoint(best_path, payload)
            save_csv(output_dir / "training_history.csv", history_rows)
            LOGGER.info(
                "iteration=%d val_rel_l2=%.6f data=%.6g interface=%.6g sens=(%.6g,%.6g) physics=%.6g",
                iteration,
                validation_metrics["relative_l2_mean"],
                row["data_loss"],
                row["interface_loss"],
                row["k_sensitivity_loss"],
                row["kappa_sensitivity_loss"],
                row["physics_loss"],
            )
            model.train()

    best_model, best_payload = load_operator_checkpoint(best_path, device)
    validation_metrics, _, _ = evaluate_operator(
        best_model, splits["validation"], device, evaluation_batch
    )
    test_metrics, test_predictions, test_rows = evaluate_operator(
        best_model, splits["test"], device, evaluation_batch
    )
    save_json(output_dir / "validation_metrics.json", validation_metrics)
    save_json(output_dir / "test_metrics.json", test_metrics)
    save_csv(output_dir / "test_per_case_metrics.csv", test_rows)
    np.save(output_dir / "test_predictions.npy", test_predictions)
    acceptance = config["acceptance"]
    checks = {
        "validation_relative_l2": validation_metrics["relative_l2_mean"]
        <= float(acceptance["validation_relative_l2_max"]),
        "test_relative_l2": test_metrics["relative_l2_mean"]
        <= float(acceptance["test_relative_l2_max"]),
        "test_temperature_relative_l2": test_metrics["T_relative_l2_mean"]
        <= float(acceptance["test_temperature_relative_l2_max"]),
        "zero_predictor_improvement": test_metrics["zero_predictor_improvement"]
        >= float(acceptance["zero_predictor_improvement_minimum"]),
    }
    summary = {
        "parameter_count": count_trainable_parameters(best_model),
        "best_iteration": int(best_payload["iteration"]),
        "best_validation_mse": float(best_payload["best_validation"]),
        "elapsed_seconds_this_run": time.perf_counter() - start_time,
        "validation": validation_metrics,
        "test": test_metrics,
        "checks": checks,
        "passed": all(checks.values()),
        "device": str(device),
    }
    save_json(output_dir / "training_summary.json", summary)
    return summary
