"""Continuous, explicit train--validation--test workflow for all four operators."""

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

from data import DatasetSplit
from equation import NicholsonConfig, STATE_DIM, delay_torch, rhs_torch
from model import (
    EXPECTED_FORMAL_PARAMETER_COUNTS,
    MODEL_DISPLAY_NAMES,
    OPERATOR_NETWORK_FORMAT,
    Operator4D,
    build_operator,
    count_trainable_parameters,
    load_operator_checkpoint,
)
from wandb_tracker import WandBTracker


LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set Python, NumPy, Torch and CUDA random seeds once per model job.

    ``seed`` is the sole training seed, e.g. 20260901. The function returns
    ``None`` and changes all listed process-local RNG states.  CUDA deterministic
    kernels are requested where available without forcing unsupported algorithms.
    ``train_operator`` calls it before model construction.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def learning_rate_at(iteration: int, config: Mapping[str, Any]) -> float:
    """Evaluate the common linear-warmup plus cosine-decay schedule.

    ``iteration`` is one-based and ``config`` supplies total/warmup and three LR
    values.  The return is a positive float; for example iteration one is near the
    warmup LR and the final iteration is the minimum LR.  No state changes.
    ``train_operator`` applies it before every optimizer step.
    """
    total = int(config["iterations"])
    warmup = int(config["warmup_iterations"])
    start = float(config["warmup_learning_rate"])
    peak = float(config["learning_rate"])
    minimum = float(config["minimum_learning_rate"])
    if iteration <= warmup:
        fraction = iteration / max(warmup, 1)
        return start + fraction * (peak - start)
    fraction = min(max((iteration - warmup) / max(total - warmup, 1), 0.0), 1.0)
    return minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * fraction))


def physics_weight_at(iteration: int, config: Mapping[str, Any]) -> float:
    """Return the pretrain/ramp/final physics-loss weight.

    ``iteration`` is one-based and the mapping gives pretrain, ramp and final
    weight.  The result is zero during data pretraining, then linear, then constant;
    e.g. formal iteration 18000 reaches 0.0309308.  There is no side effect.
    ``train_operator`` calls it every iteration.
    """
    pretrain = int(config["data_pretrain_iterations"])
    ramp = int(config["physics_ramp_iterations"])
    final = float(config["physics_weight"])
    if iteration <= pretrain:
        return 0.0
    if iteration < pretrain + ramp:
        return final * (iteration - pretrain) / max(ramp, 1)
    return final


def interpolate_history_torch(
    histories: torch.Tensor, query_times: torch.Tensor, maximum_history: float
) -> torch.Tensor:
    """Linearly interpolate uniform sensor histories without leaving autograd.

    Histories are ``[B,4,M]`` and queries ``[B,Q]`` in ``[-d_bar,0]``. The return
    is ``[B,Q,4]``; query zero returns the final sensor. For example B=49,Q=48
    produces ``[49,48,4]``. Only index selection is nondifferentiable; interpolation
    weights retain time derivatives.  ``operator_physics_residual`` calls it.
    """
    sensors = histories.shape[-1]
    clipped = query_times.clamp(-maximum_history, 0.0)
    positions = (clipped + maximum_history) * (sensors - 1) / maximum_history
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
    model: Operator4D,
    histories: torch.Tensor,
    parameters: torch.Tensor,
    times: torch.Tensor,
    equation: NicholsonConfig,
) -> torch.Tensor:
    """Compute all four time-differentiated Nicholson DDE residuals.

    Inputs are a model, histories ``[B,4,M]``, parameters ``[B,2]`` and leaf query
    times ``[B,R]``. The return is ``[B,R,4]`` equal to time derivative minus RHS.
    Delayed times below zero interpolate history; nonnegative delayed times re-query
    the same operator.  For B=16,R=16 the output is ``[16,16,2]``.  It creates a
    higher-order autograd graph used by ``train_operator`` and changes no parameter.
    """
    if not times.requires_grad:
        raise ValueError("physics times must require gradients")
    current = model(histories, parameters, times)
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
    delayed_times = times - delay_torch(times, parameters, equation)
    delayed_history = interpolate_history_torch(
        histories, delayed_times, equation.maximum_history
    )
    delayed_operator = model(histories, parameters, delayed_times.clamp_min(0.0))
    delayed_state = torch.where(
        (delayed_times <= 0.0).unsqueeze(-1), delayed_history, delayed_operator
    )
    return time_derivative - rhs_torch(current, delayed_state, parameters, equation)


def normalized_model_parameter_sensitivities(
    model: Operator4D,
    histories: torch.Tensor,
    parameters: torch.Tensor,
    times: torch.Tensor,
) -> torch.Tensor:
    """Differentiate predictions with respect to both physical parameters.

    Forward-mode JVPs produce ``[B,Q,4,2]`` without materializing a dense Jacobian.
    The beta and tau0 derivatives are multiplied by their physical interval widths,
    matching the stored solver targets ``4*d y/d beta`` and ``0.6*d y/d tau0``.
    The returned tensor remains differentiable with respect to all model weights, so
    its mean-square error can be used directly as a training loss.
    """
    if parameters.ndim != 2 or parameters.shape[1] != 2:
        raise ValueError("parameters must have shape [B,2]")

    def evaluate(parameter_values: torch.Tensor) -> torch.Tensor:
        """Evaluate the operator for one differentiable parameter argument."""
        return model(histories, parameter_values, times)

    beta_tangent = torch.zeros_like(parameters)
    beta_tangent[:, 0] = 1.0
    tau_tangent = torch.zeros_like(parameters)
    tau_tangent[:, 1] = 1.0
    _, beta_derivative = torch.func.jvp(
        evaluate, (parameters,), (beta_tangent,), strict=True
    )
    _, tau_derivative = torch.func.jvp(
        evaluate, (parameters,), (tau_tangent,), strict=True
    )
    beta_normalized = model.physical_parameter_span[0] * beta_derivative
    tau_normalized = model.physical_parameter_span[1] * tau_derivative
    return torch.stack((beta_normalized, tau_normalized), dim=-1)


@torch.no_grad()
def evaluate_operator(
    model: Operator4D,
    split: DatasetSplit,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, float], np.ndarray, list[dict[str, float]]]:
    """Evaluate one fixed split without influencing selection beyond validation.

    ``model`` is frozen in eval mode, ``split`` contains N true trajectories,
    ``device`` and ``batch_size`` control memory.  Returns aggregate MSE/relative-L2,
    predictions ``[N,Q,4]`` and per-case metric rows. For formal test N=2048.
    It temporarily sets eval mode and transfers batches; no gradients are created.
    Validation calls it during training and test calls it once after best selection.
    """
    model.eval()
    predictions: list[np.ndarray] = []
    times = torch.as_tensor(split.output_times, dtype=torch.float32, device=device)
    for start in range(0, split.histories.shape[0], batch_size):
        histories = torch.as_tensor(
            split.histories[start : start + batch_size], dtype=torch.float32, device=device
        )
        parameters = torch.as_tensor(
            split.parameters[start : start + batch_size], dtype=torch.float32, device=device
        )
        predictions.append(model(histories, parameters, times).cpu().numpy())
    prediction = np.concatenate(predictions, axis=0)
    truth = np.asarray(split.solutions, dtype=np.float64)
    difference = prediction.astype(np.float64) - truth
    case_mse = np.mean(difference**2, axis=(1, 2))
    case_relative = np.linalg.norm(difference.reshape(difference.shape[0], -1), axis=1) / np.maximum(
        np.linalg.norm(truth.reshape(truth.shape[0], -1), axis=1), 1e-12
    )
    state_case_mse = np.mean(difference**2, axis=1)
    state_case_relative = np.linalg.norm(difference, axis=1) / np.maximum(
        np.linalg.norm(truth, axis=1), 1e-12
    )
    rows = [
        {
            "case_index": int(index),
            "mse": float(case_mse[index]),
            "relative_l2": float(case_relative[index]),
            **{
                f"state_{component + 1}_mse": float(state_case_mse[index, component])
                for component in range(STATE_DIM)
            },
            **{
                f"state_{component + 1}_relative_l2": float(
                    state_case_relative[index, component]
                )
                for component in range(STATE_DIM)
            },
        }
        for index in range(case_mse.size)
    ]
    metrics = {
        "mse": float(np.mean(case_mse)),
        "relative_l2_mean": float(np.mean(case_relative)),
        "relative_l2_median": float(np.median(case_relative)),
    }
    for component in range(STATE_DIM):
        metrics[f"state_{component + 1}_mse"] = float(
            np.mean(state_case_mse[:, component])
        )
        metrics[f"state_{component + 1}_relative_l2_mean"] = float(
            np.mean(state_case_relative[:, component])
        )
    return (
        metrics,
        prediction.astype(np.float32),
        rows,
    )


def save_json(path: Path, values: Mapping[str, Any]) -> None:
    """Atomically save a JSON object with deterministic indentation.

    ``path`` is the final file and ``values`` a JSON-serializable mapping.  Returns
    ``None``; it creates parents and replaces through a same-directory temporary.
    For example metrics become ``metrics.json``.  Training and pipeline reporting
    call this helper so interrupted writes cannot masquerade as complete results.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(dict(values), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def save_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    """Atomically save homogeneous metric rows as UTF-8 CSV.

    ``path`` is final and ``rows`` must be nonempty with a common key set.  Returns
    ``None``; e.g. 2048 test rows become ``forward_per_case_metrics.csv``.  It
    creates parents and atomically replaces the file.  ``train_operator`` calls it
    for history and per-case metrics.
    """
    if not rows:
        raise ValueError("cannot save an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def checkpoint_payload(
    model: Operator4D,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    best_validation: float,
    best_iteration: int,
    model_type: str,
    training_seed: int,
    resolved_config: Mapping[str, Any],
    history: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a complete, method-safe restart checkpoint dictionary.

    Inputs capture current model/optimizer state, iteration, validation selection,
    method, seed, final config and logged history.  The returned mapping includes
    network format and reconstruction config; for example ``iteration=500`` is
    sufficient to resume at 501.  Tensor state is copied by ``torch.save`` later.
    ``train_operator`` calls it at every validation point for last and improved best.
    """
    return {
        "network_format": OPERATOR_NETWORK_FORMAT,
        "model_type": model_type,
        "display_name": MODEL_DISPLAY_NAMES[model_type],
        "model_config": model.model_config(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None,
        "iteration": int(iteration),
        "best_validation": float(best_validation),
        "best_iteration": int(best_iteration),
        "training_seed": int(training_seed),
        "resolved_config": dict(resolved_config),
        "history": list(history),
    }


def save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically persist one PyTorch checkpoint.

    ``path`` names best/last model and ``payload`` comes from ``checkpoint_payload``.
    Returns ``None``; it creates parents and replaces through a temporary file.  For
    example a successful validation writes ``last_model.pt`` and possibly
    ``best_model.pt``.  ``train_operator`` is the caller.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move every tensor in a restored optimizer state to the active device.

    ``optimizer`` has just loaded a checkpoint and ``device`` is its model device.
    Returns ``None`` and mutates optimizer state tensors in place.  For example CPU
    Adam moments become CUDA tensors.  ``train_operator`` calls it only on resume.
    """
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def train_operator(
    model_type: str,
    splits: Mapping[str, DatasetSplit],
    resolved_config: Mapping[str, Any],
    output_dir: Path,
    device: torch.device,
    training_seed: int,
    smoke: bool,
) -> dict[str, Any]:
    """Train, validate, select, reload and test one requested operator structure.

    All four methods receive identical ``splits``, sampling seed, losses, AdamW and
    LR schedule; only branch factorization differs.  ``output_dir`` is method/seed
    specific and ``device`` is one logical GPU (or CPU).  The returned result contains
    method, parameter count, best validation, final test metrics and runtime.  It
    writes best/last checkpoints, history, per-case metrics, predictions and a task
    result.  Resume uses a compatible ``last_model.pt``.  ``pipeline.operator_job``
    calls this function once per method.
    """
    set_seed(training_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    operator_config = dict(resolved_config["operator"])
    data_config = resolved_config["data"]
    equation = NicholsonConfig.from_mapping(resolved_config["equation"])
    model = build_operator(
        model_type,
        int(data_config["history_sensors"]),
        float(data_config["horizon"]),
        operator_config,
        device,
        training_seed,
    )
    parameter_count = count_trainable_parameters(model)
    if not smoke:
        expected = EXPECTED_FORMAL_PARAMETER_COUNTS[model_type]
        if parameter_count != expected:
            raise RuntimeError(
                f"parameter-count drift for {model_type}: {parameter_count} != {expected}"
            )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(operator_config["learning_rate"]),
        weight_decay=float(operator_config["weight_decay"]),
    )
    LOGGER.info(
        "Starting %s (%s) | seed=%d | device=%s | trainable parameters=%d",
        model_type, MODEL_DISPLAY_NAMES[model_type], training_seed, device, parameter_count,
    )
    start_iteration = 1
    best_validation = math.inf
    best_iteration = 0
    history: list[dict[str, Any]] = []
    last_path = output_dir / "last_model.pt"
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        saved_science = checkpoint.get("resolved_config", {})
        science_keys = ("data", "equation", "operator")
        if (
            checkpoint.get("network_format") != OPERATOR_NETWORK_FORMAT
            or checkpoint.get("model_type") != model_type
            or int(checkpoint.get("training_seed", -1)) != training_seed
            or checkpoint.get("model_config") != model.model_config()
            or any(saved_science.get(key) != resolved_config.get(key) for key in science_keys)
        ):
            raise RuntimeError(f"incompatible resume checkpoint: {last_path}")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        _optimizer_to_device(optimizer, device)
        start_iteration = int(checkpoint["iteration"]) + 1
        best_validation = float(checkpoint["best_validation"])
        best_iteration = int(checkpoint["best_iteration"])
        history = [dict(row) for row in checkpoint.get("history", [])]
        LOGGER.info("Resuming %s from iteration %d", model_type, start_iteration)

    tracker = WandBTracker(
        resolved_config["wandb"], output_dir / "wandb", resolved_config,
        f"{model_type}_seed_{training_seed}_{'smoke' if smoke else 'full'}", LOGGER,
    )
    rng = np.random.default_rng(training_seed + 1000)
    train = splits["train"]
    if train.normalized_parameter_sensitivities is None:
        raise RuntimeError("training split is missing normalized parameter sensitivities")
    expected_sensitivity_shape = (*train.solutions.shape, 2)
    if train.normalized_parameter_sensitivities.shape != expected_sensitivity_shape:
        raise RuntimeError(
            "training sensitivity shape mismatch: "
            f"{train.normalized_parameter_sensitivities.shape} != {expected_sensitivity_shape}"
        )
    train_size = train.histories.shape[0]
    full_times = torch.as_tensor(train.output_times, dtype=torch.float32, device=device)
    total_iterations = int(operator_config["iterations"])
    supervised_batch = int(operator_config["supervised_pair_batch"])
    pool_batch = int(operator_config["physics_pool_batch"])
    residual_points = int(operator_config["residual_points"])
    sensitivity_points = min(
        int(operator_config["sensitivity_points"]), int(train.output_times.size)
    )
    if sensitivity_points < 1:
        raise ValueError("sensitivity_points must be positive")
    validation_every = int(operator_config["validation_every"])
    steps_per_epoch = int(operator_config["steps_per_epoch"])
    evaluation_batch = int(operator_config["evaluation_batch_size"])
    running = {
        "total": 0.0,
        "data": 0.0,
        "initial": 0.0,
        "beta_sensitivity": 0.0,
        "tau_sensitivity": 0.0,
        "physics": 0.0,
        "count": 0,
    }
    started = time.perf_counter()
    success = False
    try:
        for iteration in range(start_iteration, total_iterations + 1):
            model.train()
            learning_rate = learning_rate_at(iteration, operator_config)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            indices = rng.integers(0, train_size, size=supervised_batch)
            histories_batch = torch.as_tensor(
                train.histories[indices], dtype=torch.float32, device=device
            )
            parameters_batch = torch.as_tensor(
                train.parameters[indices], dtype=torch.float32, device=device
            )
            targets_batch = torch.as_tensor(
                train.solutions[indices], dtype=torch.float32, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            prediction = model(histories_batch, parameters_batch, full_times)
            data_loss = torch.mean((prediction - targets_batch) ** 2)
            initial_prediction = model(
                histories_batch, parameters_batch,
                torch.zeros(1, dtype=torch.float32, device=device),
            )[:, 0, :]
            initial_loss = torch.mean((initial_prediction - histories_batch[:, :, -1]) ** 2)
            sensitivity_indices = np.sort(
                rng.choice(train.output_times.size, size=sensitivity_points, replace=False)
            )
            sensitivity_times = full_times[
                torch.as_tensor(sensitivity_indices, dtype=torch.long, device=device)
            ]
            sensitivity_targets = torch.as_tensor(
                train.normalized_parameter_sensitivities[indices][:, sensitivity_indices],
                dtype=torch.float32,
                device=device,
            )
            sensitivity_prediction = normalized_model_parameter_sensitivities(
                model, histories_batch, parameters_batch, sensitivity_times
            )
            beta_sensitivity_loss = torch.mean(
                (sensitivity_prediction[..., 0] - sensitivity_targets[..., 0]) ** 2
            )
            tau_sensitivity_loss = torch.mean(
                (sensitivity_prediction[..., 1] - sensitivity_targets[..., 1]) ** 2
            )
            active_physics_weight = physics_weight_at(iteration, operator_config)
            if active_physics_weight > 0.0:
                pairing_mode = str(
                    operator_config.get("physics_pairing_mode", "online_cartesian")
                )
                if pairing_mode == "online_cartesian":
                    history_rows = rng.integers(0, train_size, size=pool_batch)
                    parameter_rows = rng.integers(0, train_size, size=pool_batch)
                    physics_histories_numpy = np.repeat(
                        train.histories[history_rows], pool_batch, axis=0
                    )
                    physics_parameters_numpy = np.tile(
                        train.parameters[parameter_rows], (pool_batch, 1)
                    )
                elif pairing_mode == "training_edges":
                    # In the combination experiment the physics loss must preserve
                    # observed history--parameter edges; otherwise an online Cartesian
                    # product would expose held-out combinations during training.
                    edge_rows = rng.integers(
                        0, train_size, size=pool_batch * pool_batch
                    )
                    physics_histories_numpy = train.histories[edge_rows]
                    physics_parameters_numpy = train.parameters[edge_rows]
                else:
                    raise ValueError(
                        "physics_pairing_mode must be online_cartesian or training_edges"
                    )
                physics_histories = torch.as_tensor(
                    physics_histories_numpy, dtype=torch.float32, device=device
                )
                physics_parameters = torch.as_tensor(
                    physics_parameters_numpy, dtype=torch.float32, device=device
                )
                physics_times = (
                    torch.rand(
                        pool_batch * pool_batch, residual_points,
                        dtype=torch.float32, device=device,
                    )
                    * float(data_config["horizon"])
                ).requires_grad_(True)
                residual = operator_physics_residual(
                    model, physics_histories, physics_parameters, physics_times, equation
                )
                physics_loss = torch.mean(residual**2)
            else:
                physics_loss = torch.zeros((), dtype=torch.float32, device=device)
            total_loss = (
                float(operator_config["data_weight"]) * data_loss
                + float(operator_config["initial_weight"]) * initial_loss
                + float(operator_config["beta_sensitivity_weight"])
                * beta_sensitivity_loss
                + float(operator_config["tau_sensitivity_weight"])
                * tau_sensitivity_loss
                + active_physics_weight * physics_loss
            )
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"non-finite loss at iteration {iteration}")
            total_loss.backward()
            epoch_due = iteration % steps_per_epoch == 0 or iteration == total_iterations
            epoch = int(math.ceil(iteration / steps_per_epoch))
            if epoch_due and bool(resolved_config["wandb"].get("log_gradients", True)):
                tracker.log_gradients(model, epoch)
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(operator_config["gradient_clip"])
            ))
            optimizer.step()
            running["total"] += float(total_loss.detach().cpu())
            running["data"] += float(data_loss.detach().cpu())
            running["initial"] += float(initial_loss.detach().cpu())
            running["beta_sensitivity"] += float(beta_sensitivity_loss.detach().cpu())
            running["tau_sensitivity"] += float(tau_sensitivity_loss.detach().cpu())
            running["physics"] += float(physics_loss.detach().cpu())
            running["count"] += 1

            validation_due = iteration % validation_every == 0 or iteration == total_iterations
            if validation_due:
                validation_metrics, _, _ = evaluate_operator(
                    model, splits["validation"], device, evaluation_batch
                )
                denominator = max(int(running["count"]), 1)
                row = {
                    "iteration": int(iteration),
                    "epoch": int(epoch),
                    "train_total_loss": running["total"] / denominator,
                    "train_data_loss": running["data"] / denominator,
                    "train_initial_loss": running["initial"] / denominator,
                    "train_beta_sensitivity_loss": running["beta_sensitivity"] / denominator,
                    "train_tau_sensitivity_loss": running["tau_sensitivity"] / denominator,
                    "train_physics_loss": running["physics"] / denominator,
                    "validation_mse": validation_metrics["mse"],
                    "learning_rate": float(learning_rate),
                    "physics_weight": float(active_physics_weight),
                    "gradient_norm_before_clip": gradient_norm,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                history.append(row)
                improved = validation_metrics["mse"] < best_validation
                if improved:
                    best_validation = validation_metrics["mse"]
                    best_iteration = iteration
                payload = checkpoint_payload(
                    model, optimizer, iteration, best_validation, best_iteration,
                    model_type, training_seed, resolved_config, history,
                )
                save_checkpoint(last_path, payload)
                if improved:
                    save_checkpoint(output_dir / "best_model.pt", payload)
                    LOGGER.info(
                        "New best checkpoint | iteration=%d | validation MSE=%.6e | path=%s",
                        iteration, best_validation, output_dir / "best_model.pt",
                    )
                save_csv(output_dir / "training_history.csv", history)
                tracker.log_epoch(
                    {
                        "train/total_loss": row["train_total_loss"],
                        "train/data_loss": row["train_data_loss"],
                        "train/initial_loss": row["train_initial_loss"],
                        "train/beta_sensitivity_loss": row[
                            "train_beta_sensitivity_loss"
                        ],
                        "train/tau_sensitivity_loss": row[
                            "train_tau_sensitivity_loss"
                        ],
                        "train/physics_loss": row["train_physics_loss"],
                        "validation/mse": row["validation_mse"],
                        "optimizer/learning_rate": row["learning_rate"],
                    },
                    epoch,
                )
                LOGGER.info(
                    "%s | iteration %d/%d | train %.6e | validation %.6e | best %.6e | lr %.3e",
                    model_type, iteration, total_iterations, row["train_total_loss"],
                    row["validation_mse"], best_validation, learning_rate,
                )
                running = {
                    "total": 0.0,
                    "data": 0.0,
                    "initial": 0.0,
                    "beta_sensitivity": 0.0,
                    "tau_sensitivity": 0.0,
                    "physics": 0.0,
                    "count": 0,
                }

        best_model, _ = load_operator_checkpoint(output_dir / "best_model.pt", device)
        test_metrics, test_prediction, per_case = evaluate_operator(
            best_model, splits["test"], device, evaluation_batch
        )
        for row in per_case:
            row.update({
                "model_type": model_type,
                "method": MODEL_DISPLAY_NAMES[model_type],
                "training_seed": int(training_seed),
            })
        np.save(output_dir / "test_predictions.npy", test_prediction)
        save_csv(output_dir / "forward_per_case_metrics.csv", per_case)
        elapsed = time.perf_counter() - started
        result = {
            "model_type": model_type,
            "method": MODEL_DISPLAY_NAMES[model_type],
            "training_seed": int(training_seed),
            "parameter_count": int(parameter_count),
            "best_iteration": int(best_iteration),
            "best_validation_mse": float(best_validation),
            "test": test_metrics,
            "training_runtime_seconds": float(elapsed),
            "best_checkpoint": str(output_dir / "best_model.pt"),
        }
        save_json(output_dir / "metrics.json", result)
        save_json(output_dir / "task_result.json", result)
        LOGGER.info(
            "Finished %s | best iteration=%d | test MSE=%.6e | relative L2=%.6e | runtime=%.2f s",
            model_type, best_iteration, test_metrics["mse"],
            test_metrics["relative_l2_mean"], elapsed,
        )
        success = True
        return result
    finally:
        tracker.finish(success)
