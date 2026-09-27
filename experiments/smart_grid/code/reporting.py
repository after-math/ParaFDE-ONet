"""Figures and five-seed summaries for the ParaFDEONet forward experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
import matplotlib
import numpy as np

from data import DatasetSplit
from equation import NODE_COUNT, STATE_DIM, STATE_NAMES
from model import MODEL_DISPLAY_NAME, MODEL_TYPE
from training import save_csv, save_json


plt.style.use(["science", "no-latex", "grid"])
matplotlib.rcParams['font.family'] = 'Source Han Serif'
matplotlib.rcParams['font.family'] = 'Source Han Serif'
matplotlib.rcParams["axes.unicode_minus"] = False


def save_figure(
    figure: plt.Figure,
    base_path: Path,
    reporting_config: Mapping[str, Any],
) -> list[Path]:
    formats = [str(item).lower() for item in reporting_config["figure_formats"]]
    if not formats or any(item not in {"png", "jpg", "jpeg", "svg"} for item in formats):
        raise ValueError("figure formats are limited to PNG, JPG and SVG")
    base_path.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in formats:
        path = base_path.with_suffix(f".{extension}")
        figure.savefig(path, dpi=int(reporting_config["raster_dpi"]), bbox_inches="tight")
        paths.append(path)
    return paths


def plot_forward_prediction(
    split: DatasetSplit,
    predictions: np.ndarray,
    output_dir: Path,
    reporting_config: Mapping[str, Any],
) -> list[Path]:
    """Draw truth and prediction for all eight physical states of one test case."""
    if predictions.shape != split.solutions.shape or predictions.shape[-1] != STATE_DIM:
        raise ValueError("predictions must match the test solution array [N,Q,8]")
    case_index = min(
        int(reporting_config["representative_forward_case"]),
        split.solutions.shape[0] - 1,
    )
    truth = split.solutions[case_index]
    prediction = predictions[case_index]
    times = split.output_times
    figure, axes = plt.subplots(4, 2, figsize=(10.0, 11.2), sharex=True)
    axes_flat = axes.ravel()
    colors = ("#0072B2",) * NODE_COUNT + ("#D55E00",) * NODE_COUNT
    for component, axis in enumerate(axes_flat):
        axis.plot(times, truth[:, component], color="black", linewidth=1.35, label="Reference")
        axis.plot(
            times,
            prediction[:, component],
            color=colors[component],
            linestyle="--",
            linewidth=1.15,
            label="ParaFDEONet",
        )
        axis.set_title(STATE_NAMES[component].replace("_", " "))
        axis.set_ylabel("Angle (rad)" if component < NODE_COUNT else "Frequency (rad/s)")
        axis.legend(loc="best", frameon=True)
    for axis in axes[-1]:
        axis.set_xlabel("Time (s)")
    parameters = split.parameters[case_index]
    figure.suptitle(
        "Representative forward prediction: "
        f"K={parameters[0]:.4f}, gamma={parameters[1]:.4f}, alpha={parameters[2]:.4f}",
        y=0.995,
    )
    figure.tight_layout()
    paths = save_figure(
        figure, output_dir / "figures" / "forward_prediction_8_states", reporting_config
    )
    plt.close(figure)
    return paths


def plot_training_history(
    history_path: Path,
    output_dir: Path,
    reporting_config: Mapping[str, Any],
) -> list[Path]:
    with history_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty training history: {history_path}")
    iterations = np.asarray([int(row["iteration"]) for row in rows])
    train_total = np.asarray([float(row["train_total_loss"]) for row in rows])
    validation = np.asarray([float(row["validation_normalized_mse"]) for row in rows])
    physics = np.asarray([float(row["train_physics_loss"]) for row in rows])
    sensitivity = np.mean(
        np.asarray(
            [
                [
                    float(row["train_K_sensitivity_loss"]),
                    float(row["train_gamma_sensitivity_loss"]),
                    float(row["train_alpha_sensitivity_loss"]),
                ]
                for row in rows
            ]
        ),
        axis=1,
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    axes[0].semilogy(iterations, train_total, label="Train total")
    axes[0].semilogy(iterations, validation, label="Validation normalized MSE")
    axes[0].set_xlabel("Training iteration")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Optimization and validation")
    axes[0].legend()
    axes[1].semilogy(iterations, np.maximum(physics, 1e-16), label="Physics residual")
    axes[1].semilogy(iterations, np.maximum(sensitivity, 1e-16), label="Mean sensitivity loss")
    axes[1].set_xlabel("Training iteration")
    axes[1].set_ylabel("Unweighted loss")
    axes[1].set_title("Auxiliary losses")
    axes[1].legend()
    figure.suptitle(f"{MODEL_DISPLAY_NAME} training history")
    figure.tight_layout()
    paths = save_figure(figure, output_dir / "figures" / "training_history", reporting_config)
    plt.close(figure)
    return paths


def aggregate_results(
    stage_dir: Path,
    training_seeds: list[int],
    reporting_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate the one model across the requested independent training seeds."""
    if not training_seeds or len(set(training_seeds)) != len(training_seeds):
        raise ValueError("training seeds must be nonempty and unique")
    rows: list[dict[str, Any]] = []
    for seed in training_seeds:
        path = stage_dir / "methods" / MODEL_TYPE / f"seed_{seed}" / "metrics.json"
        if not path.exists():
            raise FileNotFoundError(path)
        values = json.loads(path.read_text(encoding="utf-8"))
        if values.get("model_type") != MODEL_TYPE or int(values.get("training_seed", -1)) != seed:
            raise RuntimeError(f"model or seed mismatch in {path}")
        test = values["test"]
        row: dict[str, Any] = {
            "model_type": MODEL_TYPE,
            "method": MODEL_DISPLAY_NAME,
            "training_seed": int(seed),
            "parameter_count": int(values["parameter_count"]),
            "best_iteration": int(values["best_iteration"]),
            "best_validation_normalized_mse": float(values["best_validation_normalized_mse"]),
            "test_mse": float(test["mse"]),
            "test_normalized_mse": float(test["normalized_mse"]),
            "test_relative_l2_mean": float(test["relative_l2_mean"]),
            "test_relative_l2_median": float(test["relative_l2_median"]),
            "test_relative_angle_difference_l2_mean": float(
                test["relative_angle_difference_l2_mean"]
            ),
            "test_frequency_relative_l2_mean": float(test["frequency_relative_l2_mean"]),
            "training_runtime_seconds": float(values["training_runtime_seconds"]),
        }
        sensitivity = values.get("test_parameter_sensitivity")
        if sensitivity is not None:
            row["test_parameter_sensitivity_normalized_mse"] = float(
                sensitivity["normalized_mse"]
            )
            for name in ("coupling_K", "response_gamma", "damping_alpha"):
                row[f"test_{name}_sensitivity_normalized_mse"] = float(
                    sensitivity[f"{name}_normalized_mse"]
                )
                row[f"test_{name}_sensitivity_relative_l2"] = float(
                    sensitivity[f"{name}_relative_l2"]
                )
        for component in range(STATE_DIM):
            row[f"state_{component + 1}_test_mse"] = float(test[f"state_{component + 1}_mse"])
            row[f"state_{component + 1}_test_rmse"] = float(test[f"state_{component + 1}_rmse"])
            row[f"state_{component + 1}_relative_l2_mean"] = float(
                test[f"state_{component + 1}_relative_l2_mean"]
            )
        if not np.isfinite([value for value in row.values() if isinstance(value, float)]).all():
            raise RuntimeError(f"non-finite aggregate metric in {path}")
        rows.append(row)

    result_dir = stage_dir / "summary"
    result_dir.mkdir(parents=True, exist_ok=True)
    save_csv(result_dir / "all_seed_results.csv", rows)
    numeric_names = [
        name for name, value in rows[0].items()
        if isinstance(value, float)
    ]
    parameter_counts = {int(row["parameter_count"]) for row in rows}
    if len(parameter_counts) != 1:
        raise RuntimeError("ParaFDEONet parameter count changed between seeds")
    summary_row: dict[str, Any] = {
        "model_type": MODEL_TYPE,
        "method": MODEL_DISPLAY_NAME,
        "seed_count": len(rows),
        "parameter_count": parameter_counts.pop(),
    }
    for name in numeric_names:
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        summary_row[f"{name}_mean"] = float(np.mean(values))
        summary_row[f"{name}_sample_std"] = (
            float(np.std(values, ddof=1)) if values.size > 1 else None
        )
    save_csv(result_dir / "parafdeonet_summary.csv", [summary_row])

    displayed_metrics = (
        ("test_normalized_mse", "Normalized MSE", "#0072B2"),
        ("test_relative_angle_difference_l2_mean", "Angle-difference relative L2", "#009E73"),
        ("test_frequency_relative_l2_mean", "Frequency relative L2", "#D55E00"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    x = np.arange(len(rows))
    for axis, (metric, title, color) in zip(axes, displayed_metrics):
        values = np.asarray([float(row[metric]) for row in rows])
        axis.scatter(x, values, s=34, color=color, label="Independent seeds", zorder=3)
        axis.axhline(np.mean(values), color="black", linestyle="--", label="Mean")
        axis.set_yscale("log")
        axis.set_xticks(x, [str(seed) for seed in training_seeds], rotation=30, ha="right")
        axis.set_xlabel("Training seed")
        axis.set_ylabel(title)
        axis.set_title(title)
    axes[0].legend(loc="best")
    figure.suptitle(f"{MODEL_DISPLAY_NAME}: variability across training seeds")
    figure.tight_layout()
    figure_paths = save_figure(
        figure, result_dir / "forward_seed_variability", reporting_config
    )
    plt.close(figure)
    summary = {
        "model_type": MODEL_TYPE,
        "method": MODEL_DISPLAY_NAME,
        "forward_only": True,
        "training_seeds": [int(seed) for seed in training_seeds],
        "metrics": summary_row,
        "figure_paths": [str(path) for path in figure_paths],
    }
    save_json(result_dir / "summary.json", summary)
    return summary
