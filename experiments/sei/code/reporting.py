"""Aggregate delayed-SEI forward metrics and draw English result figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

from data import DatasetSplit
from equation import STATE_DIM, STATE_NAMES
from model import MODEL_DISPLAY_NAMES
from training import save_csv, save_json


plt.style.use(["science", "no-latex", "grid"])
matplotlib.rcParams['font.family'] = 'Source Han Serif'
matplotlib.rcParams["axes.unicode_minus"] = False


def save_figure(figure: plt.Figure, base_path: Path, config: Mapping[str, Any]) -> list[Path]:
    """Save one real-result figure in configured PNG/JPG/SVG formats.

    ``figure`` is complete, ``base_path`` has no suffix and ``config`` provides
    formats and raster DPI. The returned list contains all written paths; e.g.
    ``[forward_prediction.png, forward_prediction.svg]``. The function creates the
    parent directory and is called by every plotting function in this module.
    """
    formats = [str(item).lower() for item in config["figure_formats"]]
    if not formats or any(item not in {"png", "jpg", "jpeg", "svg"} for item in formats):
        raise ValueError("figure formats are limited to png/jpg/svg")
    base_path.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for extension in formats:
        path = base_path.with_suffix(f".{extension}")
        figure.savefig(path, dpi=int(config["raster_dpi"]), bbox_inches="tight")
        paths.append(path)
    return paths


def plot_method_forward(
    split: DatasetSplit,
    predictions: np.ndarray,
    model_type: str,
    output_dir: Path,
    reporting_config: Mapping[str, Any],
) -> list[Path]:
    """Plot three true/predicted concentrations and their absolute errors.

    ``split`` is the shared test set and ``predictions`` is ``[N,Q,3]``. The model
    key selects the title and ``output_dir`` receives PNG/SVG files. The configured
    representative index is clipped to the available N. Returns the saved paths.
    ``pipeline.operator_job`` calls it after final testing; files are the side effect.
    """
    if predictions.shape != split.solutions.shape or predictions.shape[-1] != STATE_DIM:
        raise ValueError("predictions must match test solutions [N,Q,3]")
    case_index = min(
        int(reporting_config["representative_forward_case"]), split.solutions.shape[0] - 1
    )
    times = split.output_times
    truth = split.solutions[case_index]
    prediction = predictions[case_index]
    colors = ("#0072B2", "#D55E00", "#009E73")
    figure, axes_grid = plt.subplots(2, 2, figsize=(9.0, 6.5))
    axes = axes_grid.ravel()
    for component in range(STATE_DIM):
        axes[component].plot(times, truth[:, component], color="black", label="Ground truth")
        axes[component].plot(
            times, prediction[:, component], color=colors[component], linestyle="--",
            label="Prediction",
        )
        axes[component].set_xlabel("Time")
        axes[component].set_ylabel("Concentration")
        axes[component].set_title(STATE_NAMES[component])
        axes[component].legend()
    for component in range(STATE_DIM):
        axes[3].plot(
            times, np.abs(prediction[:, component] - truth[:, component]),
            color=colors[component], label=STATE_NAMES[component],
        )
    axes[3].set_xlabel("Time")
    axes[3].set_ylabel("Absolute error")
    axes[3].set_title(
        "Pointwise errors; "
        f"b = {float(split.parameters[case_index, 0]):.4f}, "
        f"a = {float(split.parameters[case_index, 1]):.4f}"
    )
    axes[3].legend()
    figure.suptitle(f"{MODEL_DISPLAY_NAMES[model_type]}: representative prediction")
    figure.tight_layout()
    paths = save_figure(figure, output_dir / "figures" / "forward_prediction", reporting_config)
    plt.close(figure)
    return paths


def plot_training_history(
    history_path: Path,
    model_type: str,
    output_dir: Path,
    reporting_config: Mapping[str, Any],
) -> list[Path]:
    """Plot saved train total loss and validation MSE on one logarithmic axis.

    ``history_path`` is the actual CSV, ``model_type`` names the curve and
    ``output_dir`` receives files. Returns the written PNG/SVG paths. Reading the CSV
    and writing figures are its only side effects; ``pipeline.operator_job`` calls it.
    """
    with history_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    iterations = np.asarray([int(row["iteration"]) for row in rows])
    train_loss = np.asarray([float(row["train_total_loss"]) for row in rows])
    validation = np.asarray([float(row["validation_mse"]) for row in rows])
    figure, axis = plt.subplots(figsize=(6.0, 4.0))
    axis.semilogy(iterations, train_loss, label="Train total loss")
    axis.semilogy(iterations, validation, label="Validation MSE")
    axis.set_xlabel("Training iteration")
    axis.set_ylabel("Loss")
    axis.set_title(f"{MODEL_DISPLAY_NAMES[model_type]} training history")
    axis.legend()
    figure.tight_layout()
    paths = save_figure(figure, output_dir / "figures" / "training_history", reporting_config)
    plt.close(figure)
    return paths


def aggregate_results(
    stage_dir: Path,
    methods: list[str],
    training_seeds: list[int],
    reporting_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate complete method-seed metrics into CSV, JSON and a comparison plot.

    ``stage_dir`` contains method outputs, ``methods`` contains four model keys,
    ``training_seeds`` contains one or five unique seeds and ``reporting_config``
    controls image output. Returns the summary dictionary. It verifies every grid
    cell and writes per-seed rows, mean-plus-sample-SD rows and the comparison figure.
    ``pipeline.run_stage`` calls it only after all jobs finish.
    """
    if not training_seeds or len(set(training_seeds)) != len(training_seeds):
        raise ValueError("training_seeds must be nonempty and unique")
    run_rows: list[dict[str, Any]] = []
    for model_type in methods:
        for training_seed in training_seeds:
            path = stage_dir / "methods" / model_type / f"seed_{training_seed}" / "metrics.json"
            if not path.exists():
                raise FileNotFoundError(path)
            values = json.loads(path.read_text(encoding="utf-8"))
            if values.get("model_type") != model_type or int(values.get("training_seed")) != training_seed:
                raise RuntimeError(f"method/seed mismatch in {path}")
            test = values["test"]
            numeric = [values["best_validation_mse"], test["mse"], test["relative_l2_mean"]]
            if not np.isfinite(numeric).all():
                raise RuntimeError(f"non-finite aggregate metric in {path}")
            row: dict[str, Any] = {
                "model_type": model_type,
                "method": MODEL_DISPLAY_NAMES[model_type],
                "training_seed": int(training_seed),
                "parameter_count": int(values["parameter_count"]),
                "best_iteration": int(values["best_iteration"]),
                "best_validation_mse": float(values["best_validation_mse"]),
                "test_mse": float(test["mse"]),
                "test_relative_l2_mean": float(test["relative_l2_mean"]),
                "test_relative_l2_median": float(test["relative_l2_median"]),
                "training_runtime_seconds": float(values["training_runtime_seconds"]),
            }
            for component in range(STATE_DIM):
                row[f"state_{component + 1}_test_mse"] = float(test[f"state_{component + 1}_mse"])
                row[f"state_{component + 1}_relative_l2_mean"] = float(
                    test[f"state_{component + 1}_relative_l2_mean"]
                )
            run_rows.append(row)
    comparison_dir = stage_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    save_csv(comparison_dir / "all_seed_results.csv", run_rows)
    metric_names = (
        "best_validation_mse", "test_mse", "test_relative_l2_mean",
        "test_relative_l2_median", "training_runtime_seconds",
        *(f"state_{component}_test_mse" for component in range(1, STATE_DIM + 1)),
        *(f"state_{component}_relative_l2_mean" for component in range(1, STATE_DIM + 1)),
    )
    summary_rows: list[dict[str, Any]] = []
    for model_type in methods:
        selected = [row for row in run_rows if row["model_type"] == model_type]
        counts = {int(row["parameter_count"]) for row in selected}
        if len(counts) != 1:
            raise RuntimeError(f"parameter count changed across seeds for {model_type}")
        summary_row: dict[str, Any] = {
            "model_type": model_type, "method": MODEL_DISPLAY_NAMES[model_type],
            "seed_count": len(selected), "parameter_count": counts.pop(),
        }
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in selected])
            summary_row[f"{metric}_mean"] = float(np.mean(values))
            summary_row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(summary_row)
    save_csv(comparison_dir / "method_summary.csv", summary_rows)
    positions = np.arange(len(summary_rows))
    labels = [row["method"] for row in summary_rows]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for axis, metric, error, title, ylabel, color in (
        (axes[0], "test_mse_mean", "test_mse_std", "Mean test error across seeds", "Test MSE", "#0072B2"),
        (axes[1], "test_relative_l2_mean_mean", "test_relative_l2_mean_std", "Mean relative error across seeds", "Mean relative L2 error", "#D55E00"),
    ):
        axis.bar(positions, [row[metric] for row in summary_rows],
                 yerr=[row[error] for row in summary_rows], capsize=4, color=color)
        axis.set_yscale("log")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.set_xticks(positions, labels, rotation=25, ha="right")
    figure.tight_layout()
    figure_paths = save_figure(figure, comparison_dir / "forward_method_comparison", reporting_config)
    plt.close(figure)
    best = min(summary_rows, key=lambda row: row["test_mse_mean"])
    summary = {
        "method_count": len(summary_rows), "seed_count": len(training_seeds),
        "training_seeds": [int(seed) for seed in training_seeds], "forward_only": True,
        "parameter_normalization": True, "parameter_sensitivity_supervision": True,
        "learned_parameters": ["transmission_b", "convexity_a"],
        "methods": summary_rows,
        "best_test_mse_method": best["model_type"],
        "best_test_mse_mean": best["test_mse_mean"],
        "figure_paths": [str(path) for path in figure_paths],
    }
    save_json(comparison_dir / "summary.json", summary)
    return summary
