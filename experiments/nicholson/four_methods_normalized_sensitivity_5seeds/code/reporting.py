"""Forward-only metrics aggregation and English scientific figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401  # registers the required SciencePlots styles

from data import DatasetSplit
from model import MODEL_DISPLAY_NAMES
from training import save_csv, save_json


plt.style.use(["science", "no-latex", "grid"])
matplotlib.rcParams['font.family'] = 'Source Han Serif'
matplotlib.rcParams["axes.unicode_minus"] = False


def save_figure(figure: plt.Figure, base_path: Path, config: Mapping[str, Any]) -> list[Path]:
    """Save one true-result figure in every configured non-TIFF format.

    ``figure`` is a completed Matplotlib figure, ``base_path`` omits a suffix and
    ``config`` supplies formats/DPI.  Returns the created paths; e.g. PNG and SVG.
    It creates the figure directory and writes local files.  Plot functions call it
    and then close the figure themselves.
    """
    formats = [str(item).lower() for item in config["figure_formats"]]
    if any(item not in {"png", "jpg", "jpeg", "svg"} for item in formats):
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
    """Plot truth, prediction and absolute errors for one shared test case.

    ``split`` is the fixed test set, ``predictions`` has ``[N,Q,4]``, ``model_type``
    identifies the title and ``output_dir`` receives figures.  The configured case
    index is clipped to N. Returns PNG/SVG paths with four patch panels and one
    error panel, all from real saved test data. Files are its only side effect.
    ``pipeline.operator_job`` calls it after final test evaluation.
    """
    case_index = min(
        int(reporting_config["representative_forward_case"]), split.solutions.shape[0] - 1
    )
    times = split.output_times
    truth = split.solutions[case_index]
    prediction = predictions[case_index]
    figure, axes_grid = plt.subplots(2, 3, figsize=(12.0, 6.5))
    axes = axes_grid.ravel()
    state_labels = ("Patch 1", "Patch 2", "Patch 3", "Patch 4")
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    for component in range(4):
        axes[component].plot(
            times, truth[:, component], color="black", linewidth=1.5,
            label="Ground truth",
        )
        axes[component].plot(
            times, prediction[:, component], color=colors[component], linestyle="--",
            linewidth=1.4, label="Prediction",
        )
        axes[component].set_xlabel("Time")
        axes[component].set_ylabel("Population density")
        axes[component].set_title(state_labels[component])
        axes[component].legend()
    for component in range(4):
        axes[4].plot(
            times, np.abs(prediction[:, component] - truth[:, component]),
            color=colors[component], label=f"Patch {component + 1}",
        )
    axes[4].set_xlabel("Time")
    axes[4].set_ylabel("Absolute error")
    axes[4].set_title("Pointwise errors")
    axes[4].legend(ncol=2)
    beta, tau0 = split.parameters[case_index]
    axes[5].axis("off")
    axes[5].text(
        0.05, 0.70,
        f"Test case {case_index}\n"
        f"Recruitment beta = {beta:.4f}\n"
        f"Mean delay tau0 = {tau0:.4f}\n"
        f"Delay amplitude = 0.2",
        fontsize=11, va="top",
    )
    figure.suptitle(f"{MODEL_DISPLAY_NAMES[model_type]}: representative forward prediction")
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
    """Plot train total loss and validation MSE from the real optimization history.

    ``history_path`` is the saved CSV, method supplies the English title and output
    receives files.  Returns created paths; e.g. PNG/SVG with logarithmic y-axis.
    It reads CSV and writes figures only.  ``pipeline.operator_job`` calls it after
    a successful train/test task.
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
    """Aggregate every method/seed run and report method-wise mean and std."""
    if not training_seeds or len(set(training_seeds)) != len(training_seeds):
        raise ValueError("training_seeds must be nonempty and unique")
    run_rows: list[dict[str, Any]] = []
    for model_type in methods:
        for training_seed in training_seeds:
            path = (
                stage_dir / "methods" / model_type /
                f"seed_{training_seed}" / "metrics.json"
            )
            if not path.exists():
                raise FileNotFoundError(path)
            values = json.loads(path.read_text(encoding="utf-8"))
            if (
                values.get("model_type") != model_type
                or int(values.get("training_seed")) != training_seed
            ):
                raise RuntimeError(f"method/seed mismatch in {path}")
            test = values["test"]
            numeric = [
                values["best_validation_mse"], test["mse"],
                test["relative_l2_mean"], test["relative_l2_median"],
            ]
            if not np.isfinite(numeric).all():
                raise RuntimeError(f"non-finite aggregate metric in {path}")
            row = {
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
            for component in range(4):
                row[f"state_{component + 1}_test_mse"] = float(
                    test[f"state_{component + 1}_mse"]
                )
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
        *(f"state_{component}_test_mse" for component in range(1, 5)),
        *(f"state_{component}_relative_l2_mean" for component in range(1, 5)),
    )
    summary_rows: list[dict[str, Any]] = []
    for model_type in methods:
        selected = [row for row in run_rows if row["model_type"] == model_type]
        counts = {int(row["parameter_count"]) for row in selected}
        if len(counts) != 1:
            raise RuntimeError(f"parameter count changed across seeds for {model_type}")
        summary_row: dict[str, Any] = {
            "model_type": model_type,
            "method": MODEL_DISPLAY_NAMES[model_type],
            "seed_count": len(selected),
            "parameter_count": counts.pop(),
        }
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
            summary_row[f"{metric}_mean"] = float(np.mean(values))
            summary_row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(summary_row)
    save_csv(comparison_dir / "method_summary.csv", summary_rows)

    labels = [row["method"] for row in summary_rows]
    positions = np.arange(len(summary_rows))
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axes[0].bar(
        positions,
        [row["test_mse_mean"] for row in summary_rows],
        yerr=[row["test_mse_std"] for row in summary_rows],
        capsize=4,
        color="#0072B2",
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Test MSE")
    axes[0].set_title("Mean test error across seeds")
    axes[1].bar(
        positions,
        [row["test_relative_l2_mean_mean"] for row in summary_rows],
        yerr=[row["test_relative_l2_mean_std"] for row in summary_rows],
        capsize=4,
        color="#D55E00",
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Mean relative L2 error")
    axes[1].set_title("Mean relative error across seeds")
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=25, ha="right")
    figure.tight_layout()
    figure_paths = save_figure(
        figure, comparison_dir / "forward_method_comparison", reporting_config
    )
    plt.close(figure)
    best = min(summary_rows, key=lambda row: row["test_mse_mean"])
    summary = {
        "method_count": len(summary_rows),
        "seed_count": len(training_seeds),
        "training_seeds": [int(seed) for seed in training_seeds],
        "forward_only": True,
        "parameter_normalization": True,
        "parameter_sensitivity_supervision": True,
        "methods": summary_rows,
        "best_test_mse_method": best["model_type"],
        "best_test_mse_mean": best["test_mse_mean"],
        "figure_paths": [str(path) for path in figure_paths],
    }
    save_json(comparison_dir / "summary.json", summary)
    return summary
