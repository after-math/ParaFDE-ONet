"""Forward-only metrics aggregation and English scientific figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib
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

    ``split`` is the fixed test set, ``predictions`` has ``[N,Q,2]``, ``model_type``
    identifies the title and ``output_dir`` receives figures.  The configured case
    index is clipped to N.  Returns PNG/SVG paths; for example two state panels and
    one error panel all use real saved test data.  Files are its only side effect.
    ``pipeline.operator_job`` calls it after final test evaluation.
    """
    case_index = min(
        int(reporting_config["representative_forward_case"]), split.solutions.shape[0] - 1
    )
    times = split.output_times
    truth = split.solutions[case_index]
    prediction = predictions[case_index]
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.4))
    state_labels = ("Species 1", "Species 2")
    colors = ("#0072B2", "#D55E00")
    for component in range(2):
        axes[component].plot(times, truth[:, component], color="black", linewidth=1.5, label="Ground truth")
        axes[component].plot(
            times, prediction[:, component], color=colors[component], linestyle="--",
            linewidth=1.4, label="Prediction",
        )
        axes[component].set_xlabel("Time")
        axes[component].set_ylabel("Population density")
        axes[component].set_title(state_labels[component])
        axes[component].legend()
    axes[2].plot(times, np.abs(prediction[:, 0] - truth[:, 0]), label="Species 1 error")
    axes[2].plot(times, np.abs(prediction[:, 1] - truth[:, 1]), label="Species 2 error")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Absolute error")
    axes[2].set_title("Pointwise error")
    axes[2].legend()
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
    """Validate and aggregate the complete method-by-seed forward comparison.

    ``stage_dir`` contains method/seed outputs, ``methods`` fixes paper order and
    ``training_seeds`` contains the independent model/training seeds, for example
    five integers in the formal experiment.  Returns seed-level rows plus mean and
    sample standard deviation across seeds.  It writes JSON, two CSV files and a
    two-panel comparison figure with error bars.  Missing, duplicate or non-finite
    outputs raise instead of silently producing a table.  ``pipeline.run_stage``
    calls it after every requested job succeeds.
    """
    if not training_seeds or len(training_seeds) != len(set(training_seeds)):
        raise ValueError("training_seeds must be nonempty and unique")
    seed_rows: list[dict[str, Any]] = []
    for model_type in methods:
        for training_seed in training_seeds:
            path = stage_dir / "methods" / model_type / f"seed_{training_seed}" / "metrics.json"
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
                values["training_runtime_seconds"],
            ]
            if not np.isfinite(numeric).all():
                raise RuntimeError(f"non-finite aggregate metric in {path}")
            seed_rows.append({
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
            })
    comparison_dir = stage_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    save_csv(comparison_dir / "method_seed_results.csv", seed_rows)

    metric_names = (
        "best_iteration", "best_validation_mse", "test_mse",
        "test_relative_l2_mean", "test_relative_l2_median",
        "training_runtime_seconds",
    )
    method_rows: list[dict[str, Any]] = []
    for model_type in methods:
        selected = [row for row in seed_rows if row["model_type"] == model_type]
        if len(selected) != len(training_seeds):
            raise RuntimeError(f"incomplete seed grid for {model_type}")
        parameter_counts = {int(row["parameter_count"]) for row in selected}
        if len(parameter_counts) != 1:
            raise RuntimeError(f"parameter count changed across seeds for {model_type}")
        aggregate: dict[str, Any] = {
            "model_type": model_type,
            "method": MODEL_DISPLAY_NAMES[model_type],
            "seed_count": len(selected),
            "parameter_count": parameter_counts.pop(),
        }
        for metric_name in metric_names:
            values = np.asarray([float(row[metric_name]) for row in selected], dtype=np.float64)
            aggregate[f"{metric_name}_mean"] = float(values.mean())
            aggregate[f"{metric_name}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        method_rows.append(aggregate)
    save_csv(comparison_dir / "method_summary.csv", method_rows)

    labels = [row["method"] for row in method_rows]
    positions = np.arange(len(method_rows))
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axes[0].bar(
        positions,
        [row["test_mse_mean"] for row in method_rows],
        yerr=[row["test_mse_std"] for row in method_rows],
        capsize=3,
        color="#0072B2",
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Test MSE (mean $\\pm$ SD)")
    axes[0].set_title("Forward prediction error")
    axes[1].bar(
        positions,
        [row["test_relative_l2_mean_mean"] for row in method_rows],
        yerr=[row["test_relative_l2_mean_std"] for row in method_rows],
        capsize=3,
        color="#D55E00",
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Relative L2 error (mean $\\pm$ SD)")
    axes[1].set_title("Relative trajectory error")
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=25, ha="right")
    figure.tight_layout()
    figure_paths = save_figure(
        figure, comparison_dir / "forward_method_comparison", reporting_config
    )
    plt.close(figure)
    best = min(method_rows, key=lambda row: row["test_mse_mean"])
    summary = {
        "method_count": len(method_rows),
        "seed_count": len(training_seeds),
        "run_count": len(seed_rows),
        "training_seeds": [int(seed) for seed in training_seeds],
        "forward_only": True,
        "parameter_normalization": True,
        "parameter_sensitivity_supervision": False,
        "methods": method_rows,
        "per_seed_runs": seed_rows,
        "best_test_mse_mean_method": best["model_type"],
        "best_test_mse_mean": best["test_mse_mean"],
        "figure_paths": [str(path) for path in figure_paths],
    }
    save_json(comparison_dir / "summary.json", summary)
    return summary
