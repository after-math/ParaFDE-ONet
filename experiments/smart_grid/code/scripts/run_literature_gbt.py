#!/usr/bin/env python3
"""Evaluate an adapted published GBT probability baseline without label leakage.

The baseline follows the non-graph GradientBoostingRegressor comparator in
Nauck et al. (IET GTD, 2026), adapted to the present fixed-topology delayed-grid
survivability target. Every reported GBT prediction is out of fold with respect
to the 64 physical parameter combinations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
CODE_DIR = PROJECT_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from scripts.run_survivability import spearman_correlation


def save_json(path: Path, values: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(values), indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def save_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to save empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def normalize_parameters(parameters: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    values = np.asarray(parameters, dtype=np.float64)
    limits = np.asarray(bounds, dtype=np.float64)
    if values.ndim != 2 or limits.shape != (values.shape[1], 2):
        raise ValueError("parameter/bound dimensions are inconsistent")
    lower = limits[:, 0]
    upper = limits[:, 1]
    if np.any(upper <= lower):
        raise ValueError("each parameter upper bound must exceed its lower bound")
    tolerance = 1e-7 * np.maximum(1.0, np.abs(upper - lower))
    if np.any(values < lower - tolerance) or np.any(values > upper + tolerance):
        raise ValueError("a parameter falls outside the predeclared bounds")
    return 2.0 * (values - lower) / (upper - lower) - 1.0


def out_of_fold_gbt_predictions(
    features: np.ndarray,
    targets: np.ndarray,
    fold_count: int,
    split_seed: int,
    estimator_random_state: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Return one strictly out-of-fold prediction for each parameter point."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import KFold

    x_values = np.asarray(features, dtype=np.float64)
    y_values = np.asarray(targets, dtype=np.float64).reshape(-1)
    if x_values.ndim != 2 or x_values.shape[0] != y_values.size:
        raise ValueError("features and targets must share their first dimension")
    if not 2 <= fold_count <= y_values.size:
        raise ValueError("fold_count must be between two and the number of samples")
    prediction = np.full(y_values.shape, np.nan, dtype=np.float64)
    fold_assignment = np.full(y_values.shape, -1, dtype=np.int64)
    split_rows: list[dict[str, Any]] = []
    estimator_parameters: dict[str, Any] | None = None
    splitter = KFold(n_splits=fold_count, shuffle=True, random_state=split_seed)
    for fold_index, (train_indices, test_indices) in enumerate(splitter.split(x_values)):
        overlap = np.intersect1d(train_indices, test_indices)
        if overlap.size:
            raise RuntimeError("cross-validation train/test leakage detected")
        estimator = GradientBoostingRegressor(random_state=estimator_random_state)
        estimator.fit(x_values[train_indices], y_values[train_indices])
        prediction[test_indices] = estimator.predict(x_values[test_indices])
        fold_assignment[test_indices] = fold_index
        if estimator_parameters is None:
            estimator_parameters = estimator.get_params(deep=False)
        split_rows.append(
            {
                "split_seed": int(split_seed),
                "fold": int(fold_index),
                "train_count": int(train_indices.size),
                "test_count": int(test_indices.size),
                "train_indices": ";".join(str(int(value)) for value in train_indices),
                "test_indices": ";".join(str(int(value)) for value in test_indices),
                "overlap_count": int(overlap.size),
            }
        )
    if np.any(~np.isfinite(prediction)) or np.any(fold_assignment < 0):
        raise RuntimeError("not every parameter point received an out-of-fold prediction")
    unique, counts = np.unique(
        np.concatenate(
            [
                np.fromstring(row["test_indices"], sep=";", dtype=np.int64)
                for row in split_rows
            ]
        ),
        return_counts=True,
    )
    if unique.size != y_values.size or not np.array_equal(unique, np.arange(y_values.size)):
        raise RuntimeError("cross-validation did not cover every row exactly once")
    if np.any(counts != 1):
        raise RuntimeError("a row was evaluated in more than one fold")
    assert estimator_parameters is not None
    return prediction, fold_assignment, split_rows, estimator_parameters


def nested_tuned_gbt_predictions(
    features: np.ndarray,
    targets: np.ndarray,
    outer_fold_count: int,
    outer_split_seed: int,
    estimator_random_state: int,
    tuning_config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Tune GBT only inside each outer training fold, then predict its held-out fold."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import GridSearchCV, KFold

    x_values = np.asarray(features, dtype=np.float64)
    y_values = np.asarray(targets, dtype=np.float64).reshape(-1)
    if x_values.ndim != 2 or x_values.shape[0] != y_values.size:
        raise ValueError("features and targets must share their first dimension")
    inner_fold_count = int(tuning_config["inner_fold_count"])
    if inner_fold_count < 2:
        raise ValueError("nested tuning requires at least two inner folds")
    parameter_grid = dict(tuning_config["parameter_grid"])
    if not parameter_grid or any(not values for values in parameter_grid.values()):
        raise ValueError("nested tuning parameter_grid cannot be empty")
    outer = KFold(
        n_splits=outer_fold_count, shuffle=True, random_state=outer_split_seed
    )
    prediction = np.full(y_values.shape, np.nan, dtype=np.float64)
    fold_assignment = np.full(y_values.shape, -1, dtype=np.int64)
    tuning_rows: list[dict[str, Any]] = []
    for outer_fold, (outer_train, outer_test) in enumerate(outer.split(x_values)):
        if np.intersect1d(outer_train, outer_test).size:
            raise RuntimeError("nested-CV outer train/test leakage detected")
        inner_seed = int(tuning_config["inner_split_seed_base"]) + outer_fold
        inner = KFold(
            n_splits=inner_fold_count, shuffle=True, random_state=inner_seed
        )
        search = GridSearchCV(
            estimator=GradientBoostingRegressor(
                random_state=estimator_random_state
            ),
            param_grid=parameter_grid,
            scoring=str(tuning_config["scoring"]),
            cv=inner,
            n_jobs=int(tuning_config["parallel_jobs"]),
            refit=True,
            return_train_score=False,
        )
        search.fit(x_values[outer_train], y_values[outer_train])
        prediction[outer_test] = search.predict(x_values[outer_test])
        fold_assignment[outer_test] = outer_fold
        row: dict[str, Any] = {
            "outer_fold": outer_fold,
            "outer_train_count": int(outer_train.size),
            "outer_test_count": int(outer_test.size),
            "outer_train_test_overlap_count": 0,
            "inner_fold_count": inner_fold_count,
            "inner_split_seed": inner_seed,
            "best_inner_cv_mae": float(-search.best_score_),
        }
        row.update({f"best_{key}": value for key, value in search.best_params_.items()})
        tuning_rows.append(row)
    if np.any(~np.isfinite(prediction)) or np.any(fold_assignment < 0):
        raise RuntimeError("nested CV did not predict every parameter point")
    return prediction, fold_assignment, tuning_rows


def top_risk_recall(
    reference_probability: np.ndarray,
    predicted_probability: np.ndarray,
    fraction: float,
) -> tuple[float, int]:
    """Recall of the lowest-survivability parameter points with stable tie breaking."""
    reference = np.asarray(reference_probability, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted_probability, dtype=np.float64).reshape(-1)
    if reference.shape != predicted.shape or not 0.0 < fraction <= 1.0:
        raise ValueError("invalid probability arrays or risk fraction")
    count = max(1, int(math.ceil(fraction * reference.size)))
    indices = np.arange(reference.size)
    reference_top = set(np.lexsort((indices, reference))[:count].tolist())
    predicted_top = set(np.lexsort((indices, predicted))[:count].tolist())
    return len(reference_top & predicted_top) / count, count


def regression_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    top_risk_fractions: Sequence[float],
) -> dict[str, Any]:
    from sklearn.metrics import r2_score

    truth = np.asarray(reference, dtype=np.float64).reshape(-1)
    estimate = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if truth.shape != estimate.shape or not np.all(np.isfinite(estimate)):
        raise ValueError("reference and prediction arrays are invalid")
    error = estimate - truth
    absolute_error = np.abs(error)
    metrics: dict[str, Any] = {
        "mae": float(np.mean(absolute_error)),
        "mae_percentage_points": float(100.0 * np.mean(absolute_error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "rmse_percentage_points": float(100.0 * np.sqrt(np.mean(error**2))),
        "maximum_absolute_error": float(np.max(absolute_error)),
        "maximum_absolute_error_percentage_points": float(100.0 * np.max(absolute_error)),
        "mean_signed_error": float(np.mean(error)),
        "mean_signed_error_percentage_points": float(100.0 * np.mean(error)),
        "r2": float(r2_score(truth, estimate)),
        "spearman": float(spearman_correlation(truth, estimate)),
        "prediction_minimum": float(np.min(estimate)),
        "prediction_maximum": float(np.max(estimate)),
        "fraction_predictions_outside_unit_interval": float(
            np.mean((estimate < 0.0) | (estimate > 1.0))
        ),
    }
    for fraction in top_risk_fractions:
        recall, count = top_risk_recall(truth, estimate, float(fraction))
        label = f"lowest_survivability_{int(round(100.0 * fraction))}_percent"
        metrics[f"{label}_count"] = int(count)
        metrics[f"{label}_recall"] = float(recall)
    return metrics


def paired_bootstrap_mae_advantage(
    reference: np.ndarray,
    para_prediction: np.ndarray,
    baseline_prediction: np.ndarray,
    repeats: int,
    seed: int,
    confidence_level: float,
) -> dict[str, float | int]:
    """Bootstrap the GBT MAE minus ParaFDEONet MAE over parameter points."""
    truth = np.asarray(reference, dtype=np.float64).reshape(-1)
    para_error = np.abs(np.asarray(para_prediction, dtype=np.float64).reshape(-1) - truth)
    baseline_error = np.abs(
        np.asarray(baseline_prediction, dtype=np.float64).reshape(-1) - truth
    )
    if para_error.shape != baseline_error.shape or repeats < 1:
        raise ValueError("invalid paired bootstrap inputs")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between zero and one")
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(
        0, truth.size, size=(repeats, truth.size), endpoint=False
    )
    differences = np.mean(
        (baseline_error - para_error)[sample_indices], axis=1, dtype=np.float64
    )
    alpha = 0.5 * (1.0 - confidence_level)
    lower, upper = np.quantile(differences, [alpha, 1.0 - alpha])
    return {
        "repeats": int(repeats),
        "seed": int(seed),
        "confidence_level": float(confidence_level),
        "mae_advantage": float(np.mean(baseline_error - para_error)),
        "mae_advantage_percentage_points": float(
            100.0 * np.mean(baseline_error - para_error)
        ),
        "confidence_interval_lower": float(lower),
        "confidence_interval_upper": float(upper),
        "confidence_interval_lower_percentage_points": float(100.0 * lower),
        "confidence_interval_upper_percentage_points": float(100.0 * upper),
        "bootstrap_fraction_para_mae_lower": float(np.mean(differences > 0.0)),
    }


def create_figure(
    output_dir: Path,
    config: Mapping[str, Any],
    direct: np.ndarray,
    para: np.ndarray,
    tuned_gbt: np.ndarray,
    metrics: Mapping[str, Any],
) -> list[Path]:
    import matplotlib
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401

    plt.style.use(["science", "no-latex", "grid"])
    matplotlib.rcParams['font.family'] = 'Source Han Serif'
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
    matplotlib.rcParams.update(
        {
            "font.size": 6.4,
            "axes.titlesize": 7.2,
            "axes.labelsize": 6.8,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "legend.fontsize": 5.8,
        }
    )
    width = float(config["reporting"]["figure_width_inches"])
    height = float(config["reporting"]["figure_height_inches"])
    figure, axes = plt.subplots(1, 3, figsize=(width, height), constrained_layout=True)
    direct_percent = 100.0 * np.asarray(direct)
    para_percent = 100.0 * np.asarray(para)
    gbt_percent = 100.0 * np.asarray(tuned_gbt)

    axis = axes[0]
    axis.scatter(
        direct_percent,
        gbt_percent,
        s=14,
        facecolor="none",
        edgecolor="#D55E00",
        linewidth=0.65,
        alpha=0.8,
        label="GBT (nested CV)",
    )
    axis.scatter(
        direct_percent,
        para_percent,
        s=11,
        color="#0072B2",
        alpha=0.75,
        label="ParaFDEONet",
    )
    limit_min = float(min(direct_percent.min(), para_percent.min(), gbt_percent.min()) - 2.0)
    limit_max = float(max(direct_percent.max(), para_percent.max(), gbt_percent.max()) + 2.0)
    axis.plot([limit_min, limit_max], [limit_min, limit_max], "--", color="black", lw=0.75)
    axis.set_xlim(limit_min, limit_max)
    axis.set_ylim(limit_min, limit_max)
    axis.set_xlabel("Direct DDE survivability (%)")
    axis.set_ylabel("Predicted survivability (%)")
    axis.set_title("Probability agreement")
    axis.legend(loc="lower right", frameon=True)

    order = np.argsort(direct, kind="mergesort")
    axis = axes[1]
    axis.plot(
        np.arange(direct.size),
        direct_percent[order],
        color="black",
        lw=1.0,
        label="Direct DDE",
    )
    axis.plot(
        np.arange(direct.size),
        gbt_percent[order],
        color="#D55E00",
        lw=0.85,
        marker="o",
        ms=1.8,
        markevery=3,
        label="GBT (nested CV)",
    )
    axis.plot(
        np.arange(direct.size),
        para_percent[order],
        color="#0072B2",
        lw=0.9,
        label="ParaFDEONet",
    )
    axis.set_xlabel("Parameter points sorted by direct survivability")
    axis.set_ylabel("Survivability (%)")
    axis.set_title("Risk ordering")
    axis.legend(loc="lower right", frameon=True)

    axis = axes[2]
    for errors, label, color, line_style in (
        (np.abs(para_percent - direct_percent), "ParaFDEONet", "#0072B2", "-"),
        (np.abs(gbt_percent - direct_percent), "GBT (nested CV)", "#D55E00", "--"),
    ):
        sorted_errors = np.sort(errors)
        cumulative = np.arange(1, errors.size + 1, dtype=np.float64) / errors.size
        axis.step(
            sorted_errors,
            cumulative,
            where="post",
            color=color,
            linestyle=line_style,
            lw=1.0,
            label=label,
        )
    axis.set_xlabel("Absolute error (percentage points)")
    axis.set_ylabel("Cumulative fraction")
    axis.set_title("Error distribution")
    axis.set_ylim(0.0, 1.02)
    axis.legend(loc="lower right", frameon=True)
    axis.text(
        0.97,
        0.57,
        f"MAE: {metrics['parafdeonet']['mae_percentage_points']:.3f} vs "
        f"{metrics['nested_tuned_gradient_boosted_trees']['mae_percentage_points']:.3f} pp\n"
        f"{metrics['comparison']['tuned_gbt_to_parafdeonet_mae_ratio']:.1f}x lower",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
    )

    for index, axis in enumerate(axes):
        axis.text(
            0.02,
            0.98,
            chr(ord("a") + index),
            transform=axis.transAxes,
            fontsize=8.8,
            fontweight="bold",
            va="top",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "none", "pad": 1.0},
        )
        axis.set_box_aspect(1.0)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    base = figures_dir / str(config["reporting"]["figure_basename"])
    paths = [base.with_suffix(extension) for extension in (".png", ".svg", ".pdf", ".tiff")]
    figure.savefig(paths[0], dpi=int(config["reporting"]["png_dpi"]), bbox_inches="tight")
    figure.savefig(paths[1], bbox_inches="tight")
    figure.savefig(paths[2], bbox_inches="tight")
    figure.savefig(paths[3], dpi=600, bbox_inches="tight")
    plt.close(figure)
    return paths


def write_results(output_dir: Path, config: Mapping[str, Any], metrics: Mapping[str, Any]) -> None:
    para = metrics["parafdeonet"]
    gbt = metrics["gradient_boosted_trees"]
    tuned_gbt = metrics["nested_tuned_gradient_boosted_trees"]
    comparison = metrics["comparison"]
    bootstrap = comparison["paired_parameter_bootstrap"]
    sensitivity = metrics["split_sensitivity"]
    top5_key = "lowest_survivability_5_percent_recall"
    top10_key = "lowest_survivability_10_percent_recall"
    report = f"""# Literature GBT baseline

## Design

The target is direct-DDE Monte Carlo survivability at the fixed representative limits of 0.15 Hz and 25 degrees. The 64 inputs are the physical parameter combinations $(K, \\gamma, \\alpha)$. The Gradient Boosted Trees baseline follows the non-graph probability-regression comparator used by Nauck et al. (2026), using scikit-learn's default `GradientBoostingRegressor` with an explicit random state.

Every GBT value is an out-of-fold prediction. The primary result uses shuffled 8-fold cross-validation: each fold trains on 56 parameter-probability labels and predicts the other 8, so no reported point is predicted by a model fitted to that point's direct-DDE label. Each of the 56 labels is estimated from 512 direct trajectories. Thus 28,672 DDE evaluations contribute to a fold, but the effective GBT regression table contains 56 independent rows. ParaFDEONet remains the frozen 75,223,048-parameter, single-seed operator, was trained on 24,576 history--parameter trajectory pairs, and receives no survivability labels.

## Primary results

| Method | MAE (percentage points) | RMSE (percentage points) | Maximum error (percentage points) | $R^2$ | Spearman | Top-risk 5% recall | Top-risk 10% recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| ParaFDEONet | {para['mae_percentage_points']:.4f} | {para['rmse_percentage_points']:.4f} | {para['maximum_absolute_error_percentage_points']:.4f} | {para['r2']:.6f} | {para['spearman']:.6f} | {100.0 * para[top5_key]:.2f}% | {100.0 * para[top10_key]:.2f}% |
| Literature GBT (published default, out of fold) | {gbt['mae_percentage_points']:.4f} | {gbt['rmse_percentage_points']:.4f} | {gbt['maximum_absolute_error_percentage_points']:.4f} | {gbt['r2']:.6f} | {gbt['spearman']:.6f} | {100.0 * gbt[top5_key]:.2f}% | {100.0 * gbt[top10_key]:.2f}% |
| Literature GBT (nested-CV tuned) | {tuned_gbt['mae_percentage_points']:.4f} | {tuned_gbt['rmse_percentage_points']:.4f} | {tuned_gbt['maximum_absolute_error_percentage_points']:.4f} | {tuned_gbt['r2']:.6f} | {tuned_gbt['spearman']:.6f} | {100.0 * tuned_gbt[top5_key]:.2f}% | {100.0 * tuned_gbt[top10_key]:.2f}% |

Against the nested-CV tuned GBT, ParaFDEONet has {comparison['tuned_gbt_to_parafdeonet_mae_ratio']:.2f} times lower MAE and {comparison['tuned_gbt_to_parafdeonet_rmse_ratio']:.2f} times lower RMSE. Resampling the 64 fixed parameter points gives an empirical 95% range of {bootstrap['confidence_interval_lower_percentage_points']:.4f} to {bootstrap['confidence_interval_upper_percentage_points']:.4f} percentage points for the MAE advantage. Because the parameter combinations form a fixed factorial design, this range is descriptive and is not a population confidence interval over parameter space.

Across the five fixed split seeds, GBT MAE is {sensitivity['mae_percentage_points_mean']:.4f} plus or minus {sensitivity['mae_percentage_points_sample_std']:.4f} percentage points, with range {sensitivity['mae_percentage_points_minimum']:.4f} to {sensitivity['mae_percentage_points_maximum']:.4f}. The fixed ParaFDEONet MAE is {para['mae_percentage_points']:.4f} percentage points.

## Interpretation

This is an adapted literature comparator rather than a reproduction of the Nauck et al. dataset. It is favorable to GBT in label proximity because GBT learns final scalar probabilities at 56 neighbouring points on the application grid, but it is data-limited because those probabilities provide only 56 independent regression rows per fold. The information supplied to the methods also differs: GBT observes only $(K, \\gamma, \\alpha)$ and estimates a probability marginalized over histories, whereas ParaFDEONet observes each history and predicts its complete eight-state trajectory before aggregation. The comparison therefore supports more accurate history-conditioned screening for this fixed system, parameter grid, and history distribution; it does not establish universal architectural superiority over tree or graph models.

The GBT runtime is not compared with the trajectory deployment timing because it outputs only 64 scalar probabilities and its required direct-DDE label-generation cost is separate. The existing direct-DDE comparison remains the valid reference for the 32,768-trajectory speedup.

## References

- Nauck, C. et al. Predicting the Fault-Ride-Through Probability of Inverter-Dominated Power Grids Using Machine Learning. *IET Generation, Transmission & Distribution* **20**, e70264 (2026). https://doi.org/{config['literature_baseline']['doi']}
- Public paper companion and non-graph baseline implementation: {config['literature_baseline']['public_companion']}
- Hellmann, F. et al. Survivability of deterministic dynamical systems. *Scientific Reports* **6**, 29654 (2016). https://doi.org/10.1038/srep29654
"""
    (output_dir / "RESULTS_GBT.md").write_text(report, encoding="utf-8")


def run_experiment(
    survivability_dir: Path,
    config_path: Path,
    output_dir: Path | None = None,
) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if output_dir is None:
        output_dir = survivability_dir / str(config["reporting"]["output_subdirectory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        output_dir / "status.json",
        {"state": "running", "started_unix_seconds": time.time()},
    )
    input_path = survivability_dir / str(config["input"]["survivability_data_file"])
    parent_config_path = survivability_dir / "config_resolved.json"
    parent_metrics_path = survivability_dir / "metrics.json"
    if not input_path.is_file() or not parent_config_path.is_file() or not parent_metrics_path.is_file():
        raise FileNotFoundError("the completed survivability experiment inputs are missing")
    parent_config = json.loads(parent_config_path.read_text(encoding="utf-8"))
    parent_metrics = json.loads(parent_metrics_path.read_text(encoding="utf-8"))
    target_frequency = float(
        parent_config["survivability"]["representative_frequency_threshold_hz"]
    )
    target_angle = float(
        parent_config["survivability"]["representative_edge_angle_threshold_degrees"]
    )
    if not math.isclose(target_frequency, 0.15) or not math.isclose(target_angle, 25.0):
        raise ValueError("the input representative screening limits are not 0.15 Hz and 25 degrees")
    with np.load(input_path) as values:
        parameters = np.asarray(values["parameters"], dtype=np.float64)
        direct = np.asarray(
            values["direct_representative_survivability"], dtype=np.float64
        )
        para = np.asarray(
            values["parafdeonet_representative_survivability"], dtype=np.float64
        )
    expected_count = int(config["input"]["expected_parameter_count"])
    if parameters.shape != (expected_count, 3) or direct.shape != (expected_count,):
        raise ValueError("the input does not contain the expected 64 by 3 parameter design")
    if para.shape != direct.shape or np.any((direct < 0.0) | (direct > 1.0)):
        raise ValueError("the survivability probability arrays are invalid")
    bounds = np.asarray(config["input"]["parameter_bounds"], dtype=np.float64)
    # Validate against the declared design bounds, then retain raw values because
    # Nauck et al. explicitly use no feature scaling for GBT. Affine scaling would
    # leave tree splits equivalent, but raw inputs keep the adaptation closest to
    # the published setup.
    normalize_parameters(parameters, bounds)
    features = parameters.copy()
    validation = config["validation"]
    top_fractions = [float(value) for value in validation["top_risk_fractions"]]
    primary_seed = int(validation["primary_split_seed"])
    estimator_seed = int(validation["estimator_random_state"])
    fold_count = int(validation["fold_count"])
    sensitivity_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    split_audit_rows: list[dict[str, Any]] = []
    primary_prediction: np.ndarray | None = None
    primary_folds: np.ndarray | None = None
    estimator_parameters: dict[str, Any] | None = None
    start = time.perf_counter()
    for split_seed_value in validation["split_sensitivity_seeds"]:
        split_seed = int(split_seed_value)
        prediction, folds, audit_rows, current_parameters = out_of_fold_gbt_predictions(
            features,
            direct,
            fold_count=fold_count,
            split_seed=split_seed,
            estimator_random_state=estimator_seed,
        )
        current_metrics = regression_metrics(direct, prediction, top_fractions)
        sensitivity_rows.append(
            {
                "split_seed": split_seed,
                "mae_percentage_points": current_metrics["mae_percentage_points"],
                "rmse_percentage_points": current_metrics["rmse_percentage_points"],
                "maximum_absolute_error_percentage_points": current_metrics[
                    "maximum_absolute_error_percentage_points"
                ],
                "r2": current_metrics["r2"],
                "spearman": current_metrics["spearman"],
                "lowest_survivability_5_percent_recall": current_metrics[
                    "lowest_survivability_5_percent_recall"
                ],
                "lowest_survivability_10_percent_recall": current_metrics[
                    "lowest_survivability_10_percent_recall"
                ],
            }
        )
        for row in audit_rows:
            split_audit_rows.append(row)
        for parameter_index, value in enumerate(prediction):
            prediction_rows.append(
                {
                    "split_seed": split_seed,
                    "parameter_index": parameter_index,
                    "fold": int(folds[parameter_index]),
                    "coupling_K": parameters[parameter_index, 0],
                    "delayed_feedback_gamma": parameters[parameter_index, 1],
                    "damping_alpha": parameters[parameter_index, 2],
                    "direct_survivability": direct[parameter_index],
                    "gbt_out_of_fold_survivability": value,
                    "absolute_error_percentage_points": 100.0
                    * abs(value - direct[parameter_index]),
                }
            )
        if split_seed == primary_seed:
            primary_prediction = prediction.copy()
            primary_folds = folds.copy()
            estimator_parameters = current_parameters
    elapsed = time.perf_counter() - start
    if primary_prediction is None or primary_folds is None or estimator_parameters is None:
        raise ValueError("primary_split_seed must be included in split_sensitivity_seeds")

    tuning_config = validation["nested_tuning"]
    if not bool(tuning_config["enabled"]):
        raise ValueError("the strong-baseline audit requires nested tuning to be enabled")
    tuning_start = time.perf_counter()
    tuned_prediction, tuned_folds, tuning_rows = nested_tuned_gbt_predictions(
        features,
        direct,
        outer_fold_count=fold_count,
        outer_split_seed=primary_seed,
        estimator_random_state=estimator_seed,
        tuning_config=tuning_config,
    )
    tuning_elapsed = time.perf_counter() - tuning_start

    para_metrics = regression_metrics(direct, para, top_fractions)
    gbt_metrics = regression_metrics(direct, primary_prediction, top_fractions)
    tuned_gbt_metrics = regression_metrics(direct, tuned_prediction, top_fractions)
    uncertainty = config["uncertainty"]
    bootstrap = paired_bootstrap_mae_advantage(
        direct,
        para,
        tuned_prediction,
        repeats=int(uncertainty["paired_parameter_bootstrap_repeats"]),
        seed=int(uncertainty["bootstrap_seed"]),
        confidence_level=float(uncertainty["confidence_level"]),
    )
    sensitivity_mae = np.asarray(
        [row["mae_percentage_points"] for row in sensitivity_rows], dtype=np.float64
    )
    metrics: dict[str, Any] = {
        "experiment_name": config["experiment_name"],
        "target": {
            "frequency_threshold_hz": target_frequency,
            "edge_angle_threshold_degrees": target_angle,
            "history_count_per_parameter": int(parent_metrics["design"]["history_count"]),
            "parameter_count": expected_count,
        },
        "frozen_model": parent_metrics["frozen_model"],
        "direct_reference": parent_metrics["baseline"],
        "literature_baseline": config["literature_baseline"],
        "validation": {
            "method": validation["method"],
            "fold_count": fold_count,
            "training_parameter_count_per_fold": int(expected_count - expected_count // fold_count),
            "test_parameter_count_per_fold": int(expected_count // fold_count),
            "histories_per_probability_label": int(
                parent_metrics["design"]["history_count"]
            ),
            "direct_trajectory_outcomes_available_to_gbt_per_fold": int(
                (expected_count - expected_count // fold_count)
                * parent_metrics["design"]["history_count"]
            ),
            "parafdeonet_supervised_training_trajectory_count": int(
                config["input"]["parafdeonet_supervised_training_trajectory_count"]
            ),
            "primary_split_seed": primary_seed,
            "estimator_random_state": estimator_seed,
            "no_train_test_overlap_verified": True,
            "each_parameter_predicted_exactly_once_per_split_verified": True,
            "nested_tuning": tuning_config,
            "nested_tuning_outer_test_labels_never_used_for_selection_verified": True,
        },
        "parafdeonet": para_metrics,
        "gradient_boosted_trees": gbt_metrics,
        "nested_tuned_gradient_boosted_trees": tuned_gbt_metrics,
        "comparison": {
            "default_gbt_to_parafdeonet_mae_ratio": float(
                gbt_metrics["mae"] / para_metrics["mae"]
            ),
            "default_gbt_to_parafdeonet_rmse_ratio": float(
                gbt_metrics["rmse"] / para_metrics["rmse"]
            ),
            "tuned_gbt_to_parafdeonet_mae_ratio": float(
                tuned_gbt_metrics["mae"] / para_metrics["mae"]
            ),
            "tuned_gbt_to_parafdeonet_rmse_ratio": float(
                tuned_gbt_metrics["rmse"] / para_metrics["rmse"]
            ),
            "paired_parameter_bootstrap": bootstrap,
        },
        "split_sensitivity": {
            "split_seeds": [int(value) for value in validation["split_sensitivity_seeds"]],
            "mae_percentage_points_mean": float(np.mean(sensitivity_mae)),
            "mae_percentage_points_sample_std": float(np.std(sensitivity_mae, ddof=1)),
            "mae_percentage_points_minimum": float(np.min(sensitivity_mae)),
            "mae_percentage_points_maximum": float(np.max(sensitivity_mae)),
        },
        "runtime": {
            "five_complete_oof_evaluations_seconds": float(elapsed),
            "nested_tuned_oof_evaluation_seconds": float(tuning_elapsed),
            "runtime_not_comparable_to_trajectory_deployment": True,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "provenance": {
            "input_file": str(input_path.resolve()),
            "input_sha256": file_sha256(input_path),
            "parent_config_sha256": file_sha256(parent_config_path),
            "parent_metrics_sha256": file_sha256(parent_metrics_path),
            "config_sha256": file_sha256(config_path),
        },
        "estimator_parameters": estimator_parameters,
    }
    import sklearn

    metrics["software"]["scikit_learn"] = sklearn.__version__
    save_json(output_dir / "config_resolved.json", config)
    save_json(output_dir / "metrics.json", metrics)
    primary_rows: list[dict[str, Any]] = []
    for index in range(expected_count):
        primary_rows.append(
            {
                "parameter_index": index,
                "fold": int(primary_folds[index]),
                "coupling_K": parameters[index, 0],
                "delayed_feedback_gamma": parameters[index, 1],
                "damping_alpha": parameters[index, 2],
                "direct_survivability": direct[index],
                "parafdeonet_survivability": para[index],
                "gbt_out_of_fold_survivability": primary_prediction[index],
                "tuned_gbt_out_of_fold_survivability": tuned_prediction[index],
                "parafdeonet_absolute_error_percentage_points": 100.0
                * abs(para[index] - direct[index]),
                "gbt_absolute_error_percentage_points": 100.0
                * abs(primary_prediction[index] - direct[index]),
                "tuned_gbt_absolute_error_percentage_points": 100.0
                * abs(tuned_prediction[index] - direct[index]),
                "tuned_gbt_outer_fold": int(tuned_folds[index]),
            }
        )
    save_csv(output_dir / "source_data_primary.csv", primary_rows)
    save_csv(output_dir / "source_data_split_sensitivity.csv", prediction_rows)
    save_csv(output_dir / "split_audit.csv", split_audit_rows)
    save_csv(output_dir / "split_sensitivity_metrics.csv", sensitivity_rows)
    save_csv(output_dir / "nested_tuning_choices.csv", tuning_rows)
    np.savez_compressed(
        output_dir / "predictions.npz",
        parameters=parameters,
        direct_survivability=direct,
        parafdeonet_survivability=para,
        gbt_out_of_fold_survivability=primary_prediction,
        tuned_gbt_out_of_fold_survivability=tuned_prediction,
        primary_fold_assignment=primary_folds,
        tuned_gbt_outer_fold_assignment=tuned_folds,
    )
    figure_paths = create_figure(
        output_dir, config, direct, para, tuned_prediction, metrics
    )
    write_results(output_dir, config, metrics)
    qa = f"""# Figure QA

- Core claim: a frozen trajectory operator estimates parameterwise survivability more accurately than an adapted published scalar GBT baseline on the same direct-DDE reference.
- Target: representative 0.15 Hz and 25 degree survivability probability over 512 held-out histories at each of 64 parameter points.
- GBT validation: shuffled 8-fold out-of-fold prediction; 56 training and 8 test parameter points per fold; no label overlap. Hyperparameters shown in the figure are selected by four-fold CV inside each outer training fold.
- ParaFDEONet: one frozen 75,223,048-parameter checkpoint; no application-probability labels.
- Error units: percentage points, not relative percent.
- Source data: source_data_primary.csv, source_data_split_sensitivity.csv and split_audit.csv.
- Exports: editable-text SVG/PDF, 300 dpi PNG and 600 dpi TIFF.
- Font: Source Han Serif.
- Literature scope: adapted baseline, not a reproduction of the Nauck et al. dataset.
"""
    (output_dir / "FIGURE_QA.md").write_text(qa, encoding="utf-8")
    (output_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    save_json(
        output_dir / "status.json",
        {
            "state": "complete",
            "finished_unix_seconds": time.time(),
            "figure_files": [str(path.name) for path in figure_paths],
        },
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--survivability-dir",
        type=Path,
        required=True,
        help="Completed full/application_survivability directory.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs" / "literature_gbt_baseline.json",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    arguments = parser.parse_args()
    destination = run_experiment(
        arguments.survivability_dir.resolve(),
        arguments.config.resolve(),
        arguments.output_dir.resolve() if arguments.output_dir else None,
    )
    print(destination)


if __name__ == "__main__":
    main()
