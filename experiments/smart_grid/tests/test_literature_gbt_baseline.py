from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from scripts.run_literature_gbt import (
    normalize_parameters,
    nested_tuned_gbt_predictions,
    out_of_fold_gbt_predictions,
    paired_bootstrap_mae_advantage,
    regression_metrics,
    top_risk_recall,
)


def test_normalize_parameters_uses_predeclared_bounds() -> None:
    parameters = np.asarray([[6.0, 0.18, 0.07], [8.0, 0.25, 0.10], [10.0, 0.32, 0.13]])
    bounds = np.asarray([[6.0, 10.0], [0.18, 0.32], [0.07, 0.13]])
    normalized = normalize_parameters(parameters, bounds)
    np.testing.assert_allclose(normalized[0], -1.0)
    np.testing.assert_allclose(normalized[1], 0.0, atol=1e-14)
    np.testing.assert_allclose(normalized[2], 1.0)
    with pytest.raises(ValueError, match="outside"):
        normalize_parameters(np.asarray([[10.1, 0.25, 0.10]]), bounds)


def test_oof_gbt_assigns_each_point_once_without_overlap() -> None:
    pytest.importorskip("sklearn")
    grid = np.linspace(-1.0, 1.0, 24)
    features = np.column_stack((grid, grid**2, np.sin(grid)))
    target = 0.7 + 0.1 * grid - 0.03 * grid**2
    prediction, folds, audit, parameters = out_of_fold_gbt_predictions(
        features,
        target,
        fold_count=6,
        split_seed=17,
        estimator_random_state=19,
    )
    assert prediction.shape == target.shape
    assert np.all(np.isfinite(prediction))
    assert np.array_equal(np.unique(folds), np.arange(6))
    assert len(audit) == 6
    assert all(row["overlap_count"] == 0 for row in audit)
    assert sum(row["test_count"] for row in audit) == target.size
    assert parameters["random_state"] == 19


def test_top_risk_recall_uses_lowest_survivability() -> None:
    reference = np.asarray([0.10, 0.20, 0.30, 0.40, 0.50])
    prediction = np.asarray([0.11, 0.35, 0.21, 0.39, 0.51])
    recall, count = top_risk_recall(reference, prediction, 0.4)
    assert count == 2
    assert recall == pytest.approx(0.5)
    metrics = regression_metrics(reference, prediction, [0.4])
    assert metrics["lowest_survivability_40_percent_count"] == 2
    assert metrics["lowest_survivability_40_percent_recall"] == pytest.approx(0.5)


def test_nested_tuning_predicts_only_outer_held_out_points() -> None:
    pytest.importorskip("sklearn")
    grid = np.linspace(-1.0, 1.0, 20)
    features = np.column_stack((grid, grid**2, np.sin(grid)))
    target = 0.7 + 0.1 * grid - 0.03 * grid**2
    prediction, folds, choices = nested_tuned_gbt_predictions(
        features,
        target,
        outer_fold_count=4,
        outer_split_seed=29,
        estimator_random_state=31,
        tuning_config={
            "inner_fold_count": 3,
            "inner_split_seed_base": 37,
            "scoring": "neg_mean_absolute_error",
            "parallel_jobs": 1,
            "parameter_grid": {
                "n_estimators": [20, 40],
                "max_depth": [1, 2],
            },
        },
    )
    assert np.all(np.isfinite(prediction))
    assert np.array_equal(np.unique(folds), np.arange(4))
    assert len(choices) == 4
    assert all(row["outer_train_test_overlap_count"] == 0 for row in choices)
    assert all(row["outer_test_count"] == 5 for row in choices)


def test_paired_bootstrap_reports_positive_para_advantage() -> None:
    reference = np.linspace(0.5, 0.9, 16)
    para = reference + 0.001
    baseline = reference + np.linspace(0.01, 0.04, 16)
    result = paired_bootstrap_mae_advantage(
        reference,
        para,
        baseline,
        repeats=2000,
        seed=23,
        confidence_level=0.95,
    )
    assert result["mae_advantage"] > 0.0
    assert result["confidence_interval_lower"] > 0.0
    assert result["bootstrap_fraction_para_mae_lower"] == pytest.approx(1.0)
