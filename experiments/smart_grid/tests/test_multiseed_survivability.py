from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from scripts.run_multiseed_survivability import (
    classification_metrics,
    discover_checkpoints,
    hierarchical_paired_bootstrap_mae_advantage,
    scalar_summary,
)


def test_scalar_summary_uses_sample_standard_deviation() -> None:
    result = scalar_summary([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result["seed_count"] == 5
    assert result["mean"] == pytest.approx(3.0)
    assert result["sample_std"] == pytest.approx(np.std([1, 2, 3, 4, 5], ddof=1))
    assert result["minimum"] == pytest.approx(1.0)
    assert result["maximum"] == pytest.approx(5.0)


def test_classification_metrics_separate_false_safe_and_false_alarm() -> None:
    direct_frequency = np.asarray([[0.1, 0.1], [0.3, 0.1]])
    direct_angle = np.asarray([[0.1, 0.3], [0.1, 0.1]])
    operator_frequency = np.asarray([[0.1, 0.1], [0.1, 0.3]])
    operator_angle = np.asarray([[0.1, 0.1], [0.1, 0.1]])
    result = classification_metrics(
        direct_frequency,
        direct_angle,
        operator_frequency,
        operator_angle,
        frequency_threshold=0.2,
        angle_threshold=0.2,
    )
    assert result["direct_safe_count"] == 2
    assert result["direct_unsafe_count"] == 2
    assert result["false_safe_count"] == 2
    assert result["false_safe_rate_given_direct_unsafe"] == pytest.approx(1.0)
    assert result["false_alarm_count"] == 1
    assert result["false_alarm_rate_given_direct_safe"] == pytest.approx(0.5)
    assert result["casewise_accuracy"] == pytest.approx(0.25)


def test_hierarchical_bootstrap_detects_uniform_mae_advantage() -> None:
    truth = np.asarray([0.2, 0.4, 0.6, 0.8])
    para = np.asarray(
        [
            [0.21, 0.39, 0.61, 0.79],
            [0.22, 0.38, 0.62, 0.78],
            [0.21, 0.41, 0.59, 0.81],
        ]
    )
    baseline = np.asarray([0.30, 0.30, 0.70, 0.70])
    result = hierarchical_paired_bootstrap_mae_advantage(
        truth,
        para,
        baseline,
        repeats=2000,
        seed=7,
        confidence_level=0.95,
    )
    assert result["mae_advantage"] > 0.0
    assert result["confidence_interval_lower"] > 0.0
    assert result["bootstrap_fraction_para_mae_lower"] == pytest.approx(1.0)


def test_discover_checkpoints_requires_exact_seed_set(tmp_path: Path) -> None:
    run = tmp_path / "run"
    for seed in (11, 12):
        checkpoint = (
            run
            / "full"
            / "methods"
            / "parafdeonet"
            / f"seed_{seed}"
            / "best_model.pt"
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
    found = discover_checkpoints(
        [run], "full/methods/parafdeonet/seed_*/best_model.pt", [11, 12]
    )
    assert list(found) == [11, 12]
    with pytest.raises(ValueError, match="missing"):
        discover_checkpoints(
            [run], "full/methods/parafdeonet/seed_*/best_model.pt", [11, 12, 13]
        )
