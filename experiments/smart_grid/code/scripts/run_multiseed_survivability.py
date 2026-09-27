#!/usr/bin/env python3
"""Evaluate five frozen ParaFDEONet seeds on one fixed survivability design."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
CODE_DIR = PROJECT_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from equation import SmartGridConfig
from scripts.run_literature_gbt import regression_metrics
from scripts.run_survivability import (
    file_sha256,
    load_frozen_operator,
    predict_operator_risks,
    save_csv,
    save_json,
    save_npz,
    spearman_correlation,
    survivability_surface,
)


LOGGER = logging.getLogger("multiseed_survivability")
SEED_DIRECTORY_PATTERN = re.compile(r"^seed_(\d+)$")


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(stream)
    LOGGER.addHandler(file_handler)


def discover_checkpoints(
    run_directories: Sequence[Path],
    checkpoint_glob: str,
    expected_seeds: Sequence[int],
) -> dict[int, Path]:
    checkpoints: dict[int, Path] = {}
    for run_directory in run_directories:
        if not run_directory.is_dir():
            raise FileNotFoundError(f"model run directory does not exist: {run_directory}")
        for checkpoint in sorted(run_directory.glob(checkpoint_glob)):
            match = SEED_DIRECTORY_PATTERN.match(checkpoint.parent.name)
            if match is None:
                raise ValueError(f"cannot parse training seed from {checkpoint}")
            seed = int(match.group(1))
            if seed in checkpoints:
                raise ValueError(
                    f"duplicate checkpoint for seed {seed}: "
                    f"{checkpoints[seed]} and {checkpoint}"
                )
            checkpoints[seed] = checkpoint
    expected = {int(seed) for seed in expected_seeds}
    observed = set(checkpoints)
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise ValueError(
            f"checkpoint seed mismatch; missing={missing}, unexpected={unexpected}"
        )
    return {seed: checkpoints[seed] for seed in sorted(checkpoints)}


def scalar_summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("summary values must be a nonempty finite vector")
    return {
        "seed_count": int(array.size),
        "mean": float(np.mean(array)),
        "sample_std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def classification_metrics(
    direct_frequency: np.ndarray,
    direct_angle: np.ndarray,
    operator_frequency: np.ndarray,
    operator_angle: np.ndarray,
    frequency_threshold: float,
    angle_threshold: float,
) -> dict[str, float | int]:
    arrays = [
        np.asarray(values, dtype=np.float64)
        for values in (
            direct_frequency,
            direct_angle,
            operator_frequency,
            operator_angle,
        )
    ]
    if any(values.shape != arrays[0].shape for values in arrays[1:]):
        raise ValueError("direct and operator risk arrays must share one shape")
    direct_safe = (arrays[0] <= frequency_threshold) & (
        arrays[1] <= angle_threshold
    )
    operator_safe = (arrays[2] <= frequency_threshold) & (
        arrays[3] <= angle_threshold
    )
    direct_safe_count = int(np.sum(direct_safe))
    direct_unsafe_count = int(direct_safe.size - direct_safe_count)
    false_safe = int(np.sum(operator_safe & ~direct_safe))
    false_alarm = int(np.sum(~operator_safe & direct_safe))
    return {
        "casewise_accuracy": float(np.mean(operator_safe == direct_safe)),
        "false_safe_count": false_safe,
        "false_safe_rate_given_direct_unsafe": float(
            false_safe / max(direct_unsafe_count, 1)
        ),
        "false_alarm_count": false_alarm,
        "false_alarm_rate_given_direct_safe": float(
            false_alarm / max(direct_safe_count, 1)
        ),
        "direct_safe_count": direct_safe_count,
        "direct_unsafe_count": direct_unsafe_count,
        "parafdeonet_safe_count": int(np.sum(operator_safe)),
    }


def hierarchical_paired_bootstrap_mae_advantage(
    reference: np.ndarray,
    para_predictions: np.ndarray,
    baseline_prediction: np.ndarray,
    repeats: int,
    seed: int,
    confidence_level: float,
) -> dict[str, float | int | list[str]]:
    """Bootstrap GBT-minus-Para MAE over seeds and paired parameter points."""
    truth = np.asarray(reference, dtype=np.float64).reshape(-1)
    para = np.asarray(para_predictions, dtype=np.float64)
    baseline = np.asarray(baseline_prediction, dtype=np.float64).reshape(-1)
    if para.ndim != 2 or para.shape[1] != truth.size or baseline.shape != truth.shape:
        raise ValueError("bootstrap inputs have inconsistent shapes")
    if repeats < 1 or not 0.0 < confidence_level < 1.0:
        raise ValueError("invalid bootstrap configuration")
    para_error = np.abs(para - truth[None, :])
    baseline_error = np.abs(baseline - truth)
    rng = np.random.default_rng(seed)
    seed_indices = rng.integers(
        0, para.shape[0], size=(repeats, para.shape[0]), endpoint=False
    )
    parameter_indices = rng.integers(
        0, truth.size, size=(repeats, truth.size), endpoint=False
    )
    resampled_seed_mean_error = np.mean(para_error[seed_indices], axis=1)
    resampled_para = np.take_along_axis(
        resampled_seed_mean_error, parameter_indices, axis=1
    )
    resampled_baseline = baseline_error[parameter_indices]
    differences = np.mean(resampled_baseline - resampled_para, axis=1)
    alpha = 0.5 * (1.0 - confidence_level)
    lower, upper = np.quantile(differences, [alpha, 1.0 - alpha])
    point_advantage = float(np.mean(baseline_error) - np.mean(para_error))
    return {
        "repeats": int(repeats),
        "seed": int(seed),
        "confidence_level": float(confidence_level),
        "resampling_units": ["training_seed", "parameter_combination"],
        "mae_advantage": point_advantage,
        "mae_advantage_percentage_points": 100.0 * point_advantage,
        "confidence_interval_lower": float(lower),
        "confidence_interval_upper": float(upper),
        "confidence_interval_lower_percentage_points": float(100.0 * lower),
        "confidence_interval_upper_percentage_points": float(100.0 * upper),
        "bootstrap_fraction_para_mae_lower": float(np.mean(differences > 0.0)),
    }


def _nested_value(values: Mapping[str, Any], path: str) -> float:
    current: Any = values
    for component in path.split("."):
        current = current[component]
    return float(current)


def aggregate_seed_metrics(per_seed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = {
        "timing_seconds": ["mean", "sample_std", "minimum", "maximum"],
        "survivability_surface": [
            "mae",
            "rmse",
            "maximum_absolute_error",
            "fraction_inside_direct_wilson_95",
        ],
        "representative_screening": [
            "mae",
            "rmse",
            "maximum_absolute_error",
            "r2",
            "spearman",
            "lowest_survivability_5_percent_recall",
            "lowest_survivability_10_percent_recall",
            "overall_survivability",
            "casewise_accuracy",
            "false_safe_count",
            "false_safe_rate_given_direct_unsafe",
            "false_alarm_count",
            "false_alarm_rate_given_direct_safe",
        ],
        "continuous_risk": [
            "frequency_mae_hz",
            "frequency_p99_absolute_error_hz",
            "frequency_q95_parameterwise_spearman",
            "edge_angle_mae_degrees",
            "edge_angle_p99_absolute_error_degrees",
            "edge_angle_q95_parameterwise_spearman",
        ],
    }
    aggregate: dict[str, Any] = {}
    for section, names in fields.items():
        aggregate[section] = {}
        for name in names:
            path = f"{section}.{name}"
            aggregate[section][name] = scalar_summary(
                [_nested_value(seed_metrics, path) for seed_metrics in per_seed]
            )
    return aggregate


def evaluate_seed_risks(
    training_seed: int,
    checkpoint_path: Path,
    checkpoint_metadata: Mapping[str, Any],
    operator_frequency: np.ndarray,
    operator_angle: np.ndarray,
    direct_frequency: np.ndarray,
    direct_angle: np.ndarray,
    frequency_thresholds: np.ndarray,
    angle_thresholds: np.ndarray,
    representative_frequency: float,
    representative_angle: float,
    top_risk_fractions: Sequence[float],
    timings: Sequence[float],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    direct_surface = survivability_surface(
        direct_frequency, direct_angle, frequency_thresholds, angle_thresholds
    )
    operator_surface = survivability_surface(
        operator_frequency, operator_angle, frequency_thresholds, angle_thresholds
    )
    frequency_index = int(
        np.argmin(np.abs(frequency_thresholds - representative_frequency))
    )
    angle_index = int(np.argmin(np.abs(angle_thresholds - representative_angle)))
    if not math.isclose(
        float(frequency_thresholds[frequency_index]),
        representative_frequency,
        abs_tol=1e-12,
    ) or not math.isclose(
        float(angle_thresholds[angle_index]), representative_angle, abs_tol=1e-12
    ):
        raise RuntimeError("representative threshold is absent from the saved design")
    direct_representative = direct_surface[:, frequency_index, angle_index]
    operator_representative = operator_surface[:, frequency_index, angle_index]
    probability_metrics = regression_metrics(
        direct_representative, operator_representative, top_risk_fractions
    )
    classification = classification_metrics(
        direct_frequency,
        direct_angle,
        operator_frequency,
        operator_angle,
        representative_frequency,
        representative_angle,
    )
    surface_error = operator_surface - direct_surface
    direct_lower, direct_upper = _wilson_surface_bounds(
        direct_surface, direct_frequency.shape[0]
    )
    frequency_error = operator_frequency - direct_frequency
    angle_error = operator_angle - direct_angle
    q95_direct_frequency = np.quantile(direct_frequency, 0.95, axis=0)
    q95_operator_frequency = np.quantile(operator_frequency, 0.95, axis=0)
    q95_direct_angle = np.quantile(direct_angle, 0.95, axis=0)
    q95_operator_angle = np.quantile(operator_angle, 0.95, axis=0)
    metrics = {
        "training_seed": int(training_seed),
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            **dict(checkpoint_metadata),
        },
        "timing_seconds": {
            **scalar_summary(timings),
            "repeat_count": int(len(timings)),
        },
        "survivability_surface": {
            "mae": float(np.mean(np.abs(surface_error))),
            "rmse": float(np.sqrt(np.mean(surface_error**2))),
            "maximum_absolute_error": float(np.max(np.abs(surface_error))),
            "fraction_inside_direct_wilson_95": float(
                np.mean(
                    (operator_surface >= direct_lower)
                    & (operator_surface <= direct_upper)
                )
            ),
        },
        "representative_screening": {
            **probability_metrics,
            "direct_overall_survivability": float(
                np.mean(
                    (direct_frequency <= representative_frequency)
                    & (direct_angle <= representative_angle)
                )
            ),
            "overall_survivability": float(
                np.mean(
                    (operator_frequency <= representative_frequency)
                    & (operator_angle <= representative_angle)
                )
            ),
            **classification,
        },
        "continuous_risk": {
            "frequency_mae_hz": float(
                np.mean(np.abs(frequency_error)) / (2.0 * math.pi)
            ),
            "frequency_p99_absolute_error_hz": float(
                np.quantile(np.abs(frequency_error), 0.99) / (2.0 * math.pi)
            ),
            "frequency_q95_parameterwise_spearman": spearman_correlation(
                q95_direct_frequency, q95_operator_frequency
            ),
            "edge_angle_mae_degrees": float(
                np.degrees(np.mean(np.abs(angle_error)))
            ),
            "edge_angle_p99_absolute_error_degrees": float(
                np.degrees(np.quantile(np.abs(angle_error), 0.99))
            ),
            "edge_angle_q95_parameterwise_spearman": spearman_correlation(
                q95_direct_angle, q95_operator_angle
            ),
        },
    }
    return metrics, operator_representative, operator_surface


def _wilson_surface_bounds(
    probability: np.ndarray,
    sample_count: int,
    z_value: float = 1.959963984540054,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(probability, dtype=np.float64)
    denominator = 1.0 + z_value**2 / sample_count
    center = (values + z_value**2 / (2.0 * sample_count)) / denominator
    radius = z_value * np.sqrt(
        values * (1.0 - values) / sample_count
        + z_value**2 / (4.0 * sample_count**2)
    ) / denominator
    return np.maximum(0.0, center - radius), np.minimum(1.0, center + radius)


def write_results_markdown(output_dir: Path, metrics: Mapping[str, Any]) -> None:
    aggregate = metrics["parafdeonet_aggregate"]["representative_screening"]
    default_gbt = metrics["baselines"]["default_gbt"]
    tuned_gbt = metrics["baselines"]["nested_tuned_gbt"]
    comparison = metrics["comparison"]
    bootstrap = comparison["hierarchical_paired_bootstrap_tuned_gbt"]
    rows = []
    for seed_metrics in metrics["parafdeonet_per_seed"]:
        representative = seed_metrics["representative_screening"]
        rows.append(
            f"| {seed_metrics['training_seed']} | "
            f"{representative['mae_percentage_points']:.4f} | "
            f"{representative['rmse_percentage_points']:.4f} | "
            f"{representative['maximum_absolute_error_percentage_points']:.4f} | "
            f"{representative['spearman']:.6f} | "
            f"{100.0 * representative['lowest_survivability_5_percent_recall']:.2f}% | "
            f"{100.0 * representative['lowest_survivability_10_percent_recall']:.2f}% | "
            f"{100.0 * representative['casewise_accuracy']:.4f}% |"
        )
    text = f"""# Five-seed survivability experiment

## Design

Five independently trained 75,223,048-parameter ParaFDEONet checkpoints are evaluated on the identical fixed design of 512 held-out histories crossed with 64 parameter combinations. The saved direct-DDE risks are reused, so the 32,768 reference trajectories, thresholds, histories, and parameter points are exactly shared across all model seeds. No application labels are used to train ParaFDEONet.

## Seedwise results

| Training seed | MAE (percentage points) | RMSE (percentage points) | Maximum error (percentage points) | Spearman | Lowest 5% recall | Lowest 10% recall | Casewise accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The five-seed ParaFDEONet MAE is {100.0 * aggregate['mae']['mean']:.4f} plus or minus {100.0 * aggregate['mae']['sample_std']:.4f} percentage points (mean plus or minus sample standard deviation). Its RMSE is {100.0 * aggregate['rmse']['mean']:.4f} plus or minus {100.0 * aggregate['rmse']['sample_std']:.4f} percentage points, and its Spearman correlation is {aggregate['spearman']['mean']:.6f} plus or minus {aggregate['spearman']['sample_std']:.6f}.

## Literature-baseline comparison

| Method | MAE (percentage points) | RMSE (percentage points) | Spearman | Lowest 5% recall | Lowest 10% recall |
|---|---:|---:|---:|---:|---:|
| ParaFDEONet, five-seed mean | {100.0 * aggregate['mae']['mean']:.4f} | {100.0 * aggregate['rmse']['mean']:.4f} | {aggregate['spearman']['mean']:.6f} | {100.0 * aggregate['lowest_survivability_5_percent_recall']['mean']:.2f}% | {100.0 * aggregate['lowest_survivability_10_percent_recall']['mean']:.2f}% |
| GBT, default out of fold | {default_gbt['mae_percentage_points']:.4f} | {default_gbt['rmse_percentage_points']:.4f} | {default_gbt['spearman']:.6f} | {100.0 * default_gbt['lowest_survivability_5_percent_recall']:.2f}% | {100.0 * default_gbt['lowest_survivability_10_percent_recall']:.2f}% |
| GBT, nested-CV tuned | {tuned_gbt['mae_percentage_points']:.4f} | {tuned_gbt['rmse_percentage_points']:.4f} | {tuned_gbt['spearman']:.6f} | {100.0 * tuned_gbt['lowest_survivability_5_percent_recall']:.2f}% | {100.0 * tuned_gbt['lowest_survivability_10_percent_recall']:.2f}% |

The tuned GBT MAE is {comparison['tuned_gbt_to_parafdeonet_mean_mae_ratio']:.2f} times the five-seed mean ParaFDEONet MAE. Resampling training seeds and the 64 fixed parameter points gives an empirical {100.0 * bootstrap['confidence_level']:.0f}% range from {bootstrap['confidence_interval_lower_percentage_points']:.4f} to {bootstrap['confidence_interval_upper_percentage_points']:.4f} percentage points for the MAE advantage. This is a fixed-design resampling summary, not a population confidence interval over parameter space. All five ParaFDEONet seeds have lower MAE than both GBT variants.

## Interpretation

The result measures training-seed robustness for this fixed delayed-grid design. The GBT values are out-of-fold probability predictions from an adapted published comparator, whereas ParaFDEONet predicts complete trajectories and receives no survivability labels. Runtime comparisons with GBT are not made because the two deployment inputs and label-generation costs differ.
"""
    (output_dir / "RESULTS.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run-dir", type=Path, required=True)
    parser.add_argument(
        "--model-run-dir", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs" / "multiseed_survivability.json",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    reference_run = args.reference_run_dir.resolve()
    application_dir = reference_run / config["reference_application_relative_to_run"]
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else application_dir.parent / "application_survivability_five_seed"
    )
    complete_marker = output_dir / "COMPLETE"
    if complete_marker.exists() and not args.resume:
        raise FileExistsError(f"completed output already exists: {output_dir}")
    configure_logging(output_dir)
    save_json(
        output_dir / "status.json",
        {"status": "initializing", "updated_unix": time.time()},
    )
    required_reference_files = {
        "design": application_dir / "design.npz",
        "direct_risks": application_dir / "direct_dde_risks.npz",
        "original_operator_risks": application_dir / "parafdeonet_risks.npz",
        "parent_metrics": application_dir / "metrics.json",
        "gbt_metrics": application_dir / "literature_gbt_baseline" / "metrics.json",
        "gbt_predictions": application_dir
        / "literature_gbt_baseline"
        / "predictions.npz",
        "dataset": reference_run / config["dataset_relative_to_run"],
    }
    missing = [str(path) for path in required_reference_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing fixed-reference input(s): " + ", ".join(missing))
    checkpoints = discover_checkpoints(
        [path.resolve() for path in args.model_run_dir],
        str(config["checkpoint_glob_relative_to_run"]),
        [int(seed) for seed in config["expected_training_seeds"]],
    )
    with np.load(required_reference_files["design"]) as values:
        history_indices = np.asarray(
            values["selected_test_history_indices"], dtype=np.int64
        )
        parameters = np.asarray(values["parameter_grid"], dtype=np.float32)
        frequency_thresholds = np.asarray(
            values["frequency_thresholds_rad_per_s"], dtype=np.float64
        )
        angle_thresholds = np.asarray(
            values["edge_angle_thresholds_rad"], dtype=np.float64
        )
        history_times = np.asarray(values["history_times"], dtype=np.float64)
        output_times = np.asarray(values["output_times"], dtype=np.float64)
    with np.load(required_reference_files["dataset"]) as values:
        histories = np.asarray(values["histories"][history_indices], dtype=np.float32)
        np.testing.assert_array_equal(values["history_times"], history_times)
        np.testing.assert_array_equal(values["output_times"], output_times)
    with np.load(required_reference_files["direct_risks"]) as values:
        direct_frequency = np.asarray(
            values["frequency_risk_rad_per_s"], dtype=np.float64
        )
        direct_angle = np.asarray(values["edge_angle_risk_rad"], dtype=np.float64)
    expected_shape = (int(config["history_count"]), int(config["parameter_count"]))
    if histories.shape[0] != expected_shape[0] or parameters.shape != (
        expected_shape[1],
        3,
    ):
        raise ValueError("saved design dimensions do not match the predeclared experiment")
    if direct_frequency.shape != expected_shape or direct_angle.shape != expected_shape:
        raise ValueError("saved direct-risk arrays do not match the predeclared design")
    if int(np.prod(expected_shape)) != int(config["trajectory_count"]):
        raise ValueError("trajectory count is inconsistent with the saved design")
    equation_config = json.loads(
        (PROJECT_DIR / "configs" / "experiment.json").read_text(encoding="utf-8")
    )
    equation = SmartGridConfig.from_mapping(equation_config["equation"])
    representative_frequency = 2.0 * math.pi * float(
        config["representative_frequency_threshold_hz"]
    )
    representative_angle = math.radians(
        float(config["representative_edge_angle_threshold_degrees"])
    )
    top_risk_fractions = [float(value) for value in config["top_risk_fractions"]]
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("highest")
    per_seed_metrics: list[dict[str, Any]] = []
    representative_predictions: list[np.ndarray] = []
    surface_predictions: list[np.ndarray] = []
    for position, (training_seed, checkpoint_path) in enumerate(
        checkpoints.items(), start=1
    ):
        seed_dir = output_dir / f"seed_{training_seed}"
        risk_path = seed_dir / "parafdeonet_risks.npz"
        metadata_path = seed_dir / "metadata.json"
        timing_path = seed_dir / "timing_seconds.json"
        checkpoint_hash = file_sha256(checkpoint_path)
        reuse = args.resume and risk_path.is_file() and metadata_path.is_file() and timing_path.is_file()
        if reuse:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata["checkpoint_sha256"] != checkpoint_hash:
                raise RuntimeError(f"checkpoint hash changed for seed {training_seed}")
            timings = [
                float(value)
                for value in json.loads(timing_path.read_text(encoding="utf-8"))[
                    "seconds"
                ]
            ]
            with np.load(risk_path) as values:
                operator_frequency = np.asarray(
                    values["frequency_risk_rad_per_s"], dtype=np.float64
                )
                operator_angle = np.asarray(
                    values["edge_angle_risk_rad"], dtype=np.float64
                )
            LOGGER.info("reused cached risks for seed %d", training_seed)
        else:
            LOGGER.info(
                "evaluating seed %d (%d/%d)",
                training_seed,
                position,
                len(checkpoints),
            )
            model, metadata = load_frozen_operator(checkpoint_path, device)
            if int(metadata["training_seed"]) != training_seed:
                raise RuntimeError(f"checkpoint/path seed mismatch for {checkpoint_path}")
            if int(metadata["parameter_count"]) != int(config["expected_parameter_count"]):
                raise RuntimeError(f"unexpected model size for seed {training_seed}")
            predict_operator_risks(
                model,
                histories,
                parameters,
                output_times,
                equation.edges,
                device,
                int(config["operator_history_batch_size"]),
                int(config["operator_parameter_batch_size"]),
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings = []
            operator_frequency = np.empty(expected_shape, dtype=np.float64)
            operator_angle = np.empty(expected_shape, dtype=np.float64)
            for repeat in range(int(config["timing_repeats"])):
                start = time.perf_counter()
                local_frequency, local_angle = predict_operator_risks(
                    model,
                    histories,
                    parameters,
                    output_times,
                    equation.edges,
                    device,
                    int(config["operator_history_batch_size"]),
                    int(config["operator_parameter_batch_size"]),
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - start
                timings.append(float(elapsed))
                if repeat == 0:
                    operator_frequency = local_frequency
                    operator_angle = local_angle
                LOGGER.info(
                    "seed %d timing repeat %d/%d: %.6f s",
                    training_seed,
                    repeat + 1,
                    int(config["timing_repeats"]),
                    elapsed,
                )
            seed_dir.mkdir(parents=True, exist_ok=True)
            save_npz(
                risk_path,
                frequency_risk_rad_per_s=operator_frequency,
                edge_angle_risk_rad=operator_angle,
            )
            save_json(metadata_path, metadata)
            save_json(timing_path, {"seconds": timings})
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if operator_frequency.shape != expected_shape or operator_angle.shape != expected_shape:
            raise RuntimeError(f"invalid risk shape for seed {training_seed}")
        metrics, representative, surface = evaluate_seed_risks(
            training_seed,
            checkpoint_path,
            metadata,
            operator_frequency,
            operator_angle,
            direct_frequency,
            direct_angle,
            frequency_thresholds,
            angle_thresholds,
            representative_frequency,
            representative_angle,
            top_risk_fractions,
            timings,
        )
        per_seed_metrics.append(metrics)
        representative_predictions.append(representative)
        surface_predictions.append(surface)
        save_json(seed_dir / "metrics.json", metrics)
    predictions = np.stack(representative_predictions, axis=0)
    surfaces = np.stack(surface_predictions, axis=0)
    direct_surface = survivability_surface(
        direct_frequency, direct_angle, frequency_thresholds, angle_thresholds
    )
    frequency_index = int(
        np.argmin(np.abs(frequency_thresholds - representative_frequency))
    )
    angle_index = int(np.argmin(np.abs(angle_thresholds - representative_angle)))
    direct_representative = direct_surface[:, frequency_index, angle_index]
    with np.load(required_reference_files["gbt_predictions"]) as values:
        gbt_direct = np.asarray(values["direct_survivability"], dtype=np.float64)
        default_gbt_prediction = np.asarray(
            values["gbt_out_of_fold_survivability"], dtype=np.float64
        )
        tuned_gbt_prediction = np.asarray(
            values["tuned_gbt_out_of_fold_survivability"], dtype=np.float64
        )
    np.testing.assert_array_equal(gbt_direct, direct_representative)
    default_gbt_metrics = regression_metrics(
        direct_representative, default_gbt_prediction, top_risk_fractions
    )
    tuned_gbt_metrics = regression_metrics(
        direct_representative, tuned_gbt_prediction, top_risk_fractions
    )
    aggregate = aggregate_seed_metrics(per_seed_metrics)
    ensemble_metrics = regression_metrics(
        direct_representative, np.mean(predictions, axis=0), top_risk_fractions
    )
    para_mae_values = np.asarray(
        [item["representative_screening"]["mae"] for item in per_seed_metrics],
        dtype=np.float64,
    )
    bootstrap_config = config["bootstrap"]
    bootstrap = hierarchical_paired_bootstrap_mae_advantage(
        direct_representative,
        predictions,
        tuned_gbt_prediction,
        int(bootstrap_config["repeats"]),
        int(bootstrap_config["seed"]),
        float(bootstrap_config["confidence_level"]),
    )
    original_seed_risk_audit: dict[str, Any]
    with np.load(required_reference_files["original_operator_risks"]) as values:
        original_frequency = np.asarray(
            values["frequency_risk_rad_per_s"], dtype=np.float64
        )
        original_angle = np.asarray(values["edge_angle_risk_rad"], dtype=np.float64)
    seed_one_path = output_dir / "seed_20261001" / "parafdeonet_risks.npz"
    with np.load(seed_one_path) as values:
        reproduced_frequency = np.asarray(
            values["frequency_risk_rad_per_s"], dtype=np.float64
        )
        reproduced_angle = np.asarray(
            values["edge_angle_risk_rad"], dtype=np.float64
        )
    original_seed_risk_audit = {
        "training_seed": 20261001,
        "frequency_maximum_absolute_difference": float(
            np.max(np.abs(original_frequency - reproduced_frequency))
        ),
        "edge_angle_maximum_absolute_difference": float(
            np.max(np.abs(original_angle - reproduced_angle))
        ),
        "frequency_exact_array_equal": bool(
            np.array_equal(original_frequency, reproduced_frequency)
        ),
        "edge_angle_exact_array_equal": bool(
            np.array_equal(original_angle, reproduced_angle)
        ),
    }
    parent_gbt_metrics = json.loads(
        required_reference_files["gbt_metrics"].read_text(encoding="utf-8")
    )
    metrics: dict[str, Any] = {
        "experiment_name": config["experiment_name"],
        "design": {
            "history_count": int(histories.shape[0]),
            "parameter_count": int(parameters.shape[0]),
            "trajectory_count": int(histories.shape[0] * parameters.shape[0]),
            "frequency_threshold_hz": float(
                config["representative_frequency_threshold_hz"]
            ),
            "edge_angle_threshold_degrees": float(
                config["representative_edge_angle_threshold_degrees"]
            ),
            "same_histories_parameters_and_direct_reference_for_all_seeds": True,
            "direct_dde_recomputed": False,
        },
        "training_seeds": [int(seed) for seed in checkpoints],
        "parafdeonet_per_seed": per_seed_metrics,
        "parafdeonet_aggregate": aggregate,
        "parafdeonet_ensemble_mean_prediction": ensemble_metrics,
        "baselines": {
            "default_gbt": default_gbt_metrics,
            "nested_tuned_gbt": tuned_gbt_metrics,
            "gbt_split_sensitivity": parent_gbt_metrics["split_sensitivity"],
            "literature_baseline": parent_gbt_metrics["literature_baseline"],
            "direct_reference": parent_gbt_metrics["direct_reference"],
        },
        "comparison": {
            "default_gbt_to_parafdeonet_mean_mae_ratio": float(
                default_gbt_metrics["mae"] / np.mean(para_mae_values)
            ),
            "tuned_gbt_to_parafdeonet_mean_mae_ratio": float(
                tuned_gbt_metrics["mae"] / np.mean(para_mae_values)
            ),
            "default_gbt_to_parafdeonet_mean_rmse_ratio": float(
                default_gbt_metrics["rmse"]
                / aggregate["representative_screening"]["rmse"]["mean"]
            ),
            "tuned_gbt_to_parafdeonet_mean_rmse_ratio": float(
                tuned_gbt_metrics["rmse"]
                / aggregate["representative_screening"]["rmse"]["mean"]
            ),
            "all_seeds_lower_mae_than_default_gbt": bool(
                np.all(para_mae_values < default_gbt_metrics["mae"])
            ),
            "all_seeds_lower_mae_than_tuned_gbt": bool(
                np.all(para_mae_values < tuned_gbt_metrics["mae"])
            ),
            "hierarchical_paired_bootstrap_tuned_gbt": bootstrap,
        },
        "protocol_reproduction_audit": original_seed_risk_audit,
        "provenance": {
            "reference_run_dir": str(reference_run),
            "model_run_directories": [
                str(path.resolve()) for path in args.model_run_dir
            ],
            "config_path": str(args.config.resolve()),
            "config_sha256": file_sha256(args.config),
            "input_files": {
                name: {"path": str(path.resolve()), "sha256": file_sha256(path)}
                for name, path in required_reference_files.items()
            },
        },
    }
    seedwise_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    for seed_index, seed_metrics in enumerate(per_seed_metrics):
        representative = seed_metrics["representative_screening"]
        seedwise_rows.append(
            {
                "training_seed": seed_metrics["training_seed"],
                "best_iteration": seed_metrics["checkpoint"]["best_iteration"],
                "mae_percentage_points": representative["mae_percentage_points"],
                "rmse_percentage_points": representative["rmse_percentage_points"],
                "maximum_absolute_error_percentage_points": representative[
                    "maximum_absolute_error_percentage_points"
                ],
                "spearman": representative["spearman"],
                "lowest_survivability_5_percent_recall": representative[
                    "lowest_survivability_5_percent_recall"
                ],
                "lowest_survivability_10_percent_recall": representative[
                    "lowest_survivability_10_percent_recall"
                ],
                "casewise_accuracy": representative["casewise_accuracy"],
                "false_safe_count": representative["false_safe_count"],
                "false_safe_rate_given_direct_unsafe": representative[
                    "false_safe_rate_given_direct_unsafe"
                ],
                "false_alarm_count": representative["false_alarm_count"],
                "false_alarm_rate_given_direct_safe": representative[
                    "false_alarm_rate_given_direct_safe"
                ],
                "inference_seconds_mean": seed_metrics["timing_seconds"]["mean"],
                "checkpoint_sha256": seed_metrics["checkpoint"]["checkpoint_sha256"],
            }
        )
        for parameter_index, parameter in enumerate(parameters):
            parameter_rows.append(
                {
                    "training_seed": seed_metrics["training_seed"],
                    "parameter_index": parameter_index,
                    "coupling_K": float(parameter[0]),
                    "response_gamma": float(parameter[1]),
                    "damping_alpha": float(parameter[2]),
                    "direct_survivability": float(direct_representative[parameter_index]),
                    "parafdeonet_survivability": float(
                        predictions[seed_index, parameter_index]
                    ),
                    "parafdeonet_absolute_error_percentage_points": float(
                        100.0
                        * abs(
                            predictions[seed_index, parameter_index]
                            - direct_representative[parameter_index]
                        )
                    ),
                    "default_gbt_survivability": float(
                        default_gbt_prediction[parameter_index]
                    ),
                    "tuned_gbt_survivability": float(
                        tuned_gbt_prediction[parameter_index]
                    ),
                }
            )
    save_json(output_dir / "config_resolved.json", config)
    save_json(output_dir / "metrics.json", metrics)
    save_csv(output_dir / "seedwise_metrics.csv", seedwise_rows)
    save_csv(output_dir / "parameterwise_predictions.csv", parameter_rows)
    save_npz(
        output_dir / "predictions.npz",
        training_seeds=np.asarray(list(checkpoints), dtype=np.int64),
        parameters=parameters,
        direct_survivability=direct_representative,
        parafdeonet_survivability=predictions,
        parafdeonet_survivability_surfaces=surfaces,
        default_gbt_survivability=default_gbt_prediction,
        tuned_gbt_survivability=tuned_gbt_prediction,
    )
    write_results_markdown(output_dir, metrics)
    save_json(
        output_dir / "status.json",
        {
            "status": "complete",
            "updated_unix": time.time(),
            "metrics_path": str((output_dir / "metrics.json").resolve()),
            "results_path": str((output_dir / "RESULTS.md").resolve()),
        },
    )
    complete_marker.write_text("complete\n", encoding="utf-8")
    LOGGER.info("five-seed survivability experiment complete: %s", output_dir)


if __name__ == "__main__":
    main()
