#!/usr/bin/env python3
"""Confirm the five-seed survivability result on newly sampled histories."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
CODE_DIR = PROJECT_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from data import sample_histories
from equation import SmartGridConfig
from scripts.run_literature_gbt import regression_metrics
from scripts.run_multiseed_survivability import (
    classification_metrics,
    discover_checkpoints,
    scalar_summary,
)
from scripts.run_survivability import (
    direct_dde_risks,
    load_frozen_operator,
    parameter_grid,
    predict_operator_risks,
    save_json,
    save_npz,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary-run",
        type=Path,
        required=True,
        help="Run containing seed 20261001 and the immutable dataset.",
    )
    parser.add_argument(
        "--additional-run",
        type=Path,
        required=True,
        help="Run containing seeds 20261002--20261005.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--history-seed", type=int, default=2026091601)
    parser.add_argument("--history-count", type=int, default=512)
    parser.add_argument("--processes", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=64)
    return parser.parse_args()


def row_hashes(values: np.ndarray) -> set[str]:
    array = np.ascontiguousarray(values)
    return {
        hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
        for row in array
    }


def overlap_audit(histories: np.ndarray, dataset_dir: Path) -> dict[str, int]:
    fresh_hashes = row_hashes(histories)
    result = {
        "fresh_history_count": int(histories.shape[0]),
        "fresh_unique_history_count": int(len(fresh_hashes)),
    }
    for split in ("train", "validation", "test"):
        with np.load(dataset_dir / f"{split}.npz") as values:
            existing = row_hashes(values["histories"])
        result[f"exact_overlap_with_{split}"] = int(len(fresh_hashes & existing))
    return result


def continuous_metrics(
    direct_frequency: np.ndarray,
    direct_angle: np.ndarray,
    predicted_frequency: np.ndarray,
    predicted_angle: np.ndarray,
) -> dict[str, float]:
    frequency_error = np.abs(predicted_frequency - direct_frequency)
    angle_error = np.abs(predicted_angle - direct_angle)
    return {
        "frequency_mae_hz": float(np.mean(frequency_error) / (2.0 * math.pi)),
        "frequency_p99_absolute_error_hz": float(
            np.quantile(frequency_error, 0.99) / (2.0 * math.pi)
        ),
        "edge_angle_mae_degrees": float(np.degrees(np.mean(angle_error))),
        "edge_angle_p99_absolute_error_degrees": float(
            np.degrees(np.quantile(angle_error, 0.99))
        ),
    }


def aggregate(per_seed: list[dict[str, object]]) -> dict[str, object]:
    paths = {
        "probability_mae_percentage_points": (
            "probability_metrics",
            "mae_percentage_points",
        ),
        "probability_rmse_percentage_points": (
            "probability_metrics",
            "rmse_percentage_points",
        ),
        "probability_spearman": ("probability_metrics", "spearman"),
        "lowest_5_percent_recall": (
            "probability_metrics",
            "lowest_survivability_5_percent_recall",
        ),
        "lowest_10_percent_recall": (
            "probability_metrics",
            "lowest_survivability_10_percent_recall",
        ),
        "casewise_accuracy": ("classification_metrics", "casewise_accuracy"),
        "false_safe_rate_given_direct_unsafe": (
            "classification_metrics",
            "false_safe_rate_given_direct_unsafe",
        ),
        "false_alarm_rate_given_direct_safe": (
            "classification_metrics",
            "false_alarm_rate_given_direct_safe",
        ),
        "frequency_mae_hz": ("continuous_metrics", "frequency_mae_hz"),
        "frequency_p99_absolute_error_hz": (
            "continuous_metrics",
            "frequency_p99_absolute_error_hz",
        ),
        "edge_angle_mae_degrees": (
            "continuous_metrics",
            "edge_angle_mae_degrees",
        ),
        "edge_angle_p99_absolute_error_degrees": (
            "continuous_metrics",
            "edge_angle_p99_absolute_error_degrees",
        ),
        "inference_seconds": ("timing", "seconds"),
    }
    output: dict[str, object] = {}
    for name, (section, field) in paths.items():
        values = [float(seed[section][field]) for seed in per_seed]  # type: ignore[index]
        output[name] = scalar_summary(values)
    return output


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    complete = output_dir / "COMPLETE"
    if complete.exists():
        raise FileExistsError(f"audit already complete: {output_dir}")
    save_json(
        output_dir / "status.json",
        {"status": "running", "started_unix": time.time()},
    )

    config = json.loads(
        (PROJECT_DIR / "configs" / "experiment.json").read_text(encoding="utf-8")
    )
    equation = SmartGridConfig.from_mapping(config["equation"])
    data_config = config["data"]
    history_times = np.linspace(
        -equation.maximum_history,
        0.0,
        int(data_config["history_sensors"]),
        dtype=np.float64,
    )
    output_times = np.linspace(
        0.0,
        float(data_config["horizon"]),
        int(data_config["output_points"]),
        dtype=np.float64,
    )
    rng = np.random.default_rng(args.history_seed)
    histories = sample_histories(
        args.history_count,
        history_times,
        data_config["history_generator"],
        equation,
        rng,
    )
    parameters = parameter_grid(equation, 4)
    dataset_dir = args.primary_run.resolve() / "full" / "dataset"
    history_overlap = overlap_audit(histories, dataset_dir)
    if any(
        history_overlap[f"exact_overlap_with_{split}"] != 0
        for split in ("train", "validation", "test")
    ):
        raise RuntimeError(f"fresh-history overlap detected: {history_overlap}")
    save_npz(
        output_dir / "design.npz",
        histories=histories,
        history_times=history_times,
        output_times=output_times,
        parameters=parameters,
        history_seed=np.asarray(args.history_seed, dtype=np.int64),
    )

    pair_histories = np.repeat(histories, parameters.shape[0], axis=0)
    pair_parameters = np.tile(parameters, (histories.shape[0], 1))
    direct_start = time.perf_counter()
    direct_frequency_flat, direct_angle_flat = direct_dde_risks(
        pair_histories,
        pair_parameters,
        history_times,
        output_times,
        equation,
        0.005,
        args.processes,
        args.chunk_size,
    )
    direct_seconds = time.perf_counter() - direct_start
    shape = (histories.shape[0], parameters.shape[0])
    direct_frequency = direct_frequency_flat.reshape(shape)
    direct_angle = direct_angle_flat.reshape(shape)
    save_npz(
        output_dir / "direct_risks.npz",
        frequency_risk_rad_per_s=direct_frequency,
        edge_angle_risk_rad=direct_angle,
    )

    audit_rng = np.random.default_rng(args.history_seed + 1)
    audit_indices = np.sort(
        audit_rng.choice(pair_histories.shape[0], size=256, replace=False)
    )
    refined_frequency, refined_angle = direct_dde_risks(
        pair_histories[audit_indices],
        pair_parameters[audit_indices],
        history_times,
        output_times,
        equation,
        0.0025,
        args.processes,
        args.chunk_size,
    )
    frequency_threshold = 2.0 * math.pi * 0.15
    angle_threshold = math.radians(25.0)
    coarse_safe = (
        direct_frequency_flat[audit_indices] <= frequency_threshold
    ) & (direct_angle_flat[audit_indices] <= angle_threshold)
    refined_safe = (refined_frequency <= frequency_threshold) & (
        refined_angle <= angle_threshold
    )
    solver_audit = {
        "case_count": 256,
        "classification_agreement": float(np.mean(coarse_safe == refined_safe)),
        "maximum_frequency_risk_difference_hz": float(
            np.max(np.abs(refined_frequency - direct_frequency_flat[audit_indices]))
            / (2.0 * math.pi)
        ),
        "maximum_edge_angle_risk_difference_degrees": float(
            np.degrees(
                np.max(np.abs(refined_angle - direct_angle_flat[audit_indices]))
            )
        ),
    }

    checkpoints = discover_checkpoints(
        [args.primary_run.resolve(), args.additional_run.resolve()],
        "full/methods/parafdeonet/seed_*/best_model.pt",
        [20261001, 20261002, 20261003, 20261004, 20261005],
    )
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("highest")
    direct_probability = np.mean(
        (direct_frequency <= frequency_threshold)
        & (direct_angle <= angle_threshold),
        axis=0,
        dtype=np.float64,
    )
    per_seed: list[dict[str, object]] = []
    predictions: list[np.ndarray] = []
    for seed, checkpoint in checkpoints.items():
        model, metadata = load_frozen_operator(checkpoint, device)
        start = time.perf_counter()
        predicted_frequency, predicted_angle = predict_operator_risks(
            model,
            histories,
            parameters,
            output_times,
            equation.edges,
            device,
            32,
            64,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - start
        save_npz(
            output_dir / f"parafdeonet_risks_seed_{seed}.npz",
            frequency_risk_rad_per_s=predicted_frequency,
            edge_angle_risk_rad=predicted_angle,
        )
        predicted_probability = np.mean(
            (predicted_frequency <= frequency_threshold)
            & (predicted_angle <= angle_threshold),
            axis=0,
            dtype=np.float64,
        )
        predictions.append(predicted_probability)
        per_seed.append(
            {
                "training_seed": seed,
                "checkpoint": str(checkpoint),
                "checkpoint_metadata": metadata,
                "timing": {"seconds": float(seconds)},
                "probability_metrics": regression_metrics(
                    direct_probability,
                    predicted_probability,
                    [0.05, 0.10],
                ),
                "classification_metrics": classification_metrics(
                    direct_frequency,
                    direct_angle,
                    predicted_frequency,
                    predicted_angle,
                    frequency_threshold,
                    angle_threshold,
                ),
                "continuous_metrics": continuous_metrics(
                    direct_frequency,
                    direct_angle,
                    predicted_frequency,
                    predicted_angle,
                ),
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    prediction_array = np.asarray(predictions, dtype=np.float64)
    save_npz(
        output_dir / "predictions.npz",
        parameters=parameters,
        direct_survivability=direct_probability,
        parafdeonet_survivability=prediction_array,
        training_seeds=np.asarray(sorted(checkpoints), dtype=np.int64),
    )
    result = {
        "design": {
            "history_seed": args.history_seed,
            "history_count": int(histories.shape[0]),
            "parameter_count": int(parameters.shape[0]),
            "trajectory_count": int(histories.shape[0] * parameters.shape[0]),
            "frequency_threshold_hz": 0.15,
            "edge_angle_threshold_degrees": 25.0,
            "common_histories_across_parameters": True,
            "generated_after_model_training_and_original_application_evaluation": True,
        },
        "history_overlap_audit": history_overlap,
        "direct_solver": {
            "coarse_step_seconds": 0.005,
            "elapsed_seconds": float(direct_seconds),
            "refined_step_seconds": 0.0025,
            "refined_audit": solver_audit,
        },
        "direct_survivability_summary": {
            "minimum": float(np.min(direct_probability)),
            "mean": float(np.mean(direct_probability)),
            "maximum": float(np.max(direct_probability)),
            "sample_std_across_fixed_parameter_grid": float(
                np.std(direct_probability, ddof=1)
            ),
        },
        "parafdeonet_per_seed": per_seed,
        "parafdeonet_aggregate": aggregate(per_seed),
        "parafdeonet_seed_ensemble_mean": regression_metrics(
            direct_probability,
            np.mean(prediction_array, axis=0),
            [0.05, 0.10],
        ),
    }
    save_json(output_dir / "metrics.json", result)
    save_json(
        output_dir / "status.json",
        {"status": "complete", "completed_unix": time.time()},
    )
    complete.write_text("complete\n", encoding="utf-8")
    print(json.dumps(result["parafdeonet_aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
