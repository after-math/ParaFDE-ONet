#!/usr/bin/env python3
"""Run the four-way combination-generalization experiment for the variable-delay system."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import csv
import gc
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import platform
import socket
import sys
import time
import traceback
from typing import Any, Mapping


CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
import torch

from data import (
    COMBINATION_SPLIT_NAMES,
    combination_dataset_signature,
    generate_or_load_combination_dataset,
    load_combination_dataset,
)
from model import MODEL_DISPLAY_NAMES, MODEL_TYPES, load_operator_checkpoint
from reporting import plot_method_forward, plot_training_history, save_figure
from scripts.pipeline import (
    build_round_robin_queues,
    configure_logging,
    load_config,
    notify,
    parse_devices,
    resolve_stage_config,
)
from training import evaluate_operator, save_csv, save_json, train_operator


LOGGER = logging.getLogger("variable_delay_combination")
plt.style.use(["science", "no-latex", "grid"])
matplotlib.rcParams['font.family'] = 'Source Han Serif'
matplotlib.rcParams["axes.unicode_minus"] = False


def validate_combination_config(config: Mapping[str, Any]) -> None:
    """Validate the predeclared four-method, five-seed holdout protocol.

    ``config`` is the parsed JSON mapping.  The function returns ``None`` when the
    method registry, five unique seeds, four split names, primary UH--UP endpoint,
    training-edge physics and sample counts are valid; otherwise it raises.  For
    example an ``online_cartesian`` physics mode is rejected because it leaks unseen
    combinations.  It has no side effect and ``main`` calls it before creating data.
    """
    if tuple(config.get("methods", ())) != MODEL_TYPES:
        raise ValueError(f"methods must be exactly {MODEL_TYPES}")
    seeds = config.get("randomness", {}).get("training_seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) != 5
        or len(seeds) != len(set(seeds))
        or any(not isinstance(seed, int) or seed < 0 for seed in seeds)
    ):
        raise ValueError("five unique nonnegative training seeds are required")
    protocol = config["combination_generalization"]
    if tuple(protocol.get("split_order", ())) != COMBINATION_SPLIT_NAMES:
        raise ValueError("split_order does not match the fixed four-way protocol")
    if protocol.get("primary_split") != "unseen_history_unseen_parameter":
        raise ValueError("UH--UP must be the predeclared primary split")
    if config["operator"].get("physics_pairing_mode") != "training_edges":
        raise ValueError("combination physics must use observed training edges only")
    positive = (
        "validation_history_count", "seen_test_history_count",
        "unseen_test_history_count", "validation_parameter_count",
        "unseen_parameter_count", "evaluation_pairs_per_history",
    )
    if any(int(protocol[name]) < 1 for name in positive):
        raise ValueError("combination sample counts must be positive")
    bounds = np.asarray(protocol["training_parameter_bounds"], dtype=np.float64)
    physical = np.asarray(config["equation"]["parameter_bounds"], dtype=np.float64)
    if (
        bounds.shape != (2, 2)
        or np.any(bounds[:, 0] < physical[:, 0])
        or np.any(bounds[:, 1] > physical[:, 1])
    ):
        raise ValueError("training parameter bounds must lie inside physical bounds")


def resolve_combination_stage(
    config: Mapping[str, Any], smoke: bool
) -> dict[str, Any]:
    """Return an isolated formal or smoke configuration for the same code path.

    ``config`` is never modified and ``smoke`` selects small data/network values.
    The return starts from the parent's standard stage resolver, then replaces the
    six combination counts from smoke keys.  For example smoke uses four histories
    per semantic group and two evaluation edges per history.  It has no file side
    effect and ``run_stage`` is the only caller.
    """
    resolved = resolve_stage_config(config, smoke)
    if smoke:
        small = resolved["smoke"]
        protocol = resolved["combination_generalization"]
        protocol.update({
            "validation_history_count": int(small["combination_validation_history_count"]),
            "seen_test_history_count": int(small["combination_seen_test_history_count"]),
            "unseen_test_history_count": int(small["combination_unseen_test_history_count"]),
            "validation_parameter_count": int(small["combination_validation_parameter_count"]),
            "unseen_parameter_count": int(small["combination_unseen_parameter_count"]),
            "evaluation_pairs_per_history": int(
                small["combination_evaluation_pairs_per_history"]
            ),
        })
    return resolved


def parse_method_selection(value: str | None) -> list[str]:
    """Parse an optional comma-separated subset in the registered paper order.

    ``value=None`` returns all four methods; otherwise every unique key must occur in
    ``MODEL_TYPES``.  For example ``'two_branch_mionet,separate_parameter_shared'``
    returns those two keys.  Invalid aliases and duplicates raise.  It has no side
    effect and ``main`` uses the result to support targeted补 runs.
    """
    if value is None:
        return list(MODEL_TYPES)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("--methods must contain unique registered method keys")
    unknown = [item for item in selected if item not in MODEL_TYPES]
    if unknown:
        raise ValueError(f"unknown method keys: {unknown}")
    return [method for method in MODEL_TYPES if method in selected]


def evaluate_named_splits(
    model: torch.nn.Module,
    named_splits: Mapping[str, Any],
    device: torch.device,
    batch_size: int,
    model_type: str,
    training_seed: int,
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    """Evaluate one frozen best checkpoint on all four semantic holdouts.

    ``model`` is the selected operator, ``named_splits`` maps four split names to
    true trajectories, ``device`` and ``batch_size`` control inference, and method/
    seed label every row.  Returns aggregate metrics by split and per-case rows.  A
    formal call produces 8192 rows.  It performs inference only and is called after
    validation has already selected ``best_model.pt``.
    """
    metrics: dict[str, dict[str, float]] = {}
    per_case_rows: list[dict[str, Any]] = []
    for split_name in COMBINATION_SPLIT_NAMES:
        split_metrics, _, rows = evaluate_operator(
            model, named_splits[split_name], device, batch_size
        )
        metrics[split_name] = split_metrics
        for row in rows:
            per_case_rows.append({
                "model_type": model_type,
                "method": MODEL_DISPLAY_NAMES[model_type],
                "training_seed": int(training_seed),
                "split": split_name,
                **row,
            })
    return metrics, per_case_rows


def operator_job(
    model_type: str,
    training_seed: int,
    output_root_text: str,
    resolved_config: Mapping[str, Any],
    splits: Mapping[str, Any],
    named_splits: Mapping[str, Any],
    device_text: str,
    smoke: bool,
) -> dict[str, Any]:
    """Train one method--seed job and evaluate its best checkpoint on four splits.

    Inputs identify the method, independent seed, stage root, resolved config, shared
    data, assigned device and smoke status.  Returns ``generalization_metrics``.  It
    writes/resumes checkpoints through the common training loop, then saves per-case
    metrics and real primary prediction/training figures.  A compatible completed
    result is reused.  ``queue_worker`` calls it sequentially on one GPU.
    """
    output_root = Path(output_root_text)
    output_dir = output_root / "methods" / model_type / f"seed_{training_seed}"
    configure_logging(output_dir / "train.log")
    result_path = output_dir / "generalization_metrics.json"
    job_config = copy.deepcopy(dict(resolved_config))
    job_config["randomness"]["training_seed"] = int(training_seed)
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        checkpoint = torch.load(
            output_dir / "best_model.pt", map_location="cpu", weights_only=False
        )
        saved_science = checkpoint.get("resolved_config", {})
        if (
            result.get("model_type") != model_type
            or int(result.get("training_seed", -1)) != training_seed
            or tuple(result.get("split_order", ())) != COMBINATION_SPLIT_NAMES
            or checkpoint.get("model_type") != model_type
            or int(checkpoint.get("training_seed", -1)) != training_seed
            or any(
                saved_science.get(key) != job_config.get(key)
                for key in ("data", "equation", "operator", "combination_generalization")
            )
        ):
            raise RuntimeError(f"incompatible completed combination task: {output_dir}")
        LOGGER.info("Reusing completed combination task: %s", output_dir)
        return result

    device = torch.device(device_text)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    training_result = train_operator(
        model_type, splits, job_config, output_dir, device, training_seed, smoke
    )
    best_model, _ = load_operator_checkpoint(output_dir / "best_model.pt", device)
    split_metrics, per_case_rows = evaluate_named_splits(
        best_model,
        named_splits,
        device,
        int(job_config["operator"]["evaluation_batch_size"]),
        model_type,
        training_seed,
    )
    save_csv(output_dir / "generalization_per_case.csv", per_case_rows)
    result = {
        "model_type": model_type,
        "method": MODEL_DISPLAY_NAMES[model_type],
        "training_seed": int(training_seed),
        "parameter_count": int(training_result["parameter_count"]),
        "best_iteration": int(training_result["best_iteration"]),
        "best_validation_mse": float(training_result["best_validation_mse"]),
        "training_runtime_seconds": float(training_result["training_runtime_seconds"]),
        "split_order": list(COMBINATION_SPLIT_NAMES),
        "primary_split": "unseen_history_unseen_parameter",
        "primary_method": str(job_config["primary_method"]),
        "split_metrics": split_metrics,
    }
    save_json(result_path, result)
    plot_method_forward(
        named_splits["unseen_history_unseen_parameter"],
        np.load(output_dir / "test_predictions.npy", allow_pickle=False),
        model_type,
        output_dir,
        job_config["reporting"],
    )
    plot_training_history(
        output_dir / "training_history.csv",
        model_type,
        output_dir,
        job_config["reporting"],
    )
    return result


def queue_worker(
    device_text: str,
    jobs: list[tuple[str, int]],
    output_root_text: str,
    dataset_dir_text: str,
    resolved_config: Mapping[str, Any],
    smoke: bool,
) -> list[dict[str, Any]]:
    """Execute one long-lived round-robin queue with a single dataset load.

    ``device_text`` owns one worker, ``jobs`` contains method--seed pairs, and the
    remaining inputs locate common data/config.  Returns results in queue order.  For
    twenty jobs on eight GPUs a queue has two or three sequential jobs.  It frees
    Python/CUDA caches after each job, never runs two jobs simultaneously on one GPU,
    and is submitted by ``run_stage``.
    """
    signature = combination_dataset_signature(
        resolved_config,
        int(resolved_config["randomness"]["data_seed"]),
        smoke,
    )
    splits, named_splits, _ = load_combination_dataset(
        Path(dataset_dir_text), signature
    )
    results: list[dict[str, Any]] = []
    for model_type, training_seed in jobs:
        results.append(operator_job(
            model_type,
            training_seed,
            output_root_text,
            resolved_config,
            splits,
            named_splits,
            device_text,
            smoke,
        ))
        gc.collect()
        if device_text.startswith("cuda"):
            torch.cuda.empty_cache()
    return results


def _read_history(path: Path) -> list[dict[str, str]]:
    """Read one real training-history CSV for comparison plotting.

    ``path`` points to a method--seed ``training_history.csv``.  Returns a list of
    string dictionaries, e.g. 80 formal validation records.  It only reads disk and
    raises if empty.  ``aggregate_results`` uses the first seed for a compact curve
    comparison without selecting any model by test performance.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty training history: {path}")
    return rows


def aggregate_results(
    stage_dir: Path,
    methods: list[str],
    training_seeds: list[int],
    named_splits: Mapping[str, Any],
    reporting_config: Mapping[str, Any],
    smoke: bool,
) -> dict[str, Any]:
    """Aggregate the complete method--seed--split grid without pseudoreplication.

    ``stage_dir`` contains completed tasks, ``methods`` and ``training_seeds`` define
    the expected grid, ``named_splits`` supplies real profiles, and reporting/smoke
    control figures.  Returns a summary whose uncertainty is computed across seeds,
    not across test cases.  It writes raw case/seed CSVs, mean±sample-SD JSON, paired
    primary comparisons and three PNG/SVG figures.  Missing or non-finite rows raise.
    ``run_stage`` calls it only after all workers return.
    """
    seed_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    result_lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for model_type in methods:
        for seed in training_seeds:
            method_dir = stage_dir / "methods" / model_type / f"seed_{seed}"
            result = json.loads(
                (method_dir / "generalization_metrics.json").read_text(encoding="utf-8")
            )
            result_lookup[(model_type, seed)] = result
            with (method_dir / "generalization_per_case.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                case_rows.extend(dict(row) for row in csv.DictReader(handle))
            for split_name in COMBINATION_SPLIT_NAMES:
                metrics = result["split_metrics"][split_name]
                numeric = [metrics["mse"], metrics["relative_l2_mean"]]
                if not np.isfinite(numeric).all():
                    raise RuntimeError(f"non-finite metrics for {model_type}/{seed}/{split_name}")
                seed_rows.append({
                    "model_type": model_type,
                    "method": MODEL_DISPLAY_NAMES[model_type],
                    "training_seed": int(seed),
                    "split": split_name,
                    "parameter_count": int(result["parameter_count"]),
                    "best_iteration": int(result["best_iteration"]),
                    "best_validation_mse": float(result["best_validation_mse"]),
                    "trajectory_mse": float(metrics["mse"]),
                    "trajectory_relative_l2": float(metrics["relative_l2_mean"]),
                    "trajectory_relative_l2_median": float(metrics["relative_l2_median"]),
                    "training_runtime_seconds": float(result["training_runtime_seconds"]),
                })
    expected_rows = len(methods) * len(training_seeds) * len(COMBINATION_SPLIT_NAMES)
    if len(seed_rows) != expected_rows:
        raise RuntimeError("incomplete method--seed--split grid")
    expected_case_rows = len(methods) * len(training_seeds) * sum(
        int(named_splits[split].parameters.shape[0])
        for split in COMBINATION_SPLIT_NAMES
    )
    if len(case_rows) != expected_case_rows:
        raise RuntimeError(
            f"incomplete per-case grid: {len(case_rows)} != {expected_case_rows}"
        )
    comparison_dir = stage_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    save_csv(comparison_dir / "generalization_seed_means.csv", seed_rows)
    save_csv(comparison_dir / "generalization_per_case_per_seed.csv", case_rows)

    across: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for model_type in methods:
        across[model_type] = {}
        for split_name in COMBINATION_SPLIT_NAMES:
            selected = [
                row for row in seed_rows
                if row["model_type"] == model_type and row["split"] == split_name
            ]
            across[model_type][split_name] = {}
            for metric in ("trajectory_mse", "trajectory_relative_l2"):
                values = np.asarray([float(row[metric]) for row in selected])
                across[model_type][split_name][metric] = {
                    "mean": float(values.mean()),
                    "sample_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                    "n": int(values.size),
                }
    save_json(comparison_dir / "generalization_across_seeds.json", across)

    primary_split = "unseen_history_unseen_parameter"
    primary_method = str(next(iter(result_lookup.values()))["primary_method"])
    if primary_method not in methods:
        primary_method = "separate_parameter_shared" if "separate_parameter_shared" in methods else methods[0]
    primary_values = {
        seed: float(
            result_lookup[(primary_method, seed)]["split_metrics"][primary_split][
                "relative_l2_mean"
            ]
        )
        for seed in training_seeds
    }
    paired_rows: list[dict[str, Any]] = []
    for baseline in methods:
        if baseline == primary_method:
            continue
        differences = np.asarray([
            float(
                result_lookup[(baseline, seed)]["split_metrics"][primary_split][
                    "relative_l2_mean"
                ]
            ) - primary_values[seed]
            for seed in training_seeds
        ])
        paired_rows.append({
            "primary_method": primary_method,
            "baseline_method": baseline,
            "split": primary_split,
            "seed_count": len(training_seeds),
            "mean_baseline_minus_primary": float(differences.mean()),
            "sample_std_difference": float(differences.std(ddof=1)) if differences.size > 1 else 0.0,
            "primary_wins": int(np.sum(differences > 0.0)),
        })
    if paired_rows:
        save_csv(comparison_dir / "paired_primary_comparisons.csv", paired_rows)

    colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]
    positions = np.arange(len(COMBINATION_SPLIT_NAMES), dtype=np.float64)
    width = 0.8 / len(methods)
    figure, axis = plt.subplots(figsize=(10.5, 4.7))
    for index, model_type in enumerate(methods):
        means = [
            across[model_type][split]["trajectory_relative_l2"]["mean"]
            for split in COMBINATION_SPLIT_NAMES
        ]
        standard = [
            across[model_type][split]["trajectory_relative_l2"]["sample_std"]
            for split in COMBINATION_SPLIT_NAMES
        ]
        offset = (index - (len(methods) - 1) / 2.0) * width
        axis.bar(
            positions + offset, means, width=width, yerr=standard, capsize=2,
            color=colors[index], label=MODEL_DISPLAY_NAMES[model_type],
        )
    axis.set_yscale("log")
    axis.set_ylabel("Trajectory relative L2 (mean $\\pm$ SD)")
    axis.set_xticks(positions, ["SH–SP", "SH–UP", "UH–SP", "UH–UP"])
    axis.set_title("Combination generalization")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure_paths = save_figure(
        figure, comparison_dir / "figures" / "generalization_errors", reporting_config
    )
    plt.close(figure)

    first_seed = training_seeds[0]
    case_index = min(
        int(reporting_config["representative_forward_case"]),
        named_splits[primary_split].solutions.shape[0] - 1,
    )
    truth = named_splits[primary_split].solutions[case_index]
    times = named_splits[primary_split].output_times
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    for component, axis in enumerate(axes):
        axis.plot(times, truth[:, component], color="black", linewidth=1.7, label="Ground truth")
        for index, model_type in enumerate(methods):
            prediction = np.load(
                stage_dir / "methods" / model_type / f"seed_{first_seed}" / "test_predictions.npy",
                allow_pickle=False,
            )[case_index, :, component]
            axis.plot(
                times, prediction, linewidth=1.1, color=colors[index],
                label=MODEL_DISPLAY_NAMES[model_type],
            )
        axis.set_xlabel("Time")
        axis.set_ylabel("Population density")
        axis.set_title(f"Species {component + 1}: UH–UP")
    axes[1].legend(fontsize=7)
    figure.tight_layout()
    figure_paths.extend(save_figure(
        figure, comparison_dir / "figures" / "primary_split_profiles", reporting_config
    ))
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    for index, model_type in enumerate(methods):
        rows = _read_history(
            stage_dir / "methods" / model_type / f"seed_{first_seed}" / "training_history.csv"
        )
        iterations = np.asarray([int(row["iteration"]) for row in rows])
        axes[0].semilogy(
            iterations, [float(row["train_total_loss"]) for row in rows],
            color=colors[index], label=MODEL_DISPLAY_NAMES[model_type],
        )
        axes[1].semilogy(
            iterations, [float(row["validation_mse"]) for row in rows],
            color=colors[index], label=MODEL_DISPLAY_NAMES[model_type],
        )
    axes[0].set_title("Training total loss")
    axes[1].set_title("Validation MSE")
    for axis in axes:
        axis.set_xlabel("Training iteration")
        axis.set_ylabel("Loss")
    axes[1].legend(fontsize=7)
    figure.tight_layout()
    figure_paths.extend(save_figure(
        figure, comparison_dir / "figures" / "training_histories", reporting_config
    ))
    plt.close(figure)

    best_method = min(
        methods,
        key=lambda method: across[method][primary_split]["trajectory_relative_l2"]["mean"],
    )
    summary = {
        "smoke": bool(smoke),
        "protocol": "history_parameter_combinatorial_holdout_v1",
        "methods": methods,
        "method_count": len(methods),
        "training_seeds": training_seeds,
        "training_seed_count": len(training_seeds),
        "task_count": len(methods) * len(training_seeds),
        "split_order": list(COMBINATION_SPLIT_NAMES),
        "primary_split": primary_split,
        "case_row_count": len(case_rows),
        "best_on_primary_split": {
            "method_key": best_method,
            "method": MODEL_DISPLAY_NAMES[best_method],
            "mean_trajectory_relative_l2": across[best_method][primary_split][
                "trajectory_relative_l2"
            ]["mean"],
        },
        "across_training_seeds": across,
        "figure_paths": [str(path) for path in figure_paths],
    }
    save_json(comparison_dir / "comparison_summary.json", summary)
    return summary


def run_stage(
    stage_dir: Path,
    base_config: Mapping[str, Any],
    methods: list[str],
    training_seeds: list[int],
    devices: list[str],
    processes: int,
    cpus: int,
    smoke: bool,
) -> dict[str, Any]:
    """Generate shared data, run method--seed queues and aggregate four holdouts.

    ``stage_dir`` is smoke/full root; remaining inputs define selected science and
    resources.  Returns the complete comparison summary.  It generates one shared
    leakage-audited dataset, distributes jobs round-robin with one process per GPU,
    then aggregates only after every job succeeds.  ``main`` runs smoke first and
    formal training only after smoke passes.
    """
    resolved = resolve_combination_stage(base_config, smoke)
    resolved["runtime"]["stage_training_seeds"] = list(training_seeds)
    stage_dir.mkdir(parents=True, exist_ok=True)
    save_json(stage_dir / "resolved_config.json", resolved)
    dataset_dir = stage_dir / "dataset"
    _, named_splits, manifest = generate_or_load_combination_dataset(
        dataset_dir,
        resolved,
        int(resolved["randomness"]["data_seed"]),
        max(1, cpus),
        smoke,
    )
    save_json(stage_dir / "dataset_manifest_copy.json", manifest)
    jobs = [(method, int(seed)) for seed in training_seeds for method in methods]
    worker_count = min(processes, len(devices), len(jobs))
    queues = build_round_robin_queues(jobs, worker_count)
    LOGGER.info("Combination round-robin queues: %s", queues)
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
        futures = [
            executor.submit(
                queue_worker,
                devices[index],
                queue,
                str(stage_dir),
                str(dataset_dir),
                resolved,
                smoke,
            )
            for index, queue in enumerate(queues)
        ]
        for future in as_completed(futures):
            future.result()
    summary = aggregate_results(
        stage_dir,
        methods,
        training_seeds,
        named_splits,
        resolved["reporting"],
        smoke,
    )
    save_json(stage_dir / "stage_status.json", {"success": True, "summary": summary})
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Create the sole CLI parser for combination-generalization execution.

    The returned parser defines config/output, logical devices, process/CPU budgets,
    optional one-seed/method subset, resume, smoke, W&B and ntfy flags.  For example
    ``--seed 20261001`` schedules four jobs.  It has no side effect and ``main`` calls
    ``parse_args`` on it.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "combination_generalization.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--devices", type=str, default="0")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None, help="Run one configured training seed")
    parser.add_argument("--methods", type=str, default=None, help="Comma-separated registered method keys")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--wandb-mode", choices=("offline", "disabled"), default="disabled")
    parser.add_argument("--no-ntfy", action="store_true")
    return parser


def main() -> int:
    """Execute smoke then the selected five-seed combination experiment.

    Command-line inputs determine resources and output root.  Returns shell status 0
    only after data audit, all requested jobs, four-split evaluation and aggregation
    succeed.  It saves config/environment/status/runtime and sends best-effort ntfy
    notifications.  ``run_combination_nohup.sh`` is the normal caller.
    """
    args = build_parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir / "pipeline.log")
    allowed_existing = {"nohup.log", "pipeline.pid", "pipeline.log"}
    existing = {path.name for path in output_dir.iterdir()}
    if existing - allowed_existing and not args.resume:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    if args.resume and not existing:
        raise FileNotFoundError("--resume requires an existing output directory")
    started = time.perf_counter()
    config = load_config(args.config.expanduser().resolve())
    validate_combination_config(config)
    methods = parse_method_selection(args.methods)
    configured_seeds = [int(seed) for seed in config["randomness"]["training_seeds"]]
    if args.seed is not None and int(args.seed) not in configured_seeds:
        raise ValueError(f"--seed must be one of {configured_seeds}")
    selected_seeds = [int(args.seed)] if args.seed is not None else configured_seeds
    config["wandb"]["mode"] = args.wandb_mode
    config["wandb"]["enabled"] = args.wandb_mode != "disabled"
    devices = parse_devices(args.devices, args.processes)
    if args.cpus < 1 or args.cpus > (os.cpu_count() or 1):
        raise ValueError(f"--cpus must be within available logical CPUs: {os.cpu_count()}")
    config["runtime"] = {
        "output_dir": str(output_dir),
        "logical_devices": devices,
        "physical_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "processes": int(args.processes),
        "cpus": int(args.cpus),
        "configured_training_seeds": configured_seeds,
        "selected_training_seeds": selected_seeds,
        "selected_methods": methods,
        "forward_only": True,
    }
    save_json(output_dir / "config.json", config)
    save_json(output_dir / "environment.json", {
        "hostname": socket.gethostname(),
        "python": sys.version,
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    })
    topic = None if args.no_ntfy or not config["ntfy"].get("enabled", True) else str(config["ntfy"]["topic"])
    status: dict[str, Any] = {"success": False, "stage": "starting"}
    save_json(output_dir / "pipeline_status.json", status)
    try:
        LOGGER.info("Methods: %s", methods)
        LOGGER.info("Selected seeds: %s", selected_seeds)
        LOGGER.info("Formal task count: %d", len(methods) * len(selected_seeds))
        status["stage"] = "smoke"
        save_json(output_dir / "pipeline_status.json", status)
        run_stage(
            output_dir / "smoke", config, methods, [selected_seeds[0]], devices,
            args.processes, min(args.cpus, 8), True,
        )
        notify(topic, "Variable-delay combination smoke succeeded", str(output_dir))
        if args.smoke_only:
            runtime = time.perf_counter() - started
            save_json(output_dir / "pipeline_status.json", {
                "success": True, "stage": "smoke_complete", "runtime_seconds": runtime,
            })
            return 0
        status["stage"] = "full"
        save_json(output_dir / "pipeline_status.json", status)
        notify(topic, "Variable-delay combination experiment started", str(output_dir))
        summary = run_stage(
            output_dir / "full", config, methods, selected_seeds, devices,
            args.processes, args.cpus, False,
        )
        runtime = time.perf_counter() - started
        status = {
            "success": True, "stage": "complete", "runtime_seconds": runtime,
            "summary": summary,
        }
        save_json(output_dir / "pipeline_status.json", status)
        save_json(output_dir / "runtime.json", {"total_runtime_seconds": runtime})
        save_json(output_dir / "COMBINATION_GENERALIZATION_COMPLETED.json", status)
        notify(
            topic,
            "Variable-delay combination experiment succeeded",
            f"Best UH-UP: {summary['best_on_primary_split']['method_key']}\nRuntime: {runtime:.1f} s",
        )
        return 0
    except Exception as error:
        status = {
            "success": False,
            "stage": status.get("stage"),
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "runtime_seconds": time.perf_counter() - started,
        }
        save_json(output_dir / "pipeline_status.json", status)
        LOGGER.exception("Combination-generalization experiment failed")
        notify(topic, "Variable-delay combination experiment failed", repr(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
