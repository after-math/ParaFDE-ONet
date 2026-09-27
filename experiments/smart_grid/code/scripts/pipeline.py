#!/usr/bin/env python3
"""Run smoke validation and the complete ParaFDEONet forward experiment."""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import gc
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping
import urllib.request


PROJECT_DIR = Path(__file__).resolve().parents[2]
CODE_DIR = PROJECT_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import numpy as np
import torch

from data import dataset_signature, generate_or_load_dataset, load_dataset
from equation import PARAMETER_DIM, STATE_DIM, SmartGridConfig
from model import MODEL_DISPLAY_NAME, MODEL_TYPE
from reporting import aggregate_results, plot_forward_prediction, plot_training_history
from training import save_json, train_operator


LOGGER = logging.getLogger("smart_grid_forward_pipeline")


def configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter(
        "%(asctime)s | %(processName)s | %(levelname)s | %(message)s"
    )
    file_handler = logging.FileHandler(path, encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)


def load_config(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    with path.open("r", encoding="utf-8") as handle:
        values = json.load(handle, parse_constant=reject_constant)
    if not isinstance(values, dict):
        raise ValueError("the experiment config must be a JSON object")
    return values


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("methods") != [MODEL_TYPE]:
        raise ValueError("methods must contain only parafdeonet")
    if config.get("method_display_names") != {MODEL_TYPE: MODEL_DISPLAY_NAME}:
        raise ValueError("method display name does not match the model implementation")
    seeds = config["randomness"]["training_seeds"]
    if (
        not isinstance(seeds, list)
        or len(seeds) != 5
        or len(set(seeds)) != 5
        or any(not isinstance(seed, int) or seed < 0 for seed in seeds)
    ):
        raise ValueError("exactly five unique nonnegative training seeds are required")
    equation = SmartGridConfig.from_mapping(config["equation"])
    if config["equation"]["state_order"] != [
        "theta_1", "theta_2", "theta_3", "theta_4",
        "omega_1", "omega_2", "omega_3", "omega_4",
    ]:
        raise ValueError("the state order must match the eight-state implementation")
    if config["equation"]["parameter_order"] != [
        "coupling_K", "response_gamma", "damping_alpha"
    ]:
        raise ValueError("the parameter order must match the implementation")
    if not np.isclose(equation.fixed_delay, equation.maximum_history):
        raise ValueError("this experiment requires history length equal to the fixed delay")

    data = config["data"]
    integer_keys = (
        "history_count", "parameter_count", "pairs_per_history",
        "validation_count", "test_count", "history_sensors", "output_points",
        "solver_chunk_size",
    )
    if any(int(data[key]) < 1 for key in integer_keys):
        raise ValueError("all data counts must be positive")
    if int(data["pairs_per_history"]) > int(data["parameter_count"]):
        raise ValueError("pairs_per_history exceeds the parameter pool")
    if int(data["history_sensors"]) < 3:
        raise ValueError("at least three history sensors are required")
    if float(data["horizon"]) <= 0.0 or float(data["internal_step"]) <= 0.0:
        raise ValueError("horizon and internal step must be positive")
    steps = float(data["horizon"]) / float(data["internal_step"])
    if not np.isclose(steps, round(steps), atol=1e-10):
        raise ValueError("the internal step must divide the horizon")
    finite_steps = data["sensitivity_finite_difference_steps"]
    if len(finite_steps) != PARAMETER_DIM or any(float(value) <= 0.0 for value in finite_steps):
        raise ValueError("three positive finite-difference steps are required")
    if data.get("normalize_parameters") is not True:
        raise ValueError("parameter normalization must remain enabled")

    operator = config["operator"]
    required_architecture = {
        "architecture": MODEL_TYPE,
        "history_branch_count": STATE_DIM,
        "history_branch_output": "8_times_latent_dim",
        "parameter_feature_mode": "shared_latent_dim",
        "history_fusion": "identity_offset_hadamard_product",
        "parameter_fusion": "identity_offset_broadcast_hadamard_product",
        "physics_pairing_mode": "online_cartesian",
    }
    for key, expected in required_architecture.items():
        if operator.get(key) != expected:
            raise ValueError(f"operator.{key} must be {expected!r}")
    if operator["parameter_bounds"] != config["equation"]["parameter_bounds"]:
        raise ValueError("operator and equation parameter bounds must be identical")
    positive_operator_keys = (
        "latent_dim", "history_width", "history_depth", "parameter_width",
        "parameter_depth", "trunk_width", "trunk_depth", "iterations",
        "supervised_pair_batch", "physics_pool_batch", "residual_points",
        "sensitivity_points", "validation_every", "evaluation_batch_size",
        "learning_rate", "minimum_learning_rate", "gradient_clip",
    )
    if any(float(operator[key]) <= 0.0 for key in positive_operator_keys):
        raise ValueError("operator sizes, schedules and batch sizes must be positive")
    if int(operator["sensitivity_points"]) > int(data["output_points"]):
        raise ValueError("sensitivity_points exceeds the output grid")
    if int(operator["data_pretrain_iterations"]) > int(operator["iterations"]):
        raise ValueError("data pretraining exceeds total iterations")
    decay_end = int(operator.get("learning_rate_decay_iterations", operator["iterations"]))
    if not int(operator["warmup_iterations"]) < decay_end <= int(operator["iterations"]):
        raise ValueError("learning-rate decay endpoint must lie after warmup and within training")
    if any(
        float(operator[key]) < 0.0
        for key in (
            "data_weight", "initial_weight", "coupling_sensitivity_weight",
            "response_sensitivity_weight", "damping_sensitivity_weight",
            "physics_weight", "weight_decay",
        )
    ):
        raise ValueError("loss weights and weight decay must be nonnegative")
    if int(config["expected_formal_parameter_counts"][MODEL_TYPE]) != 75_223_048:
        raise ValueError("the formal ParaFDEONet parameter count has drifted")

    formats = {str(item).lower() for item in config["reporting"]["figure_formats"]}
    if not formats or not formats <= {"png", "jpg", "jpeg", "svg"}:
        raise ValueError("figure formats are limited to PNG, JPG and SVG")
    if bool(config["reporting"].get("report_parameter_jacobian_metrics", False)):
        if int(config["reporting"]["parameter_jacobian_evaluation_cases"]) < 1:
            raise ValueError("parameter Jacobian case count must be positive")
        if int(config["reporting"]["parameter_jacobian_evaluation_points"]) < 1:
            raise ValueError("parameter Jacobian point count must be positive")
    if float(config["smoke"]["horizon"]) <= equation.fixed_delay:
        raise ValueError("smoke horizon must exceed the fixed delay")


def resolve_stage_config(config: Mapping[str, Any], smoke: bool) -> dict[str, Any]:
    resolved = copy.deepcopy(dict(config))
    if not smoke:
        return resolved
    smoke_config = resolved["smoke"]
    data = resolved["data"]
    for key in (
        "history_count", "parameter_count", "pairs_per_history", "validation_count",
        "test_count", "history_sensors", "horizon", "output_points", "internal_step",
    ):
        data[key] = smoke_config[key]
    data["solver_chunk_size"] = min(int(data["solver_chunk_size"]), 8)
    operator = resolved["operator"]
    operator.update(
        {
            "iterations": int(smoke_config["operator_iterations"]),
            "data_pretrain_iterations": 1,
            "physics_ramp_iterations": 1,
            "latent_dim": int(smoke_config["operator_latent_dim"]),
            "history_width": int(smoke_config["operator_width"]),
            "history_depth": int(smoke_config["operator_depth"]),
            "parameter_width": int(smoke_config["operator_width"]),
            "parameter_depth": int(smoke_config["operator_depth"]),
            "trunk_width": int(smoke_config["operator_width"]),
            "trunk_depth": int(smoke_config["operator_depth"]),
            "fourier_modes": int(smoke_config["fourier_modes"]),
            "supervised_pair_batch": int(smoke_config["supervised_pair_batch"]),
            "physics_pool_batch": int(smoke_config["physics_pool_batch"]),
            "residual_points": int(smoke_config["residual_points"]),
            "sensitivity_points": int(smoke_config["sensitivity_points"]),
            "steps_per_epoch": 1,
            "validation_every": 1,
            "evaluation_batch_size": 4,
            "warmup_iterations": 1,
            "learning_rate_decay_iterations": int(smoke_config["operator_iterations"]),
        }
    )
    resolved["reporting"]["representative_forward_case"] = 0
    resolved["reporting"]["parameter_jacobian_evaluation_cases"] = min(
        4, int(data["test_count"])
    )
    resolved["reporting"]["parameter_jacobian_evaluation_points"] = min(
        7, int(data["output_points"])
    )
    resolved["wandb"]["enabled"] = False
    resolved["wandb"]["mode"] = "disabled"
    return resolved


def resume_science_compatible(
    previous: Mapping[str, Any], current: Mapping[str, Any]
) -> bool:
    """Allow only a longer iteration cap with the original LR decay endpoint."""
    for key in (
        "methods", "method_display_names", "randomness", "data", "equation",
        "quality_gates", "reporting", "expected_formal_parameter_counts", "smoke",
    ):
        if previous.get(key) != current.get(key):
            return False
    previous_operator = copy.deepcopy(dict(previous["operator"]))
    current_operator = copy.deepcopy(dict(current["operator"]))
    previous_total = int(previous_operator.pop("iterations"))
    current_total = int(current_operator.pop("iterations"))
    previous_decay = int(
        previous_operator.pop("learning_rate_decay_iterations", previous_total)
    )
    current_decay = int(
        current_operator.pop("learning_rate_decay_iterations", current_total)
    )
    return (
        current_total >= previous_total
        and current_decay == previous_decay
        and previous_operator == current_operator
    )


def completed_training_iteration(history_path: Path) -> int:
    if not history_path.exists():
        return 0
    with history_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return int(rows[-1]["iteration"]) if rows else 0


def parse_devices(value: str, processes: int) -> list[str]:
    if value.strip().lower() == "cpu":
        if processes != 1:
            raise ValueError("CPU mode requires --processes 1")
        return ["cpu"]
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if not parts or any(not item.isdigit() for item in parts):
        raise ValueError("--devices must be cpu or comma-separated logical GPU IDs")
    logical_ids = [int(item) for item in parts]
    if len(logical_ids) != len(set(logical_ids)):
        raise ValueError("duplicate logical GPU IDs are not allowed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch reports no CUDA")
    if any(index >= torch.cuda.device_count() for index in logical_ids):
        raise ValueError("a requested logical GPU does not exist")
    if processes < 1 or processes > len(logical_ids):
        raise ValueError("--processes must be between one and the number of devices")
    return [f"cuda:{index}" for index in logical_ids]


def select_training_seeds(
    configured_seeds: list[int], single_seed: int | None, seeds_text: str | None
) -> list[int]:
    """Select one or more configured seeds while preserving the requested order."""
    if single_seed is not None and seeds_text is not None:
        raise ValueError("--seed and --seeds are mutually exclusive")
    if single_seed is not None:
        selected = [int(single_seed)]
    elif seeds_text is not None:
        parts = [part.strip() for part in seeds_text.split(",") if part.strip()]
        if not parts or any(not part.isdigit() for part in parts):
            raise ValueError("--seeds must contain comma-separated nonnegative integers")
        selected = [int(part) for part in parts]
        if len(selected) != len(set(selected)):
            raise ValueError("--seeds contains duplicate values")
    else:
        selected = [int(seed) for seed in configured_seeds]
    unknown = [seed for seed in selected if seed not in configured_seeds]
    if unknown:
        raise ValueError(
            "selected training seeds are absent from the configured five: "
            + ",".join(str(seed) for seed in unknown)
        )
    return selected


def build_round_robin_queues(jobs: list[int], worker_count: int) -> list[list[int]]:
    if not jobs or worker_count < 1 or worker_count > len(jobs):
        raise ValueError("invalid jobs or worker count")
    return [jobs[index::worker_count] for index in range(worker_count)]


def operator_job(
    training_seed: int,
    stage_dir_text: str,
    dataset_dir_text: str,
    resolved_config: Mapping[str, Any],
    device_text: str,
    stage_name: str,
    resume: bool,
) -> dict[str, Any]:
    stage_dir = Path(stage_dir_text)
    output_dir = stage_dir / "methods" / MODEL_TYPE / f"seed_{training_seed}"
    configure_logging(output_dir / "train.log")
    metrics_path = output_dir / "metrics.json"
    train, validation, test = load_dataset(Path(dataset_dir_text))
    desired_iterations = int(resolved_config["operator"]["iterations"])
    completed_iterations = completed_training_iteration(output_dir / "training_history.csv")
    device = torch.device(device_text)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if metrics_path.exists():
        if not resume:
            raise FileExistsError(f"completed output already exists: {output_dir}")
        if completed_iterations > desired_iterations:
            raise RuntimeError("saved checkpoint exceeds the requested iteration cap")
        if completed_iterations < desired_iterations:
            LOGGER.info(
                "Extending seed %d from iteration %d to %d",
                training_seed, completed_iterations, desired_iterations,
            )
            metrics = train_operator(
                train, validation, test, resolved_config, output_dir, device,
                int(training_seed), stage_name, True,
            )
        else:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if int(metrics.get("training_seed", -1)) != int(training_seed):
                raise RuntimeError(f"training seed mismatch in {metrics_path}")
            LOGGER.info("Reusing complete seed %d", training_seed)
    else:
        metrics = train_operator(
            train,
            validation,
            test,
            resolved_config,
            output_dir,
            device,
            int(training_seed),
            stage_name,
            resume,
        )
    predictions = np.load(output_dir / "test_predictions.npy", allow_pickle=False)
    plot_forward_prediction(test, predictions, output_dir, resolved_config["reporting"])
    plot_training_history(
        output_dir / "training_history.csv", output_dir, resolved_config["reporting"]
    )
    return metrics


def queue_worker(
    device_text: str,
    seeds: list[int],
    stage_dir_text: str,
    dataset_dir_text: str,
    resolved_config: Mapping[str, Any],
    stage_name: str,
    resume: bool,
) -> list[dict[str, Any]]:
    results = []
    for seed in seeds:
        results.append(
            operator_job(
                seed,
                stage_dir_text,
                dataset_dir_text,
                resolved_config,
                device_text,
                stage_name,
                resume,
            )
        )
        gc.collect()
        if device_text.startswith("cuda"):
            torch.cuda.empty_cache()
    return results


def run_stage(
    stage_dir: Path,
    base_config: Mapping[str, Any],
    devices: list[str],
    process_count: int,
    cpu_count: int,
    smoke: bool,
    training_seeds: list[int],
    resume: bool,
) -> dict[str, Any]:
    stage_name = "smoke" if smoke else "full"
    resolved = resolve_stage_config(base_config, smoke)
    stage_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = stage_dir / "resolved_config.json"
    if resume and resolved_path.exists():
        previous = json.loads(resolved_path.read_text(encoding="utf-8"))
        if not resume_science_compatible(previous, resolved):
            raise RuntimeError(f"{stage_name} resolved config differs from the saved run")
    save_json(resolved_path, resolved)
    dataset_dir = stage_dir / "dataset"
    generate_or_load_dataset(dataset_dir, resolved, max(1, int(cpu_count)))
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("signature") != dataset_signature(resolved):
        raise RuntimeError("dataset signature changed after generation")

    worker_count = min(int(process_count), len(devices), len(training_seeds))
    queues = build_round_robin_queues(training_seeds, worker_count)
    LOGGER.info("%s seed queues: %s", stage_name, queues)
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
                stage_name,
                resume,
            )
            for index, queue in enumerate(queues)
        ]
        for future in as_completed(futures):
            future.result()
    summary = aggregate_results(stage_dir, training_seeds, resolved["reporting"])
    save_json(stage_dir / "stage_status.json", {"success": True, "summary": summary})
    return summary


def notify(topic: str | None, title: str, message: str) -> None:
    if not topic:
        return
    try:
        request = urllib.request.Request(
            topic,
            data=message.encode("utf-8"),
            headers={"Title": title},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read()
    except Exception:
        LOGGER.warning("notification failed; local experiment continues", exc_info=True)


def sync_offline_wandb(output_dir: Path) -> tuple[bool | None, str]:
    run_dirs = sorted(path for path in output_dir.rglob("offline-run-*") if path.is_dir())
    if not run_dirs:
        return None, "No offline W&B runs were found"
    failures = []
    for run_dir in run_dirs:
        try:
            completed = subprocess.run(
                ["wandb", "sync", str(run_dir)],
                text=True,
                capture_output=True,
                check=False,
                timeout=120,
            )
            if completed.returncode != 0:
                failures.append(f"{run_dir}: {completed.stderr.strip()}")
        except (OSError, subprocess.TimeoutExpired) as error:
            failures.append(f"{run_dir}: {error!r}")
    return (not failures), ("All W&B runs synchronized" if not failures else "\n".join(failures))


def format_final_metrics(summary: Mapping[str, Any], runtime_seconds: float) -> str:
    metrics = summary["metrics"]
    seed_count = int(metrics["seed_count"])
    normalized = float(metrics["test_normalized_mse_mean"])
    angle = float(metrics["test_relative_angle_difference_l2_mean_mean"])
    frequency = float(metrics["test_frequency_relative_l2_mean_mean"])
    return (
        f"{MODEL_DISPLAY_NAME}; seeds={seed_count}; normalized MSE={normalized:.4e}; "
        f"angle relative L2={angle:.4e}; frequency relative L2={frequency:.4e}; "
        f"runtime={runtime_seconds / 3600.0:.2f} h"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "experiment.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--devices", default="0", help="logical CUDA IDs, or cpu")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated subset of the five configured training seeds.",
    )
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb-mode", choices=("offline", "disabled"), default="disabled")
    parser.add_argument("--no-ntfy", action="store_true")
    return parser


def main() -> int:
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
    validate_config(config)
    configured_seeds = [int(seed) for seed in config["randomness"]["training_seeds"]]
    training_seeds = select_training_seeds(configured_seeds, args.seed, args.seeds)
    if args.cpus < 1 or args.cpus > (os.cpu_count() or 1):
        raise ValueError(f"--cpus must be between 1 and {os.cpu_count()}")
    devices = parse_devices(args.devices, args.processes)
    config["wandb"]["mode"] = args.wandb_mode
    config["wandb"]["enabled"] = args.wandb_mode != "disabled"
    config["runtime"] = {
        "output_dir": str(output_dir),
        "logical_devices": devices,
        "physical_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "processes": int(args.processes),
        "cpus": int(args.cpus),
        "active_training_seeds": training_seeds,
    }
    config_path = output_dir / "config.json"
    if args.resume and config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if not resume_science_compatible(previous, config):
            raise RuntimeError("--resume configuration differs from the original run")
        if previous.get("runtime", {}).get("active_training_seeds") != training_seeds:
            raise RuntimeError("--resume must use the original active seed selection")
    save_json(config_path, config)
    gpu_information = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpu_information.append(
                {
                    "logical_index": index,
                    "name": properties.name,
                    "total_memory_bytes": int(properties.total_memory),
                }
            )
    save_json(
        output_dir / "environment.json",
        {
            "hostname": socket.gethostname(),
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "logical_gpu_information": gpu_information,
        },
    )
    topic = None
    if (
        not args.no_ntfy
        and bool(config.get("ntfy", {}).get("enabled", False))
        and config.get("ntfy", {}).get("topic")
    ):
        topic = str(config["ntfy"]["topic"])
    status: dict[str, Any] = {"success": False, "stage": "starting"}
    save_json(output_dir / "pipeline_status.json", status)
    try:
        status["stage"] = "smoke"
        save_json(output_dir / "pipeline_status.json", status)
        run_stage(
            output_dir / "smoke",
            config,
            devices,
            args.processes,
            min(args.cpus, 8),
            True,
            [training_seeds[0]],
            args.resume,
        )
        if args.smoke_only:
            runtime = time.perf_counter() - started
            save_json(
                output_dir / "pipeline_status.json",
                {"success": True, "stage": "smoke_complete", "runtime_seconds": runtime},
            )
            return 0

        status["stage"] = "full"
        save_json(output_dir / "pipeline_status.json", status)
        summary = run_stage(
            output_dir / "full",
            config,
            devices,
            args.processes,
            args.cpus,
            False,
            training_seeds,
            args.resume,
        )
        sync_success: bool | None = None
        sync_message = "W&B synchronization disabled"
        if (
            config["wandb"].get("enabled", False)
            and config["wandb"].get("mode") == "offline"
            and config["wandb"].get("sync_after_success", False)
        ):
            sync_success, sync_message = sync_offline_wandb(output_dir)
        runtime = time.perf_counter() - started
        final_status = {
            "success": True,
            "stage": "complete",
            "runtime_seconds": runtime,
            "summary": summary,
            "wandb_sync_success": sync_success,
            "wandb_sync_message": sync_message,
        }
        save_json(output_dir / "pipeline_status.json", final_status)
        save_json(output_dir / "runtime.json", {"total_runtime_seconds": runtime})
        notify(topic, "Smart-grid ParaFDEONet complete", format_final_metrics(summary, runtime))
        LOGGER.info("Experiment complete in %.2f seconds", runtime)
        return 0
    except Exception as error:
        failure = {
            "success": False,
            "stage": status.get("stage"),
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "runtime_seconds": time.perf_counter() - started,
        }
        save_json(output_dir / "pipeline_status.json", failure)
        LOGGER.exception("Experiment failed")
        notify(topic, "Smart-grid ParaFDEONet failed", repr(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
