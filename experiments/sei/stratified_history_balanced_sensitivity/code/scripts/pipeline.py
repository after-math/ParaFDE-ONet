#!/usr/bin/env python3
"""Run selected delayed-SEI operators and seeds with stratified histories."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
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
import urllib.request


CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import torch

from data import dataset_signature, generate_or_load_dataset, load_dataset
from model import MODEL_DISPLAY_NAMES, MODEL_TYPES
from reporting import aggregate_results, plot_method_forward, plot_training_history
from training import save_json, train_operator
from wandb_tracker import sync_offline_runs


LOGGER = logging.getLogger("delayed_sei3d_pipeline")


def configure_logging(path: Path) -> None:
    """Configure terminal and file logging for the current main/worker process.

    ``path`` is the process-specific log file, e.g. root ``pipeline.log`` or a
    method ``train.log``.  Returns ``None`` and replaces only root logging handlers
    in the current process.  ``main`` and ``operator_job`` call it once each.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(processName)s | %(message)s")
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(logging.INFO)
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(stream)
    root.addHandler(file_handler)


def load_config(path: Path) -> dict[str, Any]:
    """Read one JSON experiment configuration into a mutable dictionary.

    ``path`` must exist and contain a JSON object.  The return is the parsed mapping;
    e.g. ``result['methods']`` has four keys.  It only reads disk.  ``main`` calls it
    before applying runtime overrides.
    """
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("config root must be a JSON object")
    return values


def validate_config(config: Mapping[str, Any]) -> None:
    """Reject scientific/configuration drift before allocating data or GPUs.

    ``config`` is the resolved formal mapping.  The function returns ``None`` when
    the four exact model keys, five seeds and common scientific settings are valid;
    otherwise it raises ``ValueError``.  It has no side effect.  ``main`` calls it
    immediately after loading.
    """
    methods = tuple(config.get("methods", ()))
    if methods != MODEL_TYPES:
        raise ValueError(f"methods must be exactly {MODEL_TYPES}")
    if config.get("method_display_names") != MODEL_DISPLAY_NAMES:
        raise ValueError("method_display_names must match the model registry")
    randomness = config["randomness"]
    seeds = randomness.get("training_seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) != 5
        or any(not isinstance(seed, int) or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("exactly five unique nonnegative training_seeds are required")
    data = config["data"]
    required_positive = (
        "history_count", "parameter_count", "pairs_per_history", "validation_count",
        "test_count", "history_sensors", "output_points", "internal_step",
    )
    if any(float(data[name]) <= 0.0 for name in required_positive):
        raise ValueError("data counts, sensors and time step must be positive")
    if int(data["pairs_per_history"]) > int(data["parameter_count"]):
        raise ValueError("pairs_per_history exceeds parameter pool")
    finite_steps = data.get("sensitivity_finite_difference_steps")
    if (
        not isinstance(finite_steps, list)
        or len(finite_steps) != 2
        or any(float(value) <= 0.0 for value in finite_steps)
    ):
        raise ValueError("sensitivity_finite_difference_steps must contain two positives")
    if len(data.get("history_level_ranges", [])) != 3:
        raise ValueError("history_level_ranges must contain S, E and I intervals")
    if len(data.get("history_sigmas", [])) != 3:
        raise ValueError("history_sigmas must contain S, E and I amplitudes")
    if len(data.get("history_bounds", [])) != 3:
        raise ValueError("history_bounds must contain S, E and I intervals")
    if float(data.get("history_total_upper", 0.0)) <= 0.0:
        raise ValueError("history_total_upper must be positive")
    if data.get("normalize_parameters") is not True:
        raise ValueError("this experiment requires parameter normalization")
    strata = data.get("history_state_3_strata")
    if not isinstance(strata, list) or len(strata) != 3:
        raise ValueError("history_state_3_strata must contain exactly three strata")
    if [item.get("name") for item in strata] != ["low", "medium", "high"]:
        raise ValueError("history strata must be ordered as low, medium and high")
    if int(data.get("history_strata_candidate_multiplier", 0)) < 1:
        raise ValueError("history_strata_candidate_multiplier must be positive")
    if int(data.get("history_strata_max_rounds", 0)) < 1:
        raise ValueError("history_strata_max_rounds must be positive")
    operator = config["operator"]
    if int(operator["iterations"]) < 1 or int(operator["validation_every"]) < 1:
        raise ValueError("training iterations and validation interval must be positive")
    if int(operator.get("sensitivity_points", 0)) < 1:
        raise ValueError("sensitivity_points must be positive")
    if float(operator.get("sensitivity_weight", -1.0)) < 0.0:
        raise ValueError("sensitivity_weight must be nonnegative")
    if float(operator.get("sensitivity_relative_epsilon", 0.0)) <= 0.0:
        raise ValueError("sensitivity_relative_epsilon must be positive")
    equation = config["equation"]
    equation_bounds = [
        equation.get("transmission_bounds"), equation.get("convexity_bounds")
    ]
    if operator.get("parameter_bounds") != equation_bounds:
        raise ValueError("operator parameter_bounds must equal the equation b/a bounds")
    if float(equation.get("delay", 0.0)) != float(equation.get("maximum_history", -1.0)):
        raise ValueError("fixed delay must equal maximum_history")
    if str(operator.get("physics_pairing_mode")) != "online_cartesian":
        raise ValueError("physics pairing must remain online_cartesian")
    formats = {str(item).lower() for item in config["reporting"]["figure_formats"]}
    if not formats or not formats <= {"png", "jpg", "jpeg", "svg"}:
        raise ValueError("only png/jpg/svg outputs are allowed")


def resolve_stage_config(config: Mapping[str, Any], smoke: bool) -> dict[str, Any]:
    """Return an isolated formal or path-identical smoke configuration.

    ``config`` is never modified; ``smoke=True`` replaces only dataset/network sizes,
    iteration count and batch sizes listed under ``smoke``.  The return still follows
    the exact production data/model/loss/training code path.  For example smoke has
    3 optimizer iterations and 21 history sensors.  ``run_stage`` calls it once.
    """
    resolved = copy.deepcopy(dict(config))
    if not smoke:
        return resolved
    smoke_config = resolved["smoke"]
    data = resolved["data"]
    for key in (
        "history_count", "parameter_count", "pairs_per_history",
        "validation_count", "test_count", "history_sensors", "horizon",
        "output_points", "internal_step",
    ):
        data[key] = smoke_config[key]
    data["solver_chunk_size"] = min(int(data["solver_chunk_size"]), 8)
    operator = resolved["operator"]
    operator.update({
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
    })
    resolved["reporting"]["representative_forward_case"] = 0
    resolved["wandb"]["run_name"] = None
    return resolved


def notify(topic: str | None, title: str, message: str) -> None:
    """Send a best-effort ntfy message without controlling experiment success.

    ``topic`` may be ``None`` to disable; ``title`` and ``message`` are UTF-8 text.
    Returns ``None``.  Network errors only produce a warning.  ``main`` calls it for
    smoke/full start, success and failure notifications.
    """
    if not topic:
        return
    try:
        request = urllib.request.Request(
            topic,
            data=message.encode("utf-8"),
            headers={"Title": title.encode("ascii", "ignore").decode("ascii") or "Experiment"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read()
        LOGGER.info("ntfy notification sent: %s", title)
    except Exception:
        LOGGER.warning("ntfy notification failed; experiment continues", exc_info=True)


def format_final_metric_notification(
    summary: Mapping[str, Any], runtime_seconds: float
) -> str:
    """Format four methods' final errors as mean plus sample standard deviation.

    ``summary`` is the dictionary returned by ``reporting.aggregate_results`` and
    ``runtime_seconds`` is full pipeline wall time. The return is a compact UTF-8
    message containing test MSE and mean trajectory relative-L2 for every method.
    For example a five-seed row is ``1.2e-6 ± 2.0e-7 (n=5)``; a selected one-seed
    run reports the value and explicitly says sample SD is unavailable instead of
    presenting zero as an estimated standard deviation. No files or network state
    change. ``main`` passes this output to ``notify`` after all results are saved.
    """
    methods = summary.get("methods")
    if not isinstance(methods, list) or not methods:
        raise ValueError("final summary must contain at least one selected method")
    lines = ["Final forward metrics"]
    for method in methods:
        seed_count = int(method["seed_count"])
        mse_mean = float(method["test_mse_mean"])
        relative_mean = float(method["test_relative_l2_mean_mean"])
        if seed_count > 1:
            mse_text = (
                f"{mse_mean:.4e} ± {float(method['test_mse_std']):.2e}"
            )
            relative_text = (
                f"{relative_mean:.4e} ± "
                f"{float(method['test_relative_l2_mean_std']):.2e}"
            )
        else:
            mse_text = f"{mse_mean:.4e} (SD unavailable)"
            relative_text = f"{relative_mean:.4e} (SD unavailable)"
        lines.extend(
            (
                str(method["method"]),
                f"  Test MSE: {mse_text}",
                f"  Relative L2: {relative_text}",
                f"  Seeds: n={seed_count}",
            )
        )
    lines.append(f"Total runtime: {runtime_seconds / 3600.0:.2f} h")
    return "\n".join(lines)


def build_round_robin_queues(jobs: list[Any], worker_count: int) -> list[list[Any]]:
    """Distribute jobs deterministically over persistent GPU workers.

    ``jobs`` is an ordered list of model keys or ``(model_key, seed)`` tuples and
    ``worker_count`` is the number of GPU workers.  The return contains exactly
    ``worker_count`` queues; for example four jobs and four workers produce four
    one-job queues.  The input list is not modified and no process or GPU state is
    created.  ``run_stage`` and the combination-generalization entry point call this
    helper before starting one long-lived process per logical GPU.
    """
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    if not jobs:
        raise ValueError("jobs must not be empty")
    if worker_count > len(jobs):
        raise ValueError("worker_count cannot exceed the number of jobs")
    return [jobs[index::worker_count] for index in range(worker_count)]


def operator_job(
    model_type: str,
    training_seed: int,
    stage_dir_text: str,
    dataset_dir_text: str,
    resolved_config: Mapping[str, Any],
    device_text: str,
    smoke: bool,
) -> dict[str, Any]:
    """Run one complete train/validation/test/plot job on one assigned device.

    Inputs identify method, stage/dataset directories, resolved stage config, logical
    device and smoke status.  Returns ``task_result``.  It reuses compatible complete
    results or resumes ``last_model.pt``; otherwise writes method checkpoints,
    metrics, predictions, figures and log.  ``queue_worker`` calls it sequentially.
    """
    stage_dir = Path(stage_dir_text)
    job_config = copy.deepcopy(dict(resolved_config))
    job_config["randomness"]["training_seed"] = int(training_seed)
    output_dir = stage_dir / "methods" / model_type / f"seed_{training_seed}"
    configure_logging(output_dir / "train.log")
    result_path = output_dir / "task_result.json"
    signature = dataset_signature(
        resolved_config, int(resolved_config["randomness"]["data_seed"]), smoke
    )
    splits = load_dataset(Path(dataset_dir_text), signature)
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        checkpoint = torch.load(
            output_dir / "best_model.pt", map_location="cpu", weights_only=False
        )
        saved_science = checkpoint.get("resolved_config", {})
        if (
            result.get("model_type") != model_type
            or int(result.get("training_seed", -1)) != training_seed
            or any(
                saved_science.get(key) != job_config.get(key)
                for key in ("data", "equation", "operator")
            )
        ):
            raise RuntimeError(f"complete task is incompatible with current config: {output_dir}")
        LOGGER.info("Reusing complete task: %s", output_dir)
    else:
        device = torch.device(device_text)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        result = train_operator(
            model_type, splits, job_config, output_dir, device, training_seed, smoke
        )
    predictions = __import__("numpy").load(output_dir / "test_predictions.npy", allow_pickle=False)
    plot_method_forward(
        splits["test"], predictions, model_type, output_dir, resolved_config["reporting"]
    )
    plot_training_history(
        output_dir / "training_history.csv", model_type, output_dir, resolved_config["reporting"]
    )
    return result


def queue_worker(
    device_text: str,
    jobs: list[tuple[str, int]],
    stage_dir_text: str,
    dataset_dir_text: str,
    resolved_config: Mapping[str, Any],
    smoke: bool,
) -> list[dict[str, Any]]:
    """Execute a round-robin method queue sequentially on one long-lived GPU worker.

    ``device_text`` owns one process and ``jobs`` is its nonempty method/seed queue.
    Remaining inputs are common job state. Returns one result per job in queue order. It
    frees model/Python/CUDA caches after each job.  ``run_stage`` submits one such
    worker per active device, preventing multiple simultaneous jobs on one GPU.
    """
    results: list[dict[str, Any]] = []
    for model_type, training_seed in jobs:
        results.append(operator_job(
            model_type, training_seed, stage_dir_text, dataset_dir_text,
            resolved_config, device_text, smoke
        ))
        gc.collect()
        if device_text.startswith("cuda"):
            torch.cuda.empty_cache()
    return results


def run_stage(
    root: Path,
    base_config: Mapping[str, Any],
    devices: list[str],
    processes: int,
    cpus: int,
    smoke: bool,
    training_seeds: list[int],
    selected_methods: list[str],
) -> dict[str, Any]:
    """Generate shared data, dispatch method/seed jobs and aggregate results.

    ``root`` is smoke/full directory, ``devices`` are logical CUDA IDs or CPU,
    ``processes`` bounds workers and ``cpus`` bounds reference solving.  Returns the
    validated four-method summary. It writes shared dataset, model outputs and
    comparison artifacts.  ``main`` runs smoke first and full only after smoke passes.
    """
    resolved = resolve_stage_config(base_config, smoke)
    root.mkdir(parents=True, exist_ok=True)
    save_json(root / "resolved_config.json", resolved)
    dataset_dir = root / "dataset"
    splits = generate_or_load_dataset(
        dataset_dir,
        resolved,
        int(resolved["randomness"]["data_seed"]),
        max(1, cpus),
        smoke,
    )
    del splits
    methods = list(selected_methods)
    jobs = [(method, int(seed)) for seed in training_seeds for method in methods]
    worker_count = min(processes, len(devices), len(jobs))
    if worker_count < 1:
        raise ValueError("at least one worker is required")
    queues = build_round_robin_queues(jobs, worker_count)
    LOGGER.info("Round-robin queues: %s", queues)
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
        futures = [
            executor.submit(
                queue_worker,
                devices[index],
                queue,
                str(root),
                str(dataset_dir),
                resolved,
                smoke,
            )
            for index, queue in enumerate(queues)
        ]
        for future in as_completed(futures):
            future.result()
    summary = aggregate_results(
        root, methods, training_seeds, resolved["reporting"]
    )
    save_json(root / "stage_status.json", {"success": True, "summary": summary})
    return summary


def parse_devices(value: str, processes: int) -> list[str]:
    """Validate logical devices supplied by the launcher or direct CPU use.

    ``value`` is ``cpu`` or comma-separated logical CUDA integers and ``processes``
    must not exceed their count.  Returns strings such as ``['cuda:0','cuda:1']``.
    It queries CUDA availability/count but changes no device.  ``main`` calls it.
    """
    if value.strip().lower() == "cpu":
        if processes != 1:
            raise ValueError("CPU mode requires --processes 1")
        return ["cpu"]
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if not parts or any(not item.isdigit() for item in parts):
        raise ValueError("--devices must be cpu or comma-separated logical GPU IDs")
    numbers = [int(item) for item in parts]
    if len(numbers) != len(set(numbers)):
        raise ValueError("duplicate logical GPU IDs are not allowed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch reports no CUDA")
    if any(item >= torch.cuda.device_count() for item in numbers):
        raise ValueError("requested logical GPU does not exist after CUDA remapping")
    if processes < 1 or processes > len(numbers):
        raise ValueError("--processes must be between 1 and the number of devices")
    return [f"cuda:{item}" for item in numbers]


def build_parser() -> argparse.ArgumentParser:
    """Create the sole command-line parser for this forward experiment.

    The returned parser defines config, output, logical devices, process/CPU budgets,
    one optional seed override, smoke-only, resume, W&B and ntfy controls.  For
    example ``parser.parse_args(['--devices','cpu'])`` selects local CPU smoke use.
    It has no external side effect.  ``main`` is the caller.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "experiment.json", help="JSON experiment config")
    parser.add_argument("--output-dir", type=Path, required=True, help="Unique experiment output directory")
    parser.add_argument("--devices", type=str, default="0", help="Logical CUDA IDs after remapping, or cpu")
    parser.add_argument("--processes", type=int, default=1, help="Maximum one-process-per-GPU workers")
    parser.add_argument("--cpus", type=int, default=1, help="Total CPU worker budget for reference solves")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Run only this seed; omit to run all five configured seeds",
    )
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="Comma-separated configured seeds; mutually exclusive with --seed",
    )
    parser.add_argument(
        "--method", type=str, default=None,
        help="Run one configured method; mutually exclusive with --methods",
    )
    parser.add_argument(
        "--methods", type=str, default=None,
        help="Comma-separated configured method keys; omit to run all four",
    )
    parser.add_argument("--smoke-only", action="store_true", help="Run only the full-path smoke stage")
    parser.add_argument("--resume", action="store_true", help="Resume an existing incomplete output directory")
    parser.add_argument(
        "--extend-resume",
        action="store_true",
        help=(
            "With --resume, allow the requested seed set to extend the seed set "
            "stored in the output directory; completed jobs are reused"
        ),
    )
    parser.add_argument("--wandb-mode", choices=("offline", "disabled"), default="disabled", help="Optional W&B mode")
    parser.add_argument("--no-ntfy", action="store_true", help="Disable ntfy notifications")
    return parser


def resolve_active_seeds(
    single_seed: int | None,
    seed_list: str | None,
    configured_seeds: list[int],
) -> list[int]:
    """Resolve one, several or all configured seeds without silent substitutions."""
    if single_seed is not None and seed_list is not None:
        raise ValueError("--seed and --seeds are mutually exclusive")
    if single_seed is not None:
        selected = [int(single_seed)]
    elif seed_list is not None:
        parts = [item.strip() for item in seed_list.split(",") if item.strip()]
        if not parts or any(not item.isdigit() for item in parts):
            raise ValueError("--seeds must be comma-separated nonnegative integers")
        selected = [int(item) for item in parts]
    else:
        selected = [int(seed) for seed in configured_seeds]
    if len(selected) != len(set(selected)):
        raise ValueError("selected training seeds must be unique")
    unknown = [seed for seed in selected if seed not in configured_seeds]
    if unknown:
        raise ValueError(
            f"selected seeds are absent from configs/experiment.json: {unknown}"
        )
    return selected


def resolve_active_methods(
    single_method: str | None,
    method_list: str | None,
    configured_methods: list[str],
) -> list[str]:
    """Resolve one, several or all configured method keys in requested order."""
    if single_method is not None and method_list is not None:
        raise ValueError("--method and --methods are mutually exclusive")
    if single_method is not None:
        selected = [single_method.strip()]
    elif method_list is not None:
        selected = [item.strip() for item in method_list.split(",") if item.strip()]
    else:
        selected = list(configured_methods)
    if not selected or any(not item for item in selected):
        raise ValueError("at least one method must be selected")
    if len(selected) != len(set(selected)):
        raise ValueError("selected methods must be unique")
    unknown = [method for method in selected if method not in configured_methods]
    if unknown:
        raise ValueError(f"unknown selected methods: {unknown}")
    return selected


def validate_resume_selection(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    training_seeds: list[int],
    selected_methods: list[str],
    extend_resume: bool,
) -> None:
    """Validate an ordinary resume or a seed-set extension without science drift.

    The science-defining configuration and active methods must be unchanged.  An
    ordinary resume also requires the identical seed list.  With ``extend_resume``,
    the new list may add configured seeds but may not remove any previously active
    seed.  The function has no side effect and raises ``RuntimeError`` on mismatch.
    """
    science_keys = ("methods", "randomness", "data", "equation", "operator")
    if any(previous.get(key) != current.get(key) for key in science_keys):
        raise RuntimeError("--resume config does not match the existing experiment")
    previous_runtime = previous.get("runtime", {})
    previous_seeds = previous_runtime.get("active_training_seeds")
    if extend_resume:
        if (
            not isinstance(previous_seeds, list)
            or not set(previous_seeds).issubset(set(training_seeds))
        ):
            raise RuntimeError(
                "--extend-resume requires the requested seeds to include every "
                "previously active seed"
            )
    elif previous_seeds != training_seeds:
        raise RuntimeError("--resume must use the same active training seeds")
    previous_methods = previous_runtime.get("active_methods")
    if previous_methods != selected_methods:
        raise RuntimeError("--resume must use the same active methods")


def main() -> int:
    """Execute the complete five-seed or selected-seed forward experiment.

    Command-line inputs select resources and an output root.  Returns shell status 0
    only after requested stages, final test, figures and summaries succeed.  It saves
    final config/environment/runtime/status and sends best-effort ntfy notifications.
    This is called by ``run_all_nohup.sh``.
    """
    args = build_parser().parse_args()
    if args.extend_resume and not args.resume:
        raise ValueError("--extend-resume requires --resume")
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
    training_seeds = resolve_active_seeds(
        args.seed,
        args.seeds,
        [int(seed) for seed in config["randomness"]["training_seeds"]],
    )
    selected_methods = resolve_active_methods(
        args.method,
        args.methods,
        [str(method) for method in config["methods"]],
    )
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
        "active_training_seeds": training_seeds,
        "active_methods": selected_methods,
        "forward_only": True,
        "parameter_normalization": True,
        "parameter_sensitivity_supervision": True,
    }
    saved_config_path = output_dir / "config.json"
    if args.resume and saved_config_path.exists():
        previous = json.loads(saved_config_path.read_text(encoding="utf-8"))
        validate_resume_selection(
            previous,
            config,
            training_seeds,
            selected_methods,
            bool(args.extend_resume),
        )
    save_json(output_dir / "config.json", config)
    gpu_information = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpu_information.append({
                "logical_index": index,
                "name": properties.name,
                "total_memory_bytes": int(properties.total_memory),
            })
    save_json(output_dir / "environment.json", {
        "hostname": socket.gethostname(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_gpu_information": gpu_information,
        "git_commit": None,
    })
    topic = None if args.no_ntfy or not config["ntfy"].get("enabled", True) else str(config["ntfy"]["topic"])
    status: dict[str, Any] = {"success": False, "stage": "starting"}
    save_json(output_dir / "pipeline_status.json", status)
    try:
        LOGGER.info("Experiment: %s", config["experiment_name"])
        LOGGER.info("Configured methods: %s", config["methods"])
        LOGGER.info("Active methods: %s", selected_methods)
        LOGGER.info("Active training seeds: %s", training_seeds)
        LOGGER.info("Devices: %s | processes: %d | CPUs: %d", devices, args.processes, args.cpus)
        LOGGER.info("Resolved config: %s", json.dumps(config, sort_keys=True))
        status["stage"] = "smoke"
        save_json(output_dir / "pipeline_status.json", status)
        run_stage(
            output_dir / "smoke", config, devices, args.processes,
            min(args.cpus, 8), True, training_seeds, selected_methods,
        )
        notify(topic, "Delayed SEI3D smoke test succeeded", str(output_dir))
        if args.smoke_only:
            status = {"success": True, "stage": "smoke_complete", "runtime_seconds": time.perf_counter() - started}
            save_json(output_dir / "pipeline_status.json", status)
            return 0
        notify(topic, "Delayed SEI3D full experiment started", str(output_dir))
        status["stage"] = "full"
        save_json(output_dir / "pipeline_status.json", status)
        summary = run_stage(
            output_dir / "full", config, devices, args.processes,
            args.cpus, False, training_seeds, selected_methods,
        )
        sync_success = None
        sync_message = "W&B sync disabled"
        if (
            bool(config["wandb"].get("enabled", False))
            and str(config["wandb"].get("mode")) == "offline"
            and bool(config["wandb"].get("sync_after_success", True))
        ):
            sync_success, sync_message = sync_offline_runs(output_dir, LOGGER)
            notify(
                topic,
                "Delayed SEI3D W&B sync succeeded" if sync_success
                else "Delayed SEI3D W&B sync failed",
                sync_message,
            )
        runtime = time.perf_counter() - started
        status = {
            "success": True,
            "stage": "complete",
            "runtime_seconds": runtime,
            "summary": summary,
            "wandb_sync_success": sync_success,
            "wandb_sync_message": sync_message,
        }
        save_json(output_dir / "pipeline_status.json", status)
        save_json(output_dir / "runtime.json", {"total_runtime_seconds": runtime})
        notify(
            topic,
            "Delayed SEI3D full experiment succeeded",
            format_final_metric_notification(summary, runtime),
        )
        LOGGER.info("Complete in %.2f seconds", runtime)
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
        LOGGER.exception("Experiment failed")
        notify(topic, "Delayed SEI3D experiment failed", repr(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
