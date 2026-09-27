#!/usr/bin/env python3
"""Paired High-75 Frozen-vs-LM experiment with strict warm online timing.

This additive entry combines the validated High-75 protocol, cached data-only
Frozen objective, adaptive projected-Adam stopping, persistent model reuse and
the conservative Direct-LM stopping rule.  Deployment initialization is timed
separately and excluded from all 40 Frozen online measurements.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
import platform
import time
from typing import Any, Mapping, Sequence
import weakref

import numpy as np
import torch

import pipeline_high75_persistent_cached_data_only as _persistent
import pipeline_high75_lm_early_stop as _lm


_pipeline = _persistent._pipeline
_cached = _persistent._cached
_BASE_VALIDATE = _persistent.validate_config
_BASE_FIXED_FEATURES = _cached._fixed_features
_BASE_OPERATOR_JOB = _pipeline.run_operator_job
_BASE_LM_JOB = _lm.run_lm_job_early_stop
_BASE_QUEUE_WORKER = _pipeline.queue_worker
_BASE_BUILD_QUEUES = _pipeline.build_round_robin_queues
_BASE_AGGREGATE = _persistent.aggregate_stage

_TRUNK_CACHE: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
_LAST_HISTORY_SECONDS: dict[str, float] = {}


def validate_config(config: Mapping[str, Any]) -> None:
    """Keep the final High-75 science settings and unify the paired protocol."""
    _BASE_VALIDATE(config)
    if not isinstance(config, dict):
        raise TypeError("the resolved experiment configuration must be mutable")
    lm = config["projected_lm"]
    lm["minimum_iterations"] = _lm.MINIMUM_ITERATIONS
    lm["relative_objective_tolerance"] = _lm.RELATIVE_OBJECTIVE_TOLERANCE
    lm["early_stopping_patience"] = _lm.EARLY_STOPPING_PATIENCE
    config["experiment_name"] = "delayed_sei3d_high75_strict_warm_paired"
    config["timing_protocol"] = {
        "persistent_model_per_gpu_worker": True,
        "all_workers_ready_barrier": True,
        "untimed_cuda_forward_backward_warmup": True,
        "untimed_fixed_observation_trunk_cache": True,
        "timed_history_branch_per_case": True,
        "timed_grid_screening_and_projected_adam": True,
        "deployment_initialization_reported_separately": True,
    }


def _fixed_features_strict_warm(
    model: _pipeline.Operator3D,
    history: torch.Tensor,
    observation_times: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache each case's History Branch while reusing the deployment Trunk."""
    model_ref = _cached._FEATURE_CACHE.get("model_ref")
    history_ref = _cached._FEATURE_CACHE.get("history_ref")
    times_ref = _cached._FEATURE_CACHE.get("times_ref")
    cache_matches = (
        model_ref is not None
        and model_ref() is model
        and history_ref is not None
        and history_ref() is history
        and times_ref is not None
        and times_ref() is observation_times
    )
    if not cache_matches:
        _pipeline.synchronize(history.device)
        started = time.perf_counter()
        with torch.no_grad():
            encoded = [
                1.0
                + branch(history[:, component, :]).reshape(
                    history.shape[0], _pipeline.STATE_DIM, model.latent_dim
                )
                for component, branch in enumerate(model.history_branches)
            ]
            history_features = encoded[0]
            for values in encoded[1:]:
                history_features = history_features * values
            history_features = history_features.detach()
        _pipeline.synchronize(history.device)
        _LAST_HISTORY_SECONDS[str(history.device)] = time.perf_counter() - started

        cached_trunk = _TRUNK_CACHE.get(str(history.device))
        if cached_trunk is None:
            raise RuntimeError("strict warm Trunk cache was not initialized")
        cached_times, trunk_features = cached_trunk
        if cached_times.shape != observation_times.shape or not torch.equal(
            cached_times, observation_times.detach().cpu()
        ):
            raise RuntimeError("formal observation times differ from warm Trunk cache")
        _cached._FEATURE_CACHE.clear()
        _cached._FEATURE_CACHE.update(
            model_ref=weakref.ref(model),
            history_ref=weakref.ref(history),
            times_ref=weakref.ref(observation_times),
            history_features=history_features,
            trunk_features=trunk_features,
        )
    return (
        _cached._FEATURE_CACHE["history_features"],
        _cached._FEATURE_CACHE["trunk_features"],
    )


def _initialize_worker(
    device_name: str,
    jobs: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    checkpoint_path: Path,
    archive_path: Path,
    stage_dir: Path,
) -> None:
    operator_jobs = [job for job in jobs if job["method"] == "operator_projected_grid"]
    if not operator_jobs:
        return
    _persistent._MODEL_CACHE.clear()
    _cached._FEATURE_CACHE.clear()
    _TRUNK_CACHE.clear()
    _LAST_HISTORY_SECONDS.clear()

    total_started = time.perf_counter()
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    load_started = time.perf_counter()
    model, checkpoint = _pipeline.load_operator_checkpoint(checkpoint_path, device)
    load_seconds = time.perf_counter() - load_started

    warm_job = operator_jobs[0]
    case = _pipeline.load_case(archive_path, int(warm_job["case_index"]))
    observed = _pipeline.condition_observations(
        case, _pipeline.condition_by_key(config, str(warm_job["condition"]))
    )
    history = torch.as_tensor(observed["histories"], dtype=torch.float32, device=device)
    times = torch.as_tensor(observed["times"], dtype=torch.float32, device=device)
    observations = torch.as_tensor(observed["values"], dtype=torch.float32, device=device)
    equation = _pipeline.DelayedSEIConfig.from_mapping(
        checkpoint["resolved_config"]["equation"]
    )

    warm_started = time.perf_counter()
    history_features, trunk_features = _BASE_FIXED_FEATURES(model, history, times)
    _TRUNK_CACHE[str(device)] = (times.detach().cpu().clone(), trunk_features.detach())
    with torch.no_grad():
        grid_probe = torch.zeros((256, 2), dtype=torch.float32, device=device)
        _cached.cached_operator_data_objective(
            model, grid_probe, history, times, observations, observed["states"],
            equation.parameter_bounds,
        )
    parameter_probe = torch.zeros(
        (10, 2), dtype=torch.float32, device=device, requires_grad=True
    )
    optimizer_probe = torch.optim.Adam([parameter_probe], lr=1.0e-3)
    objective, _ = _cached.cached_operator_data_objective(
        model, parameter_probe, history, times, observations, observed["states"],
        equation.parameter_bounds,
    )
    objective.sum().backward()
    optimizer_probe.step()
    _pipeline.synchronize(device)
    warm_seconds = time.perf_counter() - warm_started

    # The real first case must encode its History Branch inside its own timer.
    _cached._FEATURE_CACHE.clear()
    _LAST_HISTORY_SECONDS.clear()
    del grid_probe, parameter_probe, optimizer_probe, objective, history_features
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    deployment_dir = stage_dir / "deployment_initialization"
    deployment_dir.mkdir(parents=True, exist_ok=True)
    _pipeline.write_json(
        deployment_dir / f"worker_{os.getpid()}_{device_name.replace(':', '_')}.json",
        {
            "process_id": int(os.getpid()),
            "device": device_name,
            "accelerator_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else platform.processor() or platform.machine()
            ),
            "checkpoint_load_seconds": float(load_seconds),
            "cuda_and_operator_warmup_seconds": float(warm_seconds),
            "total_deployment_initialization_seconds": float(
                time.perf_counter() - total_started
            ),
            "fixed_observation_trunk_cached": True,
            "excluded_from_per_case_online_timing": True,
        },
    )


def run_operator_job_strict_warm(*args: Any, **kwargs: Any) -> dict[str, Any]:
    device_name = str(args[7] if len(args) > 7 else kwargs["device_name"])
    _LAST_HISTORY_SECONDS.pop(device_name, None)
    result = _BASE_OPERATOR_JOB(*args, **kwargs)
    result.update(
        {
            "model_cache_hit": 1,
            "model_residency_status": "warm",
            "checkpoint_load_seconds": 0.0,
            "observation_trunk_cache_seconds": 0.0,
            "history_branch_cache_seconds": float(
                _LAST_HISTORY_SECONDS.get(device_name, 0.0)
            ),
            "deployment_initialization_excluded": 1,
            "compute_device": device_name,
            "accelerator_name": (
                torch.cuda.get_device_name(torch.device(device_name))
                if torch.device(device_name).type == "cuda"
                else platform.processor() or platform.machine()
            ),
        }
    )
    output_dir = Path(args[8] if len(args) > 8 else kwargs["output_dir"])
    _pipeline.write_json(output_dir / "result.json", result)
    return result


def _cpu_model_name() -> str:
    path = Path("/proc/cpuinfo")
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def run_lm_job_strict_paired(*args: Any, **kwargs: Any) -> dict[str, Any]:
    result = _BASE_LM_JOB(*args, **kwargs)
    result.update({"compute_device": "cpu", "cpu_model": _cpu_model_name()})
    output_dir = Path(args[7] if len(args) > 7 else kwargs["output_dir"])
    _pipeline.write_json(output_dir / "result.json", result)
    return result


def queue_worker(
    device_name: str,
    jobs: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    checkpoint_path_text: str,
    checkpoint_identity: Mapping[str, Any],
    checkpoint_science: Mapping[str, Any],
    archive_path_text: str,
    stage_dir_text: str,
    smoke: bool,
    resume: bool,
) -> list[dict[str, Any]]:
    _initialize_worker(
        device_name, jobs, config, Path(checkpoint_path_text),
        Path(archive_path_text), Path(stage_dir_text),
    )
    expected = int(jobs[0].get("_strict_warm_worker_count", 1))
    ready_dir = Path(stage_dir_text) / "deployment_initialization" / "ready"
    ready_dir.mkdir(parents=True, exist_ok=True)
    _pipeline.write_json(
        ready_dir / f"{device_name.replace(':', '_')}.json",
        {"device": device_name, "process_id": int(os.getpid()), "ready": True},
    )
    deadline = time.monotonic() + 180.0
    while len(list(ready_dir.glob("*.json"))) < expected:
        if time.monotonic() >= deadline:
            raise RuntimeError("timed out waiting for all GPU workers to warm up")
        time.sleep(0.05)
    return _BASE_QUEUE_WORKER(
        device_name, jobs, config, checkpoint_path_text, checkpoint_identity,
        checkpoint_science, archive_path_text, stage_dir_text, smoke, resume,
    )


def build_balanced_queues(
    jobs: Sequence[Mapping[str, Any]], worker_count: int
) -> list[list[Mapping[str, Any]]]:
    """Ensure every GPU worker receives Frozen jobs before its paired LM jobs."""
    queues: list[list[Mapping[str, Any]]] = [[] for _ in range(worker_count)]
    methods = _pipeline.active_methods_from_jobs(jobs) if hasattr(
        _pipeline, "active_methods_from_jobs"
    ) else []
    if not methods:
        methods = []
        for job in jobs:
            method = str(job["method"])
            if method not in methods:
                methods.append(method)
    offset = 0
    for method in methods:
        method_jobs = [job for job in jobs if str(job["method"]) == method]
        for index, job in enumerate(method_jobs):
            if isinstance(job, dict):
                job["_strict_warm_worker_count"] = int(worker_count)
            queues[(offset + index) % worker_count].append(job)
        offset = (offset + len(method_jobs)) % worker_count
    return queues


def aggregate_stage(
    stage_dir: Path, config: Mapping[str, Any], experiment_seed: int,
    case_count: int, smoke: bool,
) -> dict[str, Any]:
    summary = _BASE_AGGREGATE(stage_dir, config, experiment_seed, case_count, smoke)
    rows = []
    per_case_path = stage_dir / "per_case_results.csv"
    if per_case_path.exists():
        import csv
        with per_case_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    frozen = [row for row in rows if row.get("method_key") == "operator_projected_grid"]
    lm_rows = [row for row in rows if row.get("method_key") == "projected_lm"]
    timing_rows = []
    for key, label, group in (
        ("operator_projected_grid", "strict_warm_gpu_online", frozen),
        ("projected_lm", "cpu_online_end_to_end", lm_rows),
    ):
        values = np.asarray(
            [float(row["online_end_to_end_seconds"]) for row in group],
            dtype=np.float64,
        )
        timing_rows.append(
            {
                "method_key": key,
                "timing_definition": label,
                "case_count": int(values.size),
                "mean_seconds": float(np.mean(values)) if values.size else None,
                "median_seconds": float(np.median(values)) if values.size else None,
            }
        )
    _pipeline.write_csv(stage_dir / "timing_breakdown.csv", timing_rows)
    summary["timing_rows"] = timing_rows
    summary["timing_protocol"] = (
        "All GPU workers cross a readiness barrier after model loading, CUDA "
        "warmup and fixed-time Trunk caching. Frozen online time starts with the "
        "case History Branch. Direct LM is timed end-to-end on CPU."
    )
    summary["operator_implementation"] = (
        "High-75 eight-history, forty-observation all-state data; normalized "
        "41x61 grid, Top-10 projected Adam, min/check/tolerances/patience/max = "
        "100/20/1e-5/1e-7/5/500; data loss only; no L-BFGS-B."
    )
    summary["lm_implementation"] = (
        "Five LHS starts, direct causal RK4 projected LM, minimum five iterations, "
        "relative-objective tolerance 1e-8, patience three, maximum 100."
    )
    summary["deployment_initialization"] = [
        _pipeline.read_json(path)
        for path in sorted((stage_dir / "deployment_initialization").glob("worker_*.json"))
    ]
    _pipeline.write_json(stage_dir / "comparison.json", summary)
    return summary


_cached._fixed_features = _fixed_features_strict_warm
_pipeline.validate_config = validate_config
_pipeline.run_operator_job = run_operator_job_strict_warm
_pipeline.run_lm_job = run_lm_job_strict_paired
_pipeline.queue_worker = queue_worker
_pipeline.build_round_robin_queues = build_balanced_queues
_pipeline.aggregate_stage = aggregate_stage


def main(argv: Sequence[str] | None = None) -> int:
    return _pipeline.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
