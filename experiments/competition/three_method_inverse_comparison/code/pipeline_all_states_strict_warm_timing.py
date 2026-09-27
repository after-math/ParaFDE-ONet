#!/usr/bin/env python3
"""Strict warm-timing wrapper for the additive all-state Frozen-vs-LM study.

The scientific protocol is inherited from
``pipeline_all_states_cached_early_stop.py``.  This entry point changes timing
only: every GPU worker loads and warms the frozen operator and caches the fixed
observation-time Trunk before any formal inverse job starts.  Per-case timing
still includes History Branch encoding, grid screening and projected Adam.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import pipeline_all_states_cached_early_stop as _base


_pipeline = _base._pipeline
_BASE_VALIDATE_CONFIG = _base.validate_config
_BASE_BUILD_FIXED_FEATURES = _base._build_fixed_features
_BASE_QUEUE_WORKER = _base.queue_worker
_BASE_AGGREGATE_STAGE = _base.aggregate_stage
_BASE_BUILD_QUEUES = _base.build_method_balanced_queues

# Each spawned worker owns one process-local cache and one CUDA device.
_TRUNK_CACHE: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


def validate_config(config: Mapping[str, Any]) -> None:
    """Resolve the existing protocol and give this timing study a new signature."""
    _BASE_VALIDATE_CONFIG(config)
    if not isinstance(config, dict):
        raise TypeError("the resolved experiment configuration must be mutable")
    config["experiment_name"] = (
        "variable_delay_competition2d_all_states_strict_warm_timing"
    )
    config["timing_protocol"] = {
        "persistent_model_per_gpu_worker": True,
        "untimed_cuda_forward_backward_warmup": True,
        "untimed_fixed_observation_trunk_cache": True,
        "timed_history_branch_per_case": True,
        "timed_grid_screening_and_projected_adam": True,
        "deployment_initialization_reported_separately": True,
    }


def _trunk_cache_key(device: torch.device) -> str:
    return str(device)


def _build_fixed_features_strict_warm(
    model: Any, history: torch.Tensor, observation_times: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Time per-case History Branch work and reuse the untimed fixed Trunk."""
    _pipeline.synchronize(history.device)
    started = time.perf_counter()
    with torch.no_grad():
        encoded = [
            1.0
            + branch(history[:, component, :]).reshape(
                history.shape[0], 2, model.latent_dim
            )
            for component, branch in enumerate(model.history_branches)
        ]
        history_features = encoded[0]
        for values in encoded[1:]:
            history_features = history_features * values
        history_features = history_features.detach()
    _pipeline.synchronize(history.device)
    history_seconds = time.perf_counter() - started

    cached = _TRUNK_CACHE.get(_trunk_cache_key(history.device))
    if cached is None:
        raise RuntimeError("strict warm Trunk cache was not initialized")
    cached_times, trunk_features = cached
    if cached_times.shape != observation_times.shape or not torch.equal(
        cached_times, observation_times.detach().cpu()
    ):
        raise RuntimeError("formal observation times differ from the warm Trunk cache")
    return history_features, trunk_features, history_seconds, 0.0


def _initialize_gpu_worker(
    device_name: str,
    jobs: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    checkpoint_path: Path,
    archive_path: Path,
    stage_dir: Path,
) -> None:
    """Load, cache and warm one model before formal per-case timers start."""
    operator_jobs = [job for job in jobs if job["method"] == "operator_projected_grid"]
    if not operator_jobs:
        return

    initialization_started = time.perf_counter()
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, checkpoint, cache_hit, load_seconds = _base._load_model_once(
        checkpoint_path, device
    )
    if cache_hit:
        raise RuntimeError("worker model cache must be empty before deployment warmup")

    warm_job = operator_jobs[0]
    case = _pipeline.load_case(archive_path, int(warm_job["case_index"]))
    observed = _pipeline.condition_observations(
        case, _base.condition_by_key(config, str(warm_job["condition"]))
    )
    history = torch.as_tensor(
        observed["histories"], dtype=torch.float32, device=device
    )
    observation_times = torch.as_tensor(
        observed["times"], dtype=torch.float32, device=device
    )
    observations = torch.as_tensor(
        observed["values"], dtype=torch.float32, device=device
    )
    equation = _pipeline.CompetitionConfig.from_mapping(
        checkpoint["resolved_config"]["equation"]
    )

    warmup_started = time.perf_counter()
    history_features, trunk_features, _, _ = _BASE_BUILD_FIXED_FEATURES(
        model, history, observation_times
    )
    _TRUNK_CACHE[_trunk_cache_key(device)] = (
        observation_times.detach().cpu().clone(),
        trunk_features.detach(),
    )

    # Warm both the batched grid-screening path and the differentiable Adam path.
    with torch.no_grad():
        grid_probe = torch.zeros((256, 2), dtype=torch.float32, device=device)
        _base._cached_data_objective(
            model,
            grid_probe,
            history_features,
            trunk_features,
            observations,
            observed["states"],
            equation.parameter_bounds,
        )
    parameter_probe = torch.zeros(
        (10, 2), dtype=torch.float32, device=device, requires_grad=True
    )
    optimizer_probe = torch.optim.Adam([parameter_probe], lr=1.0e-3)
    objective, _ = _base._cached_data_objective(
        model,
        parameter_probe,
        history_features,
        trunk_features,
        observations,
        observed["states"],
        equation.parameter_bounds,
    )
    objective.sum().backward()
    optimizer_probe.step()
    _pipeline.synchronize(device)
    warmup_seconds = time.perf_counter() - warmup_started

    del parameter_probe, optimizer_probe, objective, grid_probe, history_features
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    total_seconds = time.perf_counter() - initialization_started

    deployment_dir = stage_dir / "deployment_initialization"
    deployment_dir.mkdir(parents=True, exist_ok=True)
    safe_device = device_name.replace(":", "_")
    _pipeline.write_json(
        deployment_dir / f"worker_{os.getpid()}_{safe_device}.json",
        {
            "process_id": int(os.getpid()),
            "device": device_name,
            "accelerator_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "cpu"
            ),
            "checkpoint_load_seconds": float(load_seconds),
            "cuda_and_operator_warmup_seconds": float(warmup_seconds),
            "total_deployment_initialization_seconds": float(total_seconds),
            "fixed_observation_trunk_cached": True,
            "excluded_from_per_case_online_timing": True,
        },
    )


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
    """Initialize deployment once, then run every case with a warm model."""
    _initialize_gpu_worker(
        device_name,
        jobs,
        config,
        Path(checkpoint_path_text),
        Path(archive_path_text),
        Path(stage_dir_text),
    )
    expected_workers = int(jobs[0].get("_strict_warm_worker_count", 1))
    ready_dir = Path(stage_dir_text) / "deployment_initialization" / "ready"
    ready_dir.mkdir(parents=True, exist_ok=True)
    safe_device = device_name.replace(":", "_")
    _pipeline.write_json(
        ready_dir / f"{safe_device}.json",
        {"device": device_name, "process_id": int(os.getpid()), "ready": True},
    )
    barrier_deadline = time.monotonic() + 120.0
    while len(list(ready_dir.glob("*.json"))) < expected_workers:
        if time.monotonic() >= barrier_deadline:
            raise RuntimeError("timed out waiting for all GPU workers to warm up")
        time.sleep(0.05)
    return _BASE_QUEUE_WORKER(
        device_name,
        jobs,
        config,
        checkpoint_path_text,
        checkpoint_identity,
        checkpoint_science,
        archive_path_text,
        stage_dir_text,
        smoke,
        resume,
    )


def build_method_balanced_queues(
    jobs: Sequence[Mapping[str, Any]], worker_count: int
) -> list[list[Mapping[str, Any]]]:
    """Balance methods and keep every smoke/formal worker queue nonempty."""
    queues: list[list[Mapping[str, Any]]] = [[] for _ in range(worker_count)]
    offset = 0
    for method in _base.METHODS:
        method_jobs = [job for job in jobs if job["method"] == method]
        for index, job in enumerate(method_jobs):
            queues[(offset + index) % worker_count].append(job)
        offset = (offset + len(method_jobs)) % worker_count
    for queue in queues:
        for job in queue:
            if isinstance(job, dict):
                job["_strict_warm_worker_count"] = int(worker_count)
    return queues


def aggregate_stage(
    stage_dir: Path,
    config: Mapping[str, Any],
    experiment_seed: int,
    case_count: int,
    smoke: bool,
) -> dict[str, Any]:
    summary = _BASE_AGGREGATE_STAGE(
        stage_dir, config, experiment_seed, case_count, smoke
    )
    summary["timing_protocol"] = (
        "Every GPU worker loads the checkpoint, warms forward/backward and "
        "caches the fixed observation-time Trunk before per-case timing. "
        "Online time starts with the case-specific History Branch and includes "
        "grid screening plus projected Adam."
    )
    deployment_files = sorted(
        (stage_dir / "deployment_initialization").glob("worker_*.json")
    )
    deployment_rows = [_pipeline.read_json(path) for path in deployment_files]
    summary["deployment_initialization"] = deployment_rows
    _pipeline.write_json(stage_dir / "comparison.json", summary)
    return summary


# Replace only the imported additive variant's hooks.  Original project files are
# untouched, and this entry point receives a distinct experiment signature.
_base._build_fixed_features = _build_fixed_features_strict_warm
_base.queue_worker = queue_worker
_base.build_method_balanced_queues = build_method_balanced_queues
_base.aggregate_stage = aggregate_stage
_pipeline.validate_config = validate_config
_pipeline.aggregate_stage = aggregate_stage


def main(argv: Sequence[str] | None = None) -> int:
    return _pipeline.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
