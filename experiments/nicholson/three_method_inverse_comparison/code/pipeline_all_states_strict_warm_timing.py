#!/usr/bin/env python3
"""Strict warm timing for the additive all-state Nicholson Frozen-vs-LM run."""

from __future__ import annotations

import gc
import os
from pathlib import Path
import platform
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import pipeline_all_states_cached_early_stop as _base


_pipeline = _base._pipeline
_BASE_VALIDATE = _base.validate_config
_BASE_BUILD_FEATURES = _base._build_fixed_features
_BASE_QUEUE_WORKER = _base.queue_worker
_BASE_BUILD_QUEUES = _base.build_method_balanced_queues
_BASE_AGGREGATE = _base.aggregate_stage
_TRUNK_CACHE: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


def validate_config(config: Mapping[str, Any]) -> None:
    _BASE_VALIDATE(config)
    if not isinstance(config, dict):
        raise TypeError("the resolved experiment configuration must be mutable")
    config["experiment_name"] = "nicholson_patch4d_all_states_strict_warm_timing"
    config["timing_protocol"] = {
        "persistent_model_per_gpu_worker": True,
        "all_workers_ready_barrier": True,
        "untimed_cuda_forward_backward_warmup": True,
        "untimed_fixed_observation_trunk_cache": True,
        "timed_history_branch_per_case": True,
        "timed_grid_screening_and_projected_adam": True,
    }


def _build_features_strict_warm(
    model: Any, history: torch.Tensor, observation_times: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    _pipeline.synchronize(history.device)
    started = time.perf_counter()
    with torch.no_grad():
        encoded = [
            1.0
            + branch(history[:, component, :]).reshape(
                history.shape[0], 4, model.latent_dim
            )
            for component, branch in enumerate(model.history_branches)
        ]
        history_features = encoded[0]
        for values in encoded[1:]:
            history_features = history_features * values
        history_features = history_features.detach()
    _pipeline.synchronize(history.device)
    history_seconds = time.perf_counter() - started

    cached = _TRUNK_CACHE.get(str(history.device))
    if cached is None:
        raise RuntimeError("strict warm Trunk cache was not initialized")
    cached_times, trunk_features = cached
    if cached_times.shape != observation_times.shape or not torch.equal(
        cached_times, observation_times.detach().cpu()
    ):
        raise RuntimeError("formal observation times differ from warm Trunk cache")
    return history_features, trunk_features, history_seconds, 0.0


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
    total_started = time.perf_counter()
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, checkpoint, cache_hit, load_seconds = _base._load_model_once(
        checkpoint_path, device
    )
    if cache_hit:
        raise RuntimeError("worker model cache must be empty before warmup")

    warm_job = operator_jobs[0]
    case = _pipeline.load_case(archive_path, int(warm_job["case_index"]))
    observed = _pipeline.condition_observations(
        case, _base.condition_by_key(config, str(warm_job["condition"]))
    )
    history = torch.as_tensor(observed["histories"], dtype=torch.float32, device=device)
    times = torch.as_tensor(observed["times"], dtype=torch.float32, device=device)
    observations = torch.as_tensor(observed["values"], dtype=torch.float32, device=device)
    equation = _pipeline.NicholsonConfig.from_mapping(
        checkpoint["resolved_config"]["equation"]
    )

    warm_started = time.perf_counter()
    history_features, trunk_features, _, _ = _BASE_BUILD_FEATURES(
        model, history, times
    )
    _TRUNK_CACHE[str(device)] = (times.detach().cpu().clone(), trunk_features.detach())
    with torch.no_grad():
        grid_probe = torch.zeros((256, 2), dtype=torch.float32, device=device)
        _base._cached_data_objective(
            model, grid_probe, history_features, trunk_features, observations,
            observed["states"], equation.parameter_bounds,
        )
    parameter_probe = torch.zeros(
        (10, 2), dtype=torch.float32, device=device, requires_grad=True
    )
    optimizer_probe = torch.optim.Adam([parameter_probe], lr=1.0e-3)
    objective, _ = _base._cached_data_objective(
        model, parameter_probe, history_features, trunk_features, observations,
        observed["states"], equation.parameter_bounds,
    )
    objective.sum().backward()
    optimizer_probe.step()
    _pipeline.synchronize(device)
    warm_seconds = time.perf_counter() - warm_started

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
    deadline = time.monotonic() + 120.0
    while len(list(ready_dir.glob("*.json"))) < expected:
        if time.monotonic() >= deadline:
            raise RuntimeError("timed out waiting for all GPU workers to warm up")
        time.sleep(0.05)
    return _BASE_QUEUE_WORKER(
        device_name, jobs, config, checkpoint_path_text, checkpoint_identity,
        checkpoint_science, archive_path_text, stage_dir_text, smoke, resume,
    )


def build_method_balanced_queues(
    jobs: Sequence[Mapping[str, Any]], worker_count: int
) -> list[list[Mapping[str, Any]]]:
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
    stage_dir: Path, config: Mapping[str, Any], experiment_seed: int,
    case_count: int, smoke: bool,
) -> dict[str, Any]:
    summary = _BASE_AGGREGATE(stage_dir, config, experiment_seed, case_count, smoke)
    summary["timing_protocol"] = (
        "All GPU workers cross a readiness barrier after checkpoint loading, CUDA "
        "warmup and fixed-time Trunk caching. Per-case online time starts at the "
        "case-specific History Branch and includes grid screening and Adam."
    )
    summary["deployment_initialization"] = [
        _pipeline.read_json(path)
        for path in sorted((stage_dir / "deployment_initialization").glob("worker_*.json"))
    ]
    _pipeline.write_json(stage_dir / "comparison.json", summary)
    return summary


_base._build_fixed_features = _build_features_strict_warm
_base.queue_worker = queue_worker
_base.build_method_balanced_queues = build_method_balanced_queues
_base.aggregate_stage = aggregate_stage
_pipeline.validate_config = validate_config
_pipeline.aggregate_stage = aggregate_stage


def main(argv: Sequence[str] | None = None) -> int:
    return _pipeline.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
