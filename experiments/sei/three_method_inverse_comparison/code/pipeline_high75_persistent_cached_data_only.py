#!/usr/bin/env python3
"""High-75 cached data-only inversion with one persistent model per worker."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import pipeline_high75_cached_data_only as _cached

_pipeline = _cached._pipeline
_CACHED_VALIDATE = _cached.validate_config
_CACHED_AGGREGATE = _cached.aggregate_stage
_BASE_LOAD_OPERATOR_CHECKPOINT = _pipeline.load_operator_checkpoint


_MODEL_CACHE: dict[tuple[str, str], tuple[_pipeline.Operator3D, dict[str, Any]]] = {}


def validate_config(config: Mapping[str, Any]) -> None:
    """Keep every scientific setting and record persistent worker reuse."""
    _CACHED_VALIDATE(config)
    config["operator_projected_grid"]["persistent_model_per_worker"] = True


def load_operator_checkpoint_persistent(
    path: Any,
    device: torch.device,
) -> tuple[_pipeline.Operator3D, dict[str, Any]]:
    """Load once in each worker process and reuse for its later inverse jobs."""
    key = (str(Path(path).expanduser().resolve()), str(device))
    cached = _MODEL_CACHE.get(key)
    if cached is None:
        model, checkpoint = _BASE_LOAD_OPERATOR_CHECKPOINT(path, device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        # The model owns the loaded weights now.  Keeping a second state-dict
        # copy on the GPU would roughly double persistent checkpoint memory.
        checkpoint = dict(checkpoint)
        checkpoint.pop("model_state_dict", None)
        cached = (model, checkpoint)
        _MODEL_CACHE[key] = cached
    return cached


def aggregate_stage(
    stage_dir: Path,
    config: Mapping[str, Any],
    experiment_seed: int,
    case_count: int,
    smoke: bool,
) -> dict[str, Any]:
    summary = _CACHED_AGGREGATE(
        stage_dir, config, experiment_seed, case_count, smoke
    )
    summary["operator_implementation"] = (
        "One checkpoint load per persistent GPU worker; normalized 41x61 screen; "
        "top-10 projected Adam with patience five and at most 500 steps; online "
        "data loss only; fixed history and observation-time features cached per "
        "case; no L-BFGS-B."
    )
    summary["timing_protocol"] = (
        "Per-case online time includes optimization and case-local work. The first "
        "case handled by each worker includes checkpoint loading; later cases reuse "
        "the resident model."
    )
    _pipeline.write_json(stage_dir / "comparison.json", summary)
    return summary


_pipeline.validate_config = validate_config
_pipeline.load_operator_checkpoint = load_operator_checkpoint_persistent
_pipeline.aggregate_stage = aggregate_stage


def main(argv: Sequence[str] | None = None) -> int:
    return _pipeline.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
