#!/usr/bin/env python3
"""High-75 Frozen inversion with cached fixed features and data loss only.

This additive entry point reuses the validated adaptive-Adam pilot.  It changes
only online evaluation: physics loss is disabled, while history-branch and
observation-time trunk features are cached once per inverse job.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
import weakref

import torch

import pipeline_high75_early_stop as _early

_pipeline = _early._pipeline
_EARLY_VALIDATE = _early.validate_config
_EARLY_AGGREGATE = _early.aggregate_stage


_FEATURE_CACHE: dict[str, Any] = {}


def validate_config(config: Mapping[str, Any]) -> None:
    """Retain adaptive Adam and disable only the online physics residual."""
    _EARLY_VALIDATE(config)
    config["operator_projected_grid"]["physics_weight"] = 0.0
    config["operator_projected_grid"]["cached_fixed_features"] = True


def _fixed_features(
    model: _pipeline.Operator3D,
    history: torch.Tensor,
    observation_times: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return detached history fusion and trunk features for one fixed job."""
    if model.model_type != "separate_parameter_shared":
        raise ValueError("cached inversion requires separate_parameter_shared")
    model_ref = _FEATURE_CACHE.get("model_ref")
    history_ref = _FEATURE_CACHE.get("history_ref")
    times_ref = _FEATURE_CACHE.get("times_ref")
    cache_matches = (
        model_ref is not None and model_ref() is model
        and history_ref is not None and history_ref() is history
        and times_ref is not None and times_ref() is observation_times
    )
    if not cache_matches:
        with torch.no_grad():
            encoded = [
                1.0 + branch(history[:, component, :]).reshape(
                    history.shape[0], _pipeline.STATE_DIM, model.latent_dim
                )
                for component, branch in enumerate(model.history_branches)
            ]
            history_features = encoded[0]
            for values in encoded[1:]:
                history_features = history_features * values
            trunk_features = model.time_trunk(model.time_features(observation_times))
        _FEATURE_CACHE.clear()
        _FEATURE_CACHE.update(
            model_ref=weakref.ref(model),
            history_ref=weakref.ref(history),
            times_ref=weakref.ref(observation_times),
            history_features=history_features.detach(),
            trunk_features=trunk_features.detach(),
        )
    return _FEATURE_CACHE["history_features"], _FEATURE_CACHE["trunk_features"]


def cached_operator_data_objective(
    model: _pipeline.Operator3D,
    normalized_parameters: torch.Tensor,
    history: torch.Tensor,
    observation_times: torch.Tensor,
    observations: torch.Tensor,
    observed_states: Sequence[int],
    bounds: Sequence[Sequence[float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate data MSE while recomputing only the parameter branch."""
    physical = _pipeline.normalized_to_physical(normalized_parameters, bounds)
    history_features, trunk_features = _fixed_features(
        model, history, observation_times
    )
    normalized_physical = model.normalize_parameters(physical)
    parameter_features = 1.0 + model.parameter_branch(normalized_physical)
    fused = history_features.unsqueeze(0) * parameter_features[:, None, None, :]
    predicted = (
        torch.einsum("chsl,ql->chqs", fused, trunk_features)
        * model.latent_scale
        + model.output_bias
    )
    states = torch.as_tensor(
        observed_states, dtype=torch.long, device=normalized_parameters.device
    )
    predicted = predicted.index_select(3, states)
    target = observations.index_select(2, states).unsqueeze(0)
    loss = torch.mean((predicted - target) ** 2, dim=(1, 2, 3))
    if not torch.all(torch.isfinite(loss)):
        raise FloatingPointError("non-finite cached operator data objective")
    return loss, physical


def aggregate_stage(
    stage_dir: Path,
    config: Mapping[str, Any],
    experiment_seed: int,
    case_count: int,
    smoke: bool,
) -> dict[str, Any]:
    summary = _EARLY_AGGREGATE(
        stage_dir, config, experiment_seed, case_count, smoke
    )
    summary["operator_implementation"] = (
        "Normalized 41x61 screen, top-10 projected Adam with paired early "
        "stopping and at most 500 steps; online data loss only; fixed history "
        "branch and observation-time trunk features cached once per case; no L-BFGS-B."
    )
    _pipeline.write_json(stage_dir / "comparison.json", summary)
    return summary


# The original screen and objective resolve this module-level function at run time.
_pipeline.validate_config = validate_config
_pipeline.operator_data_objective = cached_operator_data_objective
_pipeline.aggregate_stage = aggregate_stage


def main(argv: Sequence[str] | None = None) -> int:
    return _pipeline.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
