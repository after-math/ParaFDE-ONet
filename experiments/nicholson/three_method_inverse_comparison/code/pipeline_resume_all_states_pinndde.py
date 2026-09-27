#!/usr/bin/env python3
"""Resume only missing all-state PINN-DDE jobs with a finite loss barrier."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import pipeline_resume_all_states as _subset


_pipeline = _subset._pipeline
_ORIGINAL_METHODS = ("operator_projected_grid", "projected_lm", "pinndde")
_SELECTED_METHODS = ("pinndde",)
_SUBSET_VALIDATE = _pipeline.validate_config
_BASE_RUN_PINNDDE_JOB = _pipeline.run_pinndde_job
_FINITE_BARRIER = 1.0e6
_FINITE_BARRIER_HITS = 0


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate the unchanged original protocol before selecting PINN-DDE."""
    current = _pipeline.METHODS
    _pipeline.METHODS = _ORIGINAL_METHODS
    try:
        _SUBSET_VALIDATE(config)
    finally:
        _pipeline.METHODS = current


def _finite_barrier(values: torch.Tensor) -> torch.Tensor:
    """Leave finite ordinary values unchanged and bound invalid line-search trials."""
    global _FINITE_BARRIER_HITS
    invalid = ~torch.isfinite(values)
    if bool(torch.any(invalid).detach().cpu()):
        _FINITE_BARRIER_HITS += int(invalid.detach().sum().cpu())
    return torch.nan_to_num(
        values,
        nan=_FINITE_BARRIER,
        posinf=_FINITE_BARRIER,
        neginf=-_FINITE_BARRIER,
    ).clamp(-_FINITE_BARRIER, _FINITE_BARRIER)


def pinndde_losses_finite(
    model: _pipeline.MultiStartPINNDDE,
    histories: torch.Tensor,
    history_grid: torch.Tensor,
    collocation_times: torch.Tensor,
    observation_times: torch.Tensor,
    observations: torch.Tensor,
    observed_states: Sequence[int],
    equation: _pipeline.NicholsonConfig,
    adaptive_weights: bool,
    detach_adaptive_weights: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Original PINN-DDE loss with a barrier for non-finite Wolfe trial points."""
    if not collocation_times.requires_grad:
        raise ValueError("PINN collocation times must require gradients")
    current = model(collocation_times)
    derivatives = []
    for state in range(_pipeline.STATE_DIM):
        derivatives.append(
            torch.autograd.grad(
                current[..., state].sum(),
                collocation_times,
                create_graph=True,
                retain_graph=True,
            )[0]
        )
    time_derivative = torch.stack(derivatives, dim=-1)
    delayed = _pipeline.pinn_delayed_state(
        model, histories, history_grid, collocation_times, equation
    )
    restart_count, history_count, point_count, _ = current.shape
    flat_current = current.reshape(
        restart_count * history_count, point_count, _pipeline.STATE_DIM
    )
    flat_delayed = delayed.reshape(
        restart_count * history_count, point_count, _pipeline.STATE_DIM
    )
    flat_parameters = model.physical_parameters().unsqueeze(1).expand(
        -1, history_count, -1
    ).reshape(restart_count * history_count, 2)
    residual = time_derivative - _pipeline.rhs_torch(
        flat_current, flat_delayed, flat_parameters, equation
    ).reshape_as(current)
    residual = _finite_barrier(residual)
    physics_by_state = torch.mean(residual.square(), dim=(1, 2))

    zero_times = torch.zeros(
        (model.restart_count, model.history_count, 1),
        dtype=collocation_times.dtype,
        device=collocation_times.device,
    )
    initial_error = model(zero_times)[:, :, 0, :] - histories[:, :, :, -1]
    initial_by_state = _finite_barrier(initial_error).square().mean(dim=1)

    predicted_observations = model(observation_times)
    states = torch.as_tensor(
        observed_states, dtype=torch.long, device=collocation_times.device
    )
    data_error = (
        predicted_observations.index_select(3, states)
        - observations.index_select(3, states)
    )
    data_by_state = _finite_barrier(data_error).square().mean(dim=(1, 2))

    components = torch.cat(
        (physics_by_state, initial_by_state, data_by_state), dim=1
    )
    components = torch.nan_to_num(
        components,
        nan=_FINITE_BARRIER**2,
        posinf=_FINITE_BARRIER**2,
        neginf=_FINITE_BARRIER**2,
    )
    if adaptive_weights:
        weights = components / components.sum(dim=1, keepdim=True).clamp_min(1.0e-15)
        if detach_adaptive_weights:
            weights = weights.detach()
        objective = torch.sum(weights * components, dim=1)
    else:
        objective = torch.sum(components, dim=1)
    summaries = {
        "physics_loss": physics_by_state.mean(dim=1),
        "initial_loss": initial_by_state.mean(dim=1),
        "data_loss": data_by_state.mean(dim=1),
    }
    if not bool(torch.all(torch.isfinite(objective)).detach().cpu()):
        raise FloatingPointError("finite PINN-DDE barrier failed")
    return objective, summaries


def run_pinndde_job_finite(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Preserve existing results and annotate only newly completed guarded jobs."""
    global _FINITE_BARRIER_HITS
    output_dir = Path(args[8] if len(args) > 8 else kwargs["output_dir"])
    resume = bool(args[10] if len(args) > 10 else kwargs.get("resume", False))
    existed = resume and (output_dir / "result.json").is_file()
    _FINITE_BARRIER_HITS = 0
    result = _BASE_RUN_PINNDDE_JOB(*args, **kwargs)
    if not existed:
        result.update(
            {
                "nonfinite_line_search_barrier": "nan/inf residuals mapped to a finite 1e6 barrier",
                "nonfinite_line_search_barrier_hits": int(_FINITE_BARRIER_HITS),
                "existing_completed_results_modified": 0,
            }
        )
        _pipeline.write_json(output_dir / "result.json", result)
    return result


_pipeline.validate_config = validate_config
_pipeline.METHODS = _SELECTED_METHODS
_pipeline.pinndde_losses = pinndde_losses_finite
_pipeline.run_pinndde_job = run_pinndde_job_finite


if __name__ == "__main__":
    raise SystemExit(_pipeline.main())
