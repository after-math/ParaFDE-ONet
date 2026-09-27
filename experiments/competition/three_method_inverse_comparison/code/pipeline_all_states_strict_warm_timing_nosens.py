#!/usr/bin/env python3
"""No-sensitivity ablation of the all-state strict warm-timing Frozen-vs-LM study.

This is an additive entry point: it reuses
``pipeline_all_states_strict_warm_timing.py`` unchanged (all four all-state noise
conditions sigma in {0.005, 0.01, 0.02, 0.05}, one history, both species observed)
and changes nothing about the inverse protocol, case generation, Direct-LM solver,
grid screening or early-stopping rules.

The only scientific difference versus the sensitivity-supervised run is the frozen
operator checkpoint: this entry loads the model trained without parameter-sensitivity
supervision
(``four_methods_normalized_no_sensitivity_5seeds/outputs/variable_delay_competition_5seeds_*/full``).

That model differs from the sensitivity model in two ways that require no source
edits, only this wrapper:

1. its ``OPERATOR_NETWORK_FORMAT`` is
   ``variable_delay_competition_operator_2d_normalized_no_sensitivity_v1``, and its
   ``Operator2D`` keeps ``parameter_bounds`` buffers; and
2. its ``parameter_branch`` consumes ``[-1,1]``-normalized growth rates (the physical
   to ``[-1,1]`` mapping happens in ``Operator2D.normalize_parameters``), whereas the
   sensitivity model's ``parameter_branch`` consumes physical growth rates.

The wrapper therefore rebinds the checkpoint loader/registry to the no-sensitivity
model module and re-expresses the cached data objective with the required
normalization step. Direct projected-LM is untouched: it integrates the DDE
numerically and is model independent, so its results are directly comparable with
the sensitivity run.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import torch


_THIS_FILE = Path(__file__).resolve()
_INVERSE_DIR = _THIS_FILE.parents[1]          # three_method_inverse_comparison
_EXPERIMENT_ROOT = _THIS_FILE.parents[2]      # VariableDelayCompetition2D_..._2026-08-18
_NO_SENS_CODE = (
    _EXPERIMENT_ROOT / "four_methods_normalized_no_sensitivity_5seeds" / "code"
)
if not (_NO_SENS_CODE / "model.py").is_file():
    raise RuntimeError(f"no-sensitivity model source is missing: {_NO_SENS_CODE}")

# Import the no-sensitivity model module under the canonical name ``model`` so the
# base pipeline's ``from model import ...`` resolves to it.  ``data``, ``equation``
# and ``training`` remain the sensitivity project's modules (their history sampler,
# competition equation and physics residual are byte-identical to the no-sensitivity
# project), which keeps case generation perfectly paired with the sensitivity run.
for path_value in (str(_NO_SENS_CODE), str(_NO_SENS_CODE.parent)):
    while path_value in sys.path:
        sys.path.remove(path_value)
sys.path.insert(0, str(_NO_SENS_CODE))
import model as _nosens_model  # noqa: E402

# The validated strict warm-timing entry point (and, transitively, the cached
# early-stop base and the original pipeline).
import pipeline_all_states_strict_warm_timing as _timing  # noqa: E402


_pipeline = _timing._pipeline
_base = _timing._base

# Rebind every model-facing name so checkpoint metadata and loading accept the
# no-sensitivity format id and rebuild the matching ``Operator2D``.
_pipeline.OPERATOR_NETWORK_FORMAT = _nosens_model.OPERATOR_NETWORK_FORMAT
_pipeline.MODEL_TYPES = _nosens_model.MODEL_TYPES
_pipeline.MODEL_DISPLAY_NAMES = _nosens_model.MODEL_DISPLAY_NAMES
_pipeline.Operator2D = _nosens_model.Operator2D
_pipeline.load_operator_checkpoint = _nosens_model.load_operator_checkpoint


def _cached_data_objective_nosens(
    model: Any,
    normalized_parameters: torch.Tensor,
    history_features: torch.Tensor,
    trunk_features: torch.Tensor,
    observations: torch.Tensor,
    observed_states: Sequence[int],
    bounds: Sequence[Sequence[float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the no-sensitivity operator's data MSE for projected parameters.

    Identical to the base cached objective except that physical parameters are
    mapped back to ``[-1,1]`` before the parameter branch, matching this model's
    ``forward`` path (``parameter_features = parameter_branch(normalize_parameters(physical))``).
    """
    physical = _pipeline.normalized_to_physical(normalized_parameters, bounds)
    normalized_physical = model.normalize_parameters(physical)
    parameter_features = model.parameter_branch(normalized_physical)
    fusion = history_features.unsqueeze(0) * (
        1.0 + parameter_features[:, None, None, :]
    )
    prediction = (
        torch.einsum("chsp,qp->chqs", fusion, trunk_features)
        * float(model.latent_scale)
        + model.output_bias.reshape(1, 1, 1, 2)
    )
    states = torch.as_tensor(
        observed_states, dtype=torch.long, device=normalized_parameters.device
    )
    prediction = prediction.index_select(3, states)
    target = observations.index_select(2, states).unsqueeze(0)
    loss = torch.mean((prediction - target) ** 2, dim=(1, 2, 3))
    if not torch.all(torch.isfinite(loss)):
        raise FloatingPointError("non-finite cached operator observation objective")
    return loss, physical


# The base ``run_operator_job`` and ``_screen_grid`` resolve ``_cached_data_objective``
# as a module global at call time, so rebinding it here updates every call site.
_base._cached_data_objective = _cached_data_objective_nosens


def validate_config(config: Any) -> None:
    """Keep the strict warm-timing protocol and tag this run as the ablation."""
    _timing.validate_config(config)
    if not isinstance(config, dict):
        raise TypeError("the resolved experiment configuration must be mutable")
    config["experiment_name"] = (
        "variable_delay_competition2d_all_states_strict_warm_timing_no_sensitivity"
    )
    config["ablation"] = {
        "parameter_normalization": True,
        "parameter_sensitivity_supervision": False,
        "checkpoint_training_design": "normalized_no_sensitivity",
        "comparison_baseline": (
            "variable_delay_competition2d_all_states_strict_warm_timing"
            " (sensitivity-supervised frozen operator)"
        ),
    }


_pipeline.validate_config = validate_config


def main(argv: Sequence[str] | None = None) -> int:
    return _pipeline.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
