#!/usr/bin/env python3
"""No-sensitivity ablation of the High-75 strict warm paired Frozen-vs-LM study.

This is an additive entry point: it reuses ``pipeline_high75_strict_warm_paired.py``
unchanged (High-75 eight-history, forty-observation, all S/E/I observed, noise
sigma in {0.005, 0.01, 0.02, 0.05}) and changes nothing about the inverse protocol,
case generation, Direct-LM solver, grid screening or early-stopping rules.

Unlike the VariableDelayCompetition2D sibling, the Delayed SEI3D no-sensitivity
model is byte-identical to the sensitivity model apart from display names: both
use ``OPERATOR_NETWORK_FORMAT = "delayed_sei3d_normalized_sensitivity_v1"`` and both
normalize physical parameters inside ``Operator3D.normalize_parameters``.  The
cached data objective in the base pipeline already performs that normalization, so
this wrapper only tags the run with a distinct experiment signature and ablation
metadata while loading the no-sensitivity checkpoint through the existing loader.

Direct projected-LM is untouched: it integrates the DDE numerically and is model
independent, so its results are directly comparable with the sensitivity run.
"""

from __future__ import annotations

from typing import Any, Sequence

import pipeline_high75_strict_warm_paired as _paired


_pipeline = _paired._pipeline


def validate_config(config: Any) -> None:
    """Keep the strict warm paired protocol and tag this run as the ablation."""
    _paired.validate_config(config)
    if not isinstance(config, dict):
        raise TypeError("the resolved experiment configuration must be mutable")
    config["experiment_name"] = "delayed_sei3d_high75_strict_warm_paired_no_sensitivity"
    config["ablation"] = {
        "parameter_normalization": True,
        "parameter_sensitivity_supervision": False,
        "checkpoint_training_design": "normalized_no_sensitivity",
        "comparison_baseline": (
            "delayed_sei3d_high75_strict_warm_paired"
            " (sensitivity-supervised frozen operator)"
        ),
    }


_pipeline.validate_config = validate_config


def main(argv: Sequence[str] | None = None) -> int:
    return _pipeline.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
