#!/usr/bin/env python3
"""Resume only Frozen operator and Direct LM for all-state conditions."""

from __future__ import annotations

from typing import Any, Mapping

import pipeline_resume_all_states as _subset


_pipeline = _subset._pipeline
_ORIGINAL_METHODS = ("operator_projected_grid", "projected_lm", "pinndde")
_SELECTED_METHODS = ("operator_projected_grid", "projected_lm")
_SUBSET_VALIDATE = _pipeline.validate_config


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate the original three-method config before selecting two methods."""
    current = _pipeline.METHODS
    _pipeline.METHODS = _ORIGINAL_METHODS
    try:
        _SUBSET_VALIDATE(config)
    finally:
        _pipeline.METHODS = current


_pipeline.validate_config = validate_config
_pipeline.METHODS = _SELECTED_METHODS


if __name__ == "__main__":
    raise SystemExit(_pipeline.main())
