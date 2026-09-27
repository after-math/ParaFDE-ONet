#!/usr/bin/env python3
"""Resume only the four all-state noise conditions in an existing run.

This additive entry leaves the original pipeline untouched.  It preserves the
original experiment configuration and job signatures, but filters the formal
condition list to the four all-state observation conditions before scheduling.
"""

from __future__ import annotations

from typing import Any, Mapping

import pipeline as _pipeline


_ORIGINAL_CONDITION_DEFINITIONS = _pipeline.condition_definitions
_ORIGINAL_VALIDATE_CONFIG = _pipeline.validate_config
_ALL_STATE_KEYS = {
    "all_states_noise_0p005",
    "all_states_noise_0p01",
    "all_states_noise_0p02",
    "all_states_noise_0p05",
}


def all_state_condition_definitions(
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return exactly the four original all-state noise conditions."""
    selected = [
        condition
        for condition in _ORIGINAL_CONDITION_DEFINITIONS(config)
        if str(condition["condition"]) in _ALL_STATE_KEYS
    ]
    keys = {str(condition["condition"]) for condition in selected}
    if keys != _ALL_STATE_KEYS or len(selected) != 4:
        raise RuntimeError("the original protocol does not contain four all-state conditions")
    return selected


def validate_config(config: Mapping[str, Any]) -> None:
    """Run the complete original validation before enabling the subset."""
    current = _pipeline.condition_definitions
    _pipeline.condition_definitions = _ORIGINAL_CONDITION_DEFINITIONS
    try:
        _ORIGINAL_VALIDATE_CONFIG(config)
    finally:
        _pipeline.condition_definitions = current


_pipeline.validate_config = validate_config
_pipeline.condition_definitions = all_state_condition_definitions


if __name__ == "__main__":
    raise SystemExit(_pipeline.main())
