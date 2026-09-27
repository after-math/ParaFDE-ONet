"""Run the inverse pipeline when its history support is a checkpoint subset.

The original pipeline deliberately requires exact equality between the inverse
history distribution and the checkpoint training distribution.  A stratified
checkpoint has broader E/I-history support, while the established inverse
benchmark uses a strict subset of that support.  This entry point preserves all
original validation and only replaces equality by interval containment for the
two interval-valued history fields.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence

import numpy as np

import pipeline as _pipeline


_ORIGINAL_VALIDATE_CHECKPOINT_PROTOCOL = _pipeline.validate_checkpoint_protocol
_INTERVAL_FIELDS = ("history_level_ranges", "history_bounds")
_ATOL = 1.0e-12


def _validate_interval_subset(
    name: str,
    inverse_value: object,
    training_value: object,
) -> None:
    inverse = np.asarray(inverse_value, dtype=np.float64)
    training = np.asarray(training_value, dtype=np.float64)
    if inverse.shape != training.shape or inverse.ndim != 2 or inverse.shape[1] != 2:
        raise RuntimeError(
            f"inverse {name} and checkpoint training {name} must have matching "
            "(state_count, 2) shapes"
        )
    if np.any(inverse[:, 0] < training[:, 0] - _ATOL) or np.any(
        inverse[:, 1] > training[:, 1] + _ATOL
    ):
        raise RuntimeError(
            f"inverse {name} is not contained in the checkpoint training support"
        )
    if np.any(inverse[:, 0] > inverse[:, 1] + _ATOL):
        raise RuntimeError(f"inverse {name} contains an invalid interval")


def validate_checkpoint_protocol_subset(
    checkpoint: Mapping[str, object],
    config: Mapping[str, object],
) -> None:
    """Retain the original checks while allowing contained history intervals."""
    training_data = checkpoint["resolved_config"]["data"]
    for name in _INTERVAL_FIELDS:
        _validate_interval_subset(name, config[name], training_data[name])

    validation_copy = copy.deepcopy(checkpoint)
    copied_data = validation_copy["resolved_config"]["data"]
    for name in _INTERVAL_FIELDS:
        copied_data[name] = copy.deepcopy(config[name])
    _ORIGINAL_VALIDATE_CHECKPOINT_PROTOCOL(validation_copy, config)


def main(argv: Sequence[str] | None = None) -> int:
    _pipeline.validate_checkpoint_protocol = validate_checkpoint_protocol_subset
    return _pipeline.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
