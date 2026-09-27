#!/usr/bin/env python3
"""Run four raw-parameter operators with physical sensitivity supervision."""

from pathlib import Path
import math
import sys

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from _source_patch import execute_source, parent_code_path, patched_source

_SOURCE = parent_code_path(__file__, "scripts/pipeline.py")
_GUARD = '''if __name__ == "__main__":
    raise SystemExit(main())
'''
_TEXT = patched_source(_SOURCE, [(_GUARD, "")])
execute_source(_TEXT, globals(), __file__)

_parent_validate_config = validate_config
_parent_resolve_stage_config = resolve_stage_config


def validate_config(config):
    """Validate the parent protocol plus physical sensitivity supervision."""
    _parent_validate_config(config)
    data = config["data"]
    finite_difference_steps = data.get("sensitivity_finite_difference_steps")
    if (
        not isinstance(finite_difference_steps, list)
        or len(finite_difference_steps) != 2
        or any(float(value) <= 0.0 for value in finite_difference_steps)
    ):
        raise ValueError("two positive sensitivity_finite_difference_steps are required")
    operator = config["operator"]
    if int(operator.get("sensitivity_points", 0)) < 1:
        raise ValueError("sensitivity_points must be positive")
    r1_weight = float(operator.get("r1_sensitivity_weight", -1.0))
    r2_weight = float(operator.get("r2_sensitivity_weight", -1.0))
    if r1_weight < 0.0 or r2_weight < 0.0:
        raise ValueError("sensitivity weights must be nonnegative")
    if not math.isclose(r1_weight, r2_weight, rel_tol=0.0, abs_tol=1.0e-15):
        raise ValueError("symmetric r1/r2 equations require equal sensitivity weights")
    if "parameter_bounds" in operator:
        raise ValueError(
            "operator.parameter_bounds is intentionally absent: parameter inputs remain physical"
        )


def resolve_stage_config(config, smoke):
    """Add the smoke sensitivity-point budget to the unchanged parent resolver."""
    resolved = _parent_resolve_stage_config(config, smoke)
    if smoke:
        resolved["operator"]["sensitivity_points"] = int(
            resolved["smoke"]["sensitivity_points"]
        )
    return resolved


if __name__ == "__main__":
    raise SystemExit(main())
