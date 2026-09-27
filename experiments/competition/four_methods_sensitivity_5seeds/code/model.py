"""Original four operators with raw physical parameter inputs and a new format id."""

from _source_patch import execute_source, parent_code_path, patched_source

_SOURCE = parent_code_path(__file__, "model.py")
_TEXT = patched_source(
    _SOURCE,
    [
        (
            'OPERATOR_NETWORK_FORMAT = "variable_delay_competition_operator_2d_v1"',
            'OPERATOR_NETWORK_FORMAT = "variable_delay_competition_operator_2d_sensitivity_v1"',
        )
    ],
)
execute_source(_TEXT, globals(), __file__)
