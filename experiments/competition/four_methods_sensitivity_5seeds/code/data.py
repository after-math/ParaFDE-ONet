"""Parent data pipeline extended with physical parameter-sensitivity labels."""

from _source_patch import execute_source, parent_code_path, patched_source

_SOURCE = parent_code_path(__file__, "data.py")

_FIELD_OLD = """    output_times: np.ndarray


class OperatorDataset"""
_FIELD_NEW = """    output_times: np.ndarray
    parameter_sensitivities: np.ndarray | None = None


class OperatorDataset"""

_SENSITIVITY_FUNCTION = r'''

def generate_parameter_sensitivities(
    histories: np.ndarray,
    parameters: np.ndarray,
    history_times: np.ndarray,
    output_times: np.ndarray,
    equation_values: Mapping[str, Any],
    internal_step: float,
    workers: int,
    chunk_size: int,
    finite_difference_steps: tuple[float, float],
) -> np.ndarray:
    """Generate physical ``dy/dr1`` and ``dy/dr2`` labels by central differences.

    The float32 return has shape ``[N,Q,2,2]``.  The final dimension contains
    derivatives with respect to the physical growth rates ``r1`` and ``r2``.  No
    input or target normalization is applied.  For samples near a parameter bound,
    the largest valid symmetric step no greater than the requested step is used.
    Only the training split calls this function.
    """
    bounds = np.asarray(equation_values["parameter_bounds"], dtype=np.float64)
    steps = np.asarray(finite_difference_steps, dtype=np.float64)
    physical_parameters = np.asarray(parameters, dtype=np.float64)
    if bounds.shape != (2, 2) or steps.shape != (2,) or np.any(steps <= 0.0):
        raise ValueError("invalid sensitivity finite-difference configuration")
    if np.any(physical_parameters < bounds[:, 0]) or np.any(
        physical_parameters > bounds[:, 1]
    ):
        raise ValueError("parameter row lies outside the physical bounds")
    sensitivities = np.empty(
        (histories.shape[0], output_times.size, 2, 2), dtype=np.float32
    )
    for parameter_index, parameter_name in enumerate(("r1", "r2")):
        distance_to_bounds = np.minimum(
            physical_parameters[:, parameter_index] - bounds[parameter_index, 0],
            bounds[parameter_index, 1] - physical_parameters[:, parameter_index],
        )
        effective_steps = np.minimum(steps[parameter_index], distance_to_bounds)
        if np.any(effective_steps <= 1.0e-7):
            raise RuntimeError(
                f"an {parameter_name} sample is too close to its bound for a central difference"
            )
        plus_parameters = physical_parameters.copy()
        minus_parameters = physical_parameters.copy()
        plus_parameters[:, parameter_index] += effective_steps
        minus_parameters[:, parameter_index] -= effective_steps
        LOGGER.info(
            "Solving +/-%s sensitivity trajectories (requested physical step %.6g)",
            parameter_name,
            steps[parameter_index],
        )
        plus_solutions = solve_parallel(
            histories,
            plus_parameters.astype(np.float32),
            history_times,
            output_times,
            equation_values,
            internal_step,
            workers,
            chunk_size,
        )
        minus_solutions = solve_parallel(
            histories,
            minus_parameters.astype(np.float32),
            history_times,
            output_times,
            equation_values,
            internal_step,
            workers,
            chunk_size,
        )
        denominator = (2.0 * effective_steps)[:, None, None]
        sensitivities[..., parameter_index] = (
            (plus_solutions.astype(np.float64) - minus_solutions.astype(np.float64))
            / denominator
        ).astype(np.float32)
        del plus_solutions, minus_solutions
    if not np.isfinite(sensitivities).all():
        raise RuntimeError("reference sensitivity generation produced non-finite values")
    return sensitivities
'''

_GENERATE_OLD = """        splits[name] = DatasetSplit(
            histories=histories, parameters=parameters, solutions=solutions,
            history_times=history_times.astype(np.float32),
            output_times=output_times.astype(np.float32),
        )"""
_GENERATE_NEW = """        parameter_sensitivities = None
        if name == "train":
            sensitivity_steps = tuple(
                float(value) for value in data["sensitivity_finite_difference_steps"]
            )
            if len(sensitivity_steps) != 2:
                raise ValueError("sensitivity_finite_difference_steps must contain two values")
            parameter_sensitivities = generate_parameter_sensitivities(
                histories,
                parameters,
                history_times,
                output_times,
                config["equation"],
                float(data["internal_step"]),
                workers,
                int(data["solver_chunk_size"]),
                sensitivity_steps,
            )
        splits[name] = DatasetSplit(
            histories=histories, parameters=parameters, solutions=solutions,
            history_times=history_times.astype(np.float32),
            output_times=output_times.astype(np.float32),
            parameter_sensitivities=parameter_sensitivities,
        )"""

_SAVE_OLD = """        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                histories=split.histories,
                parameters=split.parameters,
                solutions=split.solutions,
                history_times=split.history_times,
                output_times=split.output_times,
            )"""
_SAVE_NEW = """        arrays = {
            "histories": split.histories,
            "parameters": split.parameters,
            "solutions": split.solutions,
            "history_times": split.history_times,
            "output_times": split.output_times,
        }
        if split.parameter_sensitivities is not None:
            arrays["parameter_sensitivities"] = split.parameter_sensitivities
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)"""

_LOAD_OLD = """            result[name] = DatasetSplit(**{key: values[key] for key in (
                "histories", "parameters", "solutions", "history_times", "output_times"
            )})"""
_LOAD_NEW = """            sensitivity = (
                values["parameter_sensitivities"]
                if "parameter_sensitivities" in values.files
                else None
            )
            result[name] = DatasetSplit(
                histories=values["histories"],
                parameters=values["parameters"],
                solutions=values["solutions"],
                history_times=values["history_times"],
                output_times=values["output_times"],
                parameter_sensitivities=sensitivity,
            )"""

_TEXT = patched_source(
    _SOURCE,
    [
        (_FIELD_OLD, _FIELD_NEW),
        ("\n\ndef dataset_signature", _SENSITIVITY_FUNCTION + "\n\ndef dataset_signature"),
        (_GENERATE_OLD, _GENERATE_NEW),
        (_SAVE_OLD, _SAVE_NEW),
        (_LOAD_OLD, _LOAD_NEW),
    ],
)
execute_source(_TEXT, globals(), __file__)
