"""Leakage-safe histories, parameter designs, trajectories and sensitivities."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from equation import (
    NODE_COUNT,
    PARAMETER_DIM,
    STATE_DIM,
    SmartGridConfig,
    solve_batch,
)


DATASET_FORMAT_VERSION = "delayed_smart_grid4node8d_forward_v1"


@dataclass(frozen=True)
class DatasetSplit:
    """One physical-unit operator split and its training-derived scales."""

    histories: np.ndarray
    parameters: np.ndarray
    solutions: np.ndarray
    history_times: np.ndarray
    output_times: np.ndarray
    state_mean: np.ndarray
    state_std: np.ndarray
    residual_rms: np.ndarray
    normalized_parameter_sensitivities: np.ndarray | None = None


def latin_hypercube(count: int, dimension: int, rng: np.random.Generator) -> np.ndarray:
    if count < 1 or dimension < 1:
        raise ValueError("Latin-hypercube dimensions must be positive")
    result = np.empty((count, dimension), dtype=np.float64)
    for column in range(dimension):
        permutation = rng.permutation(count)
        result[:, column] = (permutation + rng.random(count)) / count
    return result


def sample_parameters(
    count: int, equation: SmartGridConfig, rng: np.random.Generator
) -> np.ndarray:
    unit = latin_hypercube(count, PARAMETER_DIM, rng)
    return (equation.parameter_lower + unit * equation.parameter_span).astype(np.float32)


def sample_histories(
    count: int,
    history_times: np.ndarray,
    generator_config: Mapping[str, Any],
    equation: SmartGridConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate two-harmonic angle histories and their exact derivatives."""
    if count < 1 or history_times.ndim != 1 or history_times.size < 3:
        raise ValueError("history sampling requires a positive count and at least three sensors")
    if not math.isclose(float(history_times[0]), -equation.maximum_history, abs_tol=1e-10):
        raise ValueError("history grid begins at the wrong time")
    if not math.isclose(float(history_times[-1]), 0.0, abs_tol=1e-10):
        raise ValueError("history grid must end at zero")
    if str(generator_config["type"]) != "kinematically_compatible_two_harmonic":
        raise ValueError("unsupported history generator")
    nominal = np.asarray(generator_config["nominal_equilibrium_angles"], dtype=np.float64)
    if nominal.shape != (NODE_COUNT,):
        raise ValueError("four nominal equilibrium angles are required")
    offset_bounds = tuple(float(item) for item in generator_config["angle_offset_bounds"])
    drift_bounds = tuple(float(item) for item in generator_config["frequency_drift_bounds"])
    orders = tuple(int(item) for item in generator_config["harmonic_orders"])
    amplitude_bounds = tuple(
        tuple(float(item) for item in row)
        for row in generator_config["angle_harmonic_amplitude_bounds"]
    )
    phase_bounds = tuple(float(item) for item in generator_config["phase_bounds"])
    if orders != (1, 2) or len(amplitude_bounds) != len(orders):
        raise ValueError("the configured history must contain harmonics one and two")
    if not offset_bounds[0] < offset_bounds[1] or not drift_bounds[0] < drift_bounds[1]:
        raise ValueError("history offset and drift bounds must be increasing")

    offsets = rng.uniform(*offset_bounds, size=(count, NODE_COUNT))
    drifts = rng.uniform(*drift_bounds, size=(count, NODE_COUNT))
    theta = nominal[None, :, None] + offsets[:, :, None] + drifts[:, :, None] * history_times
    omega = np.broadcast_to(drifts[:, :, None], theta.shape).copy()
    for order, bounds in zip(orders, amplitude_bounds):
        amplitude = rng.uniform(*bounds, size=(count, NODE_COUNT))
        phase = rng.uniform(*phase_bounds, size=(count, NODE_COUNT))
        angular_frequency = 2.0 * math.pi * order / equation.fixed_delay
        argument = angular_frequency * (history_times[None, None, :] + equation.fixed_delay)
        argument = argument + phase[:, :, None]
        theta += amplitude[:, :, None] * np.sin(argument)
        omega += angular_frequency * amplitude[:, :, None] * np.cos(argument)

    if str(generator_config.get("gauge")) != "subtract_theta_1_at_t0_from_all_angle_histories":
        raise ValueError("unsupported angle gauge")
    theta -= theta[:, :1, -1:]
    histories = np.concatenate((theta, omega), axis=1)
    if not np.isfinite(histories).all():
        raise FloatingPointError("non-finite history sample")
    tolerance = float(generator_config.get("symmetry_check_tolerance", 1e-6))
    for left in range(1, NODE_COUNT):
        for right in range(left + 1, NODE_COUNT):
            difference = histories[:, [left, left + NODE_COUNT], :] - histories[:, [right, right + NODE_COUNT], :]
            if np.any(np.linalg.norm(difference.reshape(count, -1), axis=1) <= tolerance):
                raise RuntimeError("two consumer histories are numerically symmetric")
    return histories.astype(np.float32)


def balanced_pair_indices(
    history_count: int,
    parameter_count: int,
    pairs_per_history: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair every history with distinct, approximately balanced parameter indices."""
    if pairs_per_history < 1 or pairs_per_history > parameter_count:
        raise ValueError("pairs_per_history must lie between one and parameter_count")
    history_order = rng.permutation(history_count)
    parameter_order = rng.permutation(parameter_count)
    history_indices = np.repeat(history_order, pairs_per_history)
    offsets = np.arange(pairs_per_history, dtype=np.int64)[None, :]
    bases = np.arange(history_count, dtype=np.int64)[:, None] * pairs_per_history
    parameter_positions = (bases + offsets) % parameter_count
    parameter_indices = parameter_order[parameter_positions].reshape(-1)
    return history_indices.astype(np.int64), parameter_indices.astype(np.int64)


def _solve_task(arguments: tuple[Any, ...]) -> tuple[int, np.ndarray]:
    start, histories, parameters, history_times, output_times, equation, step = arguments
    return int(start), solve_batch(histories, parameters, history_times, output_times, equation, step)


def solve_parallel(
    histories: np.ndarray,
    parameters: np.ndarray,
    history_times: np.ndarray,
    output_times: np.ndarray,
    equation: SmartGridConfig,
    internal_step: float,
    processes: int,
    chunk_size: int,
) -> np.ndarray:
    """Solve ordered chunks in one process or a spawn-safe CPU pool."""
    count = histories.shape[0]
    if parameters.shape[0] != count or processes < 1 or chunk_size < 1:
        raise ValueError("invalid parallel solve inputs")
    if count == 0:
        return np.empty((0, output_times.size, STATE_DIM), dtype=np.float32)
    if processes == 1:
        return solve_batch(histories, parameters, history_times, output_times, equation, internal_step)
    tasks = [
        (
            start,
            histories[start : start + chunk_size],
            parameters[start : start + chunk_size],
            history_times,
            output_times,
            equation,
            internal_step,
        )
        for start in range(0, count, chunk_size)
    ]
    result = np.empty((count, output_times.size, STATE_DIM), dtype=np.float32)
    workers = min(int(processes), len(tasks))
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as executor:
        futures = [executor.submit(_solve_task, task) for task in tasks]
        for future in as_completed(futures):
            start, values = future.result()
            result[start : start + values.shape[0]] = values
    return result


def finite_difference_sensitivities(
    histories: np.ndarray,
    parameters: np.ndarray,
    base_solutions: np.ndarray,
    history_times: np.ndarray,
    output_times: np.ndarray,
    equation: SmartGridConfig,
    internal_step: float,
    finite_difference_steps: np.ndarray,
    processes: int,
    chunk_size: int,
) -> np.ndarray:
    """Compute interval-scaled derivatives with centered or second-order one-sided rules."""
    if finite_difference_steps.shape != (PARAMETER_DIM,) or np.any(finite_difference_steps <= 0.0):
        raise ValueError("three positive finite-difference steps are required")
    count = parameters.shape[0]
    sensitivities = np.empty((*base_solutions.shape, PARAMETER_DIM), dtype=np.float32)
    lower = equation.parameter_lower
    upper = equation.parameter_upper
    for column, step in enumerate(finite_difference_steps):
        centered = (parameters[:, column] - step >= lower[column]) & (
            parameters[:, column] + step <= upper[column]
        )
        lower_edge = parameters[:, column] - step < lower[column]
        upper_edge = parameters[:, column] + step > upper[column]
        if np.any(lower_edge & upper_edge):
            raise ValueError("finite-difference step is too large for its parameter interval")

        plus_parameters = parameters.copy()
        minus_parameters = parameters.copy()
        plus_parameters[:, column] = np.minimum(parameters[:, column] + step, upper[column])
        minus_parameters[:, column] = np.maximum(parameters[:, column] - step, lower[column])
        plus = solve_parallel(
            histories, plus_parameters, history_times, output_times, equation,
            internal_step, processes, chunk_size,
        )
        minus = solve_parallel(
            histories, minus_parameters, history_times, output_times, equation,
            internal_step, processes, chunk_size,
        )
        derivative = np.empty_like(base_solutions, dtype=np.float32)
        derivative[centered] = (plus[centered] - minus[centered]) / (2.0 * step)
        if np.any(lower_edge):
            plus_two_parameters = parameters[lower_edge].copy()
            plus_two_parameters[:, column] += 2.0 * step
            plus_two = solve_parallel(
                histories[lower_edge], plus_two_parameters, history_times, output_times,
                equation, internal_step, processes, chunk_size,
            )
            derivative[lower_edge] = (
                -3.0 * base_solutions[lower_edge] + 4.0 * plus[lower_edge] - plus_two
            ) / (2.0 * step)
        if np.any(upper_edge):
            minus_two_parameters = parameters[upper_edge].copy()
            minus_two_parameters[:, column] -= 2.0 * step
            minus_two = solve_parallel(
                histories[upper_edge], minus_two_parameters, history_times, output_times,
                equation, internal_step, processes, chunk_size,
            )
            derivative[upper_edge] = (
                3.0 * base_solutions[upper_edge] - 4.0 * minus[upper_edge] + minus_two
            ) / (2.0 * step)
        sensitivities[..., column] = derivative * equation.parameter_span[column]
    if not np.isfinite(sensitivities).all():
        raise FloatingPointError("non-finite sensitivity labels")
    return sensitivities


def compute_training_scales(solutions: np.ndarray, output_times: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute state standardization and derivative RMS using only training trajectories."""
    state_mean = np.mean(solutions, axis=(0, 1), dtype=np.float64)
    state_std = np.std(solutions, axis=(0, 1), dtype=np.float64)
    state_std = np.maximum(state_std, float(floor))
    subset = solutions[: min(4096, solutions.shape[0])].astype(np.float64)
    time_step = float(output_times[1] - output_times[0])
    derivatives = (subset[:, 2:, :] - subset[:, :-2, :]) / (2.0 * time_step)
    residual_rms = np.sqrt(np.mean(derivatives**2, axis=(0, 1)))
    residual_rms = np.maximum(residual_rms, float(floor))
    return state_mean.astype(np.float32), state_std.astype(np.float32), residual_rms.astype(np.float32)


def _save_split(path: Path, split: DatasetSplit) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values: dict[str, Any] = {
        "histories": split.histories,
        "parameters": split.parameters,
        "solutions": split.solutions,
        "history_times": split.history_times,
        "output_times": split.output_times,
        "state_mean": split.state_mean,
        "state_std": split.state_std,
        "residual_rms": split.residual_rms,
    }
    if split.normalized_parameter_sensitivities is not None:
        values["normalized_parameter_sensitivities"] = split.normalized_parameter_sensitivities
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **values)
    temporary.replace(path)


def _load_split(path: Path) -> DatasetSplit:
    with np.load(path, allow_pickle=False) as values:
        sensitivity = (
            values["normalized_parameter_sensitivities"].copy()
            if "normalized_parameter_sensitivities" in values.files else None
        )
        return DatasetSplit(
            histories=values["histories"].copy(),
            parameters=values["parameters"].copy(),
            solutions=values["solutions"].copy(),
            history_times=values["history_times"].copy(),
            output_times=values["output_times"].copy(),
            state_mean=values["state_mean"].copy(),
            state_std=values["state_std"].copy(),
            residual_rms=values["residual_rms"].copy(),
            normalized_parameter_sensitivities=sensitivity,
        )


def dataset_signature(config: Mapping[str, Any]) -> str:
    payload = {
        "format": DATASET_FORMAT_VERSION,
        "data_seed": int(config["randomness"]["data_seed"]),
        "data": config["data"],
        "equation": config["equation"],
        "quality_gates": config["quality_gates"],
        "report_parameter_jacobian_metrics": bool(
            config["reporting"].get("report_parameter_jacobian_metrics", False)
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _quality_report(
    histories: np.ndarray,
    parameters: np.ndarray,
    solutions: np.ndarray,
    history_times: np.ndarray,
    output_times: np.ndarray,
    equation: SmartGridConfig,
    data_config: Mapping[str, Any],
    quality_config: Mapping[str, Any],
    processes: int,
) -> dict[str, Any]:
    count = min(int(quality_config["convergence_check_cases"]), histories.shape[0])
    reference_step = float(quality_config["reference_step_for_convergence_check"])
    reference = solve_parallel(
        histories[:count], parameters[:count], history_times, output_times, equation,
        reference_step, processes, int(data_config["solver_chunk_size"]),
    )
    difference = solutions[:count].astype(np.float64) - reference.astype(np.float64)
    relative = np.linalg.norm(difference.reshape(count, -1), axis=1) / np.maximum(
        np.linalg.norm(reference.reshape(count, -1), axis=1), 1e-12
    )
    frequencies = np.abs(reference[..., NODE_COUNT:])
    angle_spread = np.ptp(reference[..., :NODE_COUNT], axis=-1)
    report = {
        "case_count": int(count),
        "relative_l2_median": float(np.median(relative)),
        "relative_l2_p99": float(np.quantile(relative, 0.99)),
        "relative_l2_max": float(np.max(relative)),
        "maximum_absolute_frequency": float(np.max(frequencies)),
        "maximum_pairwise_angle_difference": float(np.max(angle_spread)),
    }
    failures = []
    if report["relative_l2_median"] > float(quality_config["relative_l2_median_max"]):
        failures.append("median solver discrepancy")
    if report["relative_l2_p99"] > float(quality_config["relative_l2_p99_max"]):
        failures.append("p99 solver discrepancy")
    if report["maximum_absolute_frequency"] > float(quality_config["maximum_absolute_frequency"]):
        failures.append("frequency limit")
    if report["maximum_pairwise_angle_difference"] > float(quality_config["maximum_pairwise_angle_difference"]):
        failures.append("angle-spread limit")
    report["passed"] = not failures
    report["failures"] = failures
    return report


def _json_write(path: Path, values: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(values), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def generate_or_load_dataset(
    data_dir: Path,
    config: Mapping[str, Any],
    processes: int,
) -> tuple[DatasetSplit, DatasetSplit, DatasetSplit]:
    """Generate one immutable dataset or load an exact signature match."""
    signature = dataset_signature(config)
    manifest_path = data_dir / "manifest.json"
    paths = {name: data_dir / f"{name}.npz" for name in ("train", "validation", "test")}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("signature") != signature:
            raise RuntimeError("existing dataset signature does not match the resolved configuration")
        if not all(path.exists() for path in paths.values()):
            raise RuntimeError("dataset manifest exists but one or more split files are missing")
        return _load_split(paths["train"]), _load_split(paths["validation"]), _load_split(paths["test"])

    data_dir.mkdir(parents=True, exist_ok=True)
    data_config = config["data"]
    equation = SmartGridConfig.from_mapping(config["equation"])
    rng = np.random.default_rng(int(config["randomness"]["data_seed"]))
    history_times = np.linspace(
        -equation.maximum_history, 0.0, int(data_config["history_sensors"]), dtype=np.float64
    )
    output_times = np.linspace(
        0.0, float(data_config["horizon"]), int(data_config["output_points"]), dtype=np.float64
    )
    step = float(data_config["internal_step"])
    chunk_size = int(data_config["solver_chunk_size"])

    history_pool = sample_histories(
        int(data_config["history_count"]), history_times, data_config["history_generator"], equation, rng
    )
    parameter_pool = sample_parameters(int(data_config["parameter_count"]), equation, rng)
    history_indices, parameter_indices = balanced_pair_indices(
        history_pool.shape[0], parameter_pool.shape[0], int(data_config["pairs_per_history"]), rng
    )
    train_histories = history_pool[history_indices]
    train_parameters = parameter_pool[parameter_indices]
    train_solutions = solve_parallel(
        train_histories, train_parameters, history_times, output_times, equation,
        step, processes, chunk_size,
    )
    report = _quality_report(
        train_histories, train_parameters, train_solutions, history_times, output_times,
        equation, data_config, config["quality_gates"], processes,
    )
    _json_write(data_dir / "quality_report.json", report)
    if not report["passed"]:
        raise RuntimeError(f"dataset quality gates failed: {report['failures']}")
    finite_steps = np.asarray(data_config["sensitivity_finite_difference_steps"], dtype=np.float64)
    sensitivities = finite_difference_sensitivities(
        train_histories, train_parameters, train_solutions, history_times, output_times,
        equation, step, finite_steps, processes, chunk_size,
    )

    validation_count = int(data_config["validation_count"])
    validation_histories = sample_histories(
        validation_count, history_times, data_config["history_generator"], equation, rng
    )
    validation_parameters = sample_parameters(validation_count, equation, rng)
    validation_solutions = solve_parallel(
        validation_histories, validation_parameters, history_times, output_times,
        equation, step, processes, chunk_size,
    )
    test_count = int(data_config["test_count"])
    test_histories = sample_histories(
        test_count, history_times, data_config["history_generator"], equation, rng
    )
    test_parameters = sample_parameters(test_count, equation, rng)
    test_solutions = solve_parallel(
        test_histories, test_parameters, history_times, output_times,
        equation, step, processes, chunk_size,
    )
    test_sensitivities = None
    if bool(config["reporting"].get("report_parameter_jacobian_metrics", False)):
        test_sensitivities = finite_difference_sensitivities(
            test_histories, test_parameters, test_solutions, history_times, output_times,
            equation, step, finite_steps, processes, chunk_size,
        )

    state_mean, state_std, residual_rms = compute_training_scales(
        train_solutions, output_times, float(data_config["state_normalization_std_floor"])
    )
    train = DatasetSplit(
        train_histories, train_parameters, train_solutions, history_times, output_times,
        state_mean, state_std, residual_rms, sensitivities,
    )
    validation = DatasetSplit(
        validation_histories, validation_parameters, validation_solutions,
        history_times, output_times, state_mean, state_std, residual_rms,
    )
    test = DatasetSplit(
        test_histories, test_parameters, test_solutions,
        history_times, output_times, state_mean, state_std, residual_rms,
        test_sensitivities,
    )
    for name, split in (("train", train), ("validation", validation), ("test", test)):
        _save_split(paths[name], split)
    _json_write(
        manifest_path,
        {
            "format": DATASET_FORMAT_VERSION,
            "signature": signature,
            "train_count": int(train_histories.shape[0]),
            "validation_count": validation_count,
            "test_count": test_count,
            "state_mean": state_mean.tolist(),
            "state_std": state_std.tolist(),
            "residual_rms": residual_rms.tolist(),
            "quality_report": report,
        },
    )
    return train, validation, test


def load_dataset(data_dir: Path) -> tuple[DatasetSplit, DatasetSplit, DatasetSplit]:
    return tuple(_load_split(data_dir / f"{name}.npz") for name in ("train", "validation", "test"))  # type: ignore[return-value]
