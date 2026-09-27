"""Leakage-safe physical-history data and normalized sensitivity generation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from .equation import (
    CONDITION_DIM,
    INVERSE_DIM,
    STATE_DIM,
    CSTRConfig,
    constant_histories,
    solve_batch,
)


LOGGER = logging.getLogger(__name__)
DATASET_FORMAT = "cstr_physical_history_forward_v1"


@dataclass(frozen=True)
class DatasetSplit:
    histories: np.ndarray
    conditions: np.ndarray
    solutions: np.ndarray
    history_times: np.ndarray
    output_times: np.ndarray
    normalized_sensitivities: np.ndarray | None = None


class OperatorDataset(Dataset[tuple[torch.Tensor, ...]]):
    def __init__(self, split: DatasetSplit) -> None:
        if split.normalized_sensitivities is None:
            raise ValueError("training split requires sensitivity labels")
        self.histories = torch.from_numpy(np.asarray(split.histories, dtype=np.float32))
        self.conditions = torch.from_numpy(np.asarray(split.conditions, dtype=np.float32))
        self.solutions = torch.from_numpy(np.asarray(split.solutions, dtype=np.float32))
        self.sensitivities = torch.from_numpy(
            np.asarray(split.normalized_sensitivities, dtype=np.float32)
        )

    def __len__(self) -> int:
        return int(self.histories.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return (
            self.histories[index],
            self.conditions[index],
            self.solutions[index],
            self.sensitivities[index],
        )


def latin_hypercube(count: int, dimension: int, rng: np.random.Generator) -> np.ndarray:
    if count < 1 or dimension < 1:
        raise ValueError("Latin-hypercube dimensions must be positive")
    result = np.empty((count, dimension), dtype=np.float64)
    for column in range(dimension):
        permutation = rng.permutation(count)
        result[:, column] = (permutation + rng.random(count)) / count
    return result


def sample_conditions(
    count: int, equation: CSTRConfig, rng: np.random.Generator
) -> np.ndarray:
    unit = latin_hypercube(count, CONDITION_DIM, rng)
    bounds = equation.bounds_array
    return (bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])).astype(np.float32)


def balanced_pair_indices(
    history_count: int,
    condition_count: int,
    pairs_per_history: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if min(history_count, condition_count, pairs_per_history) < 1:
        raise ValueError("pair dimensions must be positive")
    history_order = rng.permutation(history_count)
    condition_order = rng.permutation(condition_count)
    history_indices: list[int] = []
    condition_indices: list[int] = []
    for rank, history_index in enumerate(history_order):
        for offset in range(pairs_per_history):
            history_indices.append(int(history_index))
            condition_indices.append(
                int(condition_order[(rank + offset) % condition_count])
            )
    return np.asarray(history_indices), np.asarray(condition_indices)


def normal_mask(
    trajectories: np.ndarray,
    temperature_limit: float,
    conversion_minimum: float,
    temperature_std_limit: float,
) -> np.ndarray:
    late_count = max(2, trajectories.shape[1] // 4)
    late = trajectories[:, -late_count:, :]
    maximum_temperature = late[:, :, 1].max(axis=1)
    conversion = 1.0 - late[:, :, 0].mean(axis=1)
    temperature_std = late[:, :, 1].std(axis=1)
    return (
        (maximum_temperature <= temperature_limit)
        & (conversion >= conversion_minimum)
        & (temperature_std <= temperature_std_limit)
    )


def generate_physical_history_pool(
    count: int,
    equation: CSTRConfig,
    data: Mapping[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    sensors = int(data["history_sensors"])
    step = float(data["internal_step"])
    horizon = float(data["burn_in_horizon"])
    endpoint_min = float(data["history_endpoint_min"])
    candidate_batch = int(data["candidate_batch"])
    accepted: list[np.ndarray] = []
    accepted_count = 0
    candidates_seen = 0
    valid_seen = 0
    while accepted_count < count:
        conditions = sample_conditions(candidate_batch, equation, rng)
        trajectories = solve_batch(
            constant_histories(candidate_batch, sensors),
            conditions,
            horizon,
            int(round(horizon / step)) + 1,
            step,
            equation,
            return_internal=True,
        )
        valid = normal_mask(
            trajectories,
            float(data["normal_temperature_limit"]),
            float(data["normal_conversion_minimum"]),
            float(data["normal_late_temperature_std_limit"]),
        )
        indices = np.flatnonzero(valid)
        candidates_seen += candidate_batch
        valid_seen += indices.size
        if indices.size == 0:
            continue
        endpoint_low = int(round(endpoint_min / step))
        endpoint_high = trajectories.shape[1] - 1
        endpoints = rng.integers(endpoint_low, endpoint_high + 1, size=indices.size)
        histories = np.empty((indices.size, STATE_DIM, sensors), dtype=np.float32)
        for row, (source, endpoint) in enumerate(zip(indices, endpoints, strict=True)):
            start = endpoint - sensors + 1
            histories[row] = trajectories[source, start : endpoint + 1, :].T
        accepted.append(histories)
        accepted_count += histories.shape[0]
        LOGGER.info("Physical histories accepted: %d/%d", min(accepted_count, count), count)
    pool = np.concatenate(accepted, axis=0)[:count]
    diagnostics = {
        "precondition_acceptance_fraction": valid_seen / candidates_seen,
        "history_endpoint_std_c": float(pool[:, 0, -1].std()),
        "history_endpoint_std_t": float(pool[:, 1, -1].std()),
        "history_temporal_std_c_mean": float(pool[:, 0, :].std(axis=1).mean()),
        "history_temporal_std_t_mean": float(pool[:, 1, :].std(axis=1).mean()),
        "history_c_min": float(pool[:, 0, :].min()),
        "history_c_max": float(pool[:, 0, :].max()),
        "history_t_min": float(pool[:, 1, :].min()),
        "history_t_max": float(pool[:, 1, :].max()),
    }
    return pool, diagnostics


def _solve_worker(arguments: tuple[Any, ...]) -> tuple[int, np.ndarray]:
    start, histories, conditions, horizon, output_points, step, equation_values = arguments
    equation = CSTRConfig.from_mapping(equation_values)
    return int(start), solve_batch(
        histories, conditions, float(horizon), int(output_points), float(step), equation
    )


def solve_parallel(
    histories: np.ndarray,
    conditions: np.ndarray,
    horizon: float,
    output_points: int,
    step: float,
    equation_values: Mapping[str, Any],
    workers: int,
    chunk_size: int,
) -> np.ndarray:
    count = histories.shape[0]
    tasks = [
        (
            start,
            histories[start : start + chunk_size],
            conditions[start : start + chunk_size],
            horizon,
            output_points,
            step,
            dict(equation_values),
        )
        for start in range(0, count, chunk_size)
    ]
    result = np.empty((count, output_points, STATE_DIM), dtype=np.float32)
    if workers == 1:
        iterator = map(_solve_worker, tasks)
        pool = None
    else:
        context = mp.get_context("spawn")
        pool = context.Pool(processes=min(workers, len(tasks)))
        iterator = pool.imap_unordered(_solve_worker, tasks)
    completed = 0
    try:
        for start, values in iterator:
            result[start : start + values.shape[0]] = values
            completed += values.shape[0]
            if completed == count or completed % max(chunk_size, count // 10) < chunk_size:
                LOGGER.info("Reference trajectories completed: %d/%d", completed, count)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    if not np.isfinite(result).all():
        raise RuntimeError("reference solver produced non-finite values")
    return result


def generate_normalized_sensitivities(
    histories: np.ndarray,
    conditions: np.ndarray,
    data: Mapping[str, Any],
    equation_values: Mapping[str, Any],
    workers: int,
) -> np.ndarray:
    equation = CSTRConfig.from_mapping(equation_values)
    physical = np.asarray(conditions, dtype=np.float64)
    bounds = equation.bounds_array
    requested = np.asarray(data["sensitivity_finite_difference_steps"], dtype=np.float64)
    if requested.shape != (INVERSE_DIM,) or np.any(requested <= 0.0):
        raise ValueError("two positive finite-difference steps are required")
    result = np.empty(
        (histories.shape[0], int(data["output_points"]), STATE_DIM, INVERSE_DIM),
        dtype=np.float32,
    )
    common = {
        "horizon": float(data["prediction_horizon"]),
        "output_points": int(data["output_points"]),
        "step": float(data["internal_step"]),
        "equation_values": equation_values,
        "workers": workers,
        "chunk_size": int(data["solver_chunk_size"]),
    }
    for output_index, condition_index in enumerate(equation.inverse_condition_indices):
        distances = np.minimum(
            physical[:, condition_index] - bounds[condition_index, 0],
            bounds[condition_index, 1] - physical[:, condition_index],
        )
        effective = np.minimum(requested[output_index], 0.95 * distances)
        if np.any(effective <= 1e-8):
            raise RuntimeError("a condition is too close to its bound for central differences")
        plus = physical.copy()
        minus = physical.copy()
        plus[:, condition_index] += effective
        minus[:, condition_index] -= effective
        name = equation.condition_names[condition_index]
        LOGGER.info("Solving +/-%s normalized sensitivity labels", name)
        plus_solution = solve_parallel(histories, plus.astype(np.float32), **common)
        minus_solution = solve_parallel(histories, minus.astype(np.float32), **common)
        derivative = (plus_solution.astype(np.float64) - minus_solution.astype(np.float64)) / (
            2.0 * effective[:, None, None]
        )
        result[..., output_index] = (
            equation.inverse_spans[output_index] * derivative
        ).astype(np.float32)
        del plus_solution, minus_solution
    if not np.isfinite(result).all():
        raise RuntimeError("sensitivity generation produced non-finite values")
    return result


def classify_outcomes(solutions: np.ndarray) -> dict[str, float]:
    late = solutions[:, -max(2, solutions.shape[1] // 5) :, :]
    tmax = solutions[:, :, 1].max(axis=1)
    conversion = 1.0 - late[:, :, 0].mean(axis=1)
    oscillation = late[:, :, 1].std(axis=1)
    safe = (tmax <= 0.60) & (conversion >= 0.50) & (oscillation <= 0.025)
    boundary = (
        (np.abs(tmax - 0.60) <= 0.03)
        | (np.abs(conversion - 0.50) <= 0.03)
        | (np.abs(oscillation - 0.025) <= 0.005)
    ) & ~safe
    unsafe = ~(safe | boundary)
    return {
        "safe_fraction": float(safe.mean()),
        "boundary_fraction": float(boundary.mean()),
        "unsafe_fraction": float(unsafe.mean()),
    }


def resolve_sizes(config: Mapping[str, Any], smoke: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    data = dict(config["data"])
    operator = dict(config["operator"])
    if smoke:
        small = config["smoke"]
        for key in data:
            if key in small:
                data[key] = small[key]
        for key in operator:
            if key in small:
                operator[key] = small[key]
    return data, operator


def dataset_signature(config: Mapping[str, Any], smoke: bool) -> str:
    data, _ = resolve_sizes(config, smoke)
    payload = {
        "format": DATASET_FORMAT,
        "data_seed": int(config["randomness"]["data_seed"]),
        "equation": config["equation"],
        "data": data,
        "smoke": bool(smoke),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def generate_dataset(
    config: Mapping[str, Any], workers: int, smoke: bool
) -> tuple[dict[str, DatasetSplit], dict[str, Any]]:
    data, _ = resolve_sizes(config, smoke)
    rng = np.random.default_rng(int(config["randomness"]["data_seed"]))
    equation = CSTRConfig.from_mapping(config["equation"])
    history_count = int(data["history_count"])
    validation_count = int(data["validation_count"])
    test_count = int(data["test_count"])
    total_histories = history_count + validation_count + test_count
    history_pool, diagnostics = generate_physical_history_pool(
        total_histories, equation, data, rng
    )
    train_history_pool = history_pool[:history_count]
    validation_histories = history_pool[history_count : history_count + validation_count]
    test_histories = history_pool[-test_count:]

    condition_pool = sample_conditions(int(data["condition_count"]), equation, rng)
    history_indices, condition_indices = balanced_pair_indices(
        history_count,
        condition_pool.shape[0],
        int(data["pairs_per_history"]),
        rng,
    )
    train_histories = train_history_pool[history_indices]
    train_conditions = condition_pool[condition_indices]
    validation_conditions = sample_conditions(validation_count, equation, rng)
    test_conditions = sample_conditions(test_count, equation, rng)
    history_times = np.linspace(-equation.delay, 0.0, int(data["history_sensors"]), dtype=np.float32)
    output_times = np.linspace(0.0, float(data["prediction_horizon"]), int(data["output_points"]), dtype=np.float32)
    common = {
        "horizon": float(data["prediction_horizon"]),
        "output_points": int(data["output_points"]),
        "step": float(data["internal_step"]),
        "equation_values": config["equation"],
        "workers": workers,
        "chunk_size": int(data["solver_chunk_size"]),
    }
    splits: dict[str, DatasetSplit] = {}
    for name, histories, conditions in (
        ("train", train_histories, train_conditions),
        ("validation", validation_histories, validation_conditions),
        ("test", test_histories, test_conditions),
    ):
        LOGGER.info("Generating %s reference trajectories", name)
        solutions = solve_parallel(histories, conditions, **common)
        sensitivities = None
        if name == "train":
            sensitivities = generate_normalized_sensitivities(
                histories, conditions, data, config["equation"], workers
            )
        splits[name] = DatasetSplit(
            histories.astype(np.float32),
            conditions.astype(np.float32),
            solutions,
            history_times,
            output_times,
            sensitivities,
        )
        diagnostics.update(
            {f"{name}_{key}": value for key, value in classify_outcomes(solutions).items()}
        )
    diagnostics.update(
        {
            "train_rows": int(train_histories.shape[0]),
            "validation_rows": validation_count,
            "test_rows": test_count,
            "dataset_signature": dataset_signature(config, smoke),
            "dataset_format": DATASET_FORMAT,
        }
    )
    return splits, diagnostics


def _save_split(path: Path, split: DatasetSplit) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        values: dict[str, np.ndarray] = {
            "histories": split.histories,
            "conditions": split.conditions,
            "solutions": split.solutions,
            "history_times": split.history_times,
            "output_times": split.output_times,
        }
        if split.normalized_sensitivities is not None:
            values["normalized_sensitivities"] = split.normalized_sensitivities
        np.savez(handle, **values)
    temporary.replace(path)


def _load_split(path: Path) -> DatasetSplit:
    with np.load(path) as values:
        sensitivities = (
            values["normalized_sensitivities"].copy()
            if "normalized_sensitivities" in values
            else None
        )
        return DatasetSplit(
            values["histories"].copy(),
            values["conditions"].copy(),
            values["solutions"].copy(),
            values["history_times"].copy(),
            values["output_times"].copy(),
            sensitivities,
        )


def generate_or_load_dataset(
    config: Mapping[str, Any], cache_dir: Path, workers: int, smoke: bool
) -> tuple[dict[str, DatasetSplit], dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "metadata.json"
    signature = dataset_signature(config, smoke)
    split_paths = {name: cache_dir / f"{name}.npz" for name in ("train", "validation", "test")}
    if metadata_path.exists() and all(path.exists() for path in split_paths.values()):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("dataset_signature") == signature:
            LOGGER.info("Loading compatible cached dataset from %s", cache_dir)
            return {name: _load_split(path) for name, path in split_paths.items()}, metadata
        raise RuntimeError("existing dataset cache has an incompatible signature")
    splits, metadata = generate_dataset(config, workers, smoke)
    for name, split in splits.items():
        _save_split(split_paths[name], split)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return splits, metadata


def normalization_statistics(train: DatasetSplit) -> dict[str, np.ndarray]:
    return {
        "history_mean": train.histories.mean(axis=(0, 2)).astype(np.float32),
        "history_std": (train.histories.std(axis=(0, 2)) + 1e-6).astype(np.float32),
        "output_mean": train.solutions.mean(axis=(0, 1)).astype(np.float32),
        "output_std": (train.solutions.std(axis=(0, 1)) + 1e-6).astype(np.float32),
    }
