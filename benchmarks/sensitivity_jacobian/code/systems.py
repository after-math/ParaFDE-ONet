"""System specifications, dataset loading, and reproducible test selection."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SystemData:
    """Hold one selected held-out split and its mathematical metadata."""

    name: str
    display_name: str
    histories: np.ndarray
    parameters: np.ndarray
    solutions: np.ndarray
    history_times: np.ndarray
    output_times: np.ndarray
    indices: np.ndarray
    strata: np.ndarray | None
    parameter_names: tuple[str, ...]
    parameter_bounds: np.ndarray
    equation_values: dict[str, Any]

    @property
    def parameter_spans(self) -> np.ndarray:
        return self.parameter_bounds[:, 1] - self.parameter_bounds[:, 0]


def array_digest(*arrays: np.ndarray) -> str:
    """Return a stable short digest of shape, dtype, and byte content."""
    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(array)
        digest.update(str(values.shape).encode())
        digest.update(str(values.dtype).encode())
        digest.update(values.view(np.uint8))
    return digest.hexdigest()[:20]


def _interior_mask(
    parameters: np.ndarray, bounds: np.ndarray, maximum_steps: np.ndarray
) -> np.ndarray:
    lower_distance = parameters - bounds[:, 0]
    upper_distance = bounds[:, 1] - parameters
    return np.all(
        (lower_distance >= maximum_steps[None, :])
        & (upper_distance >= maximum_steps[None, :]),
        axis=1,
    )


def canonical_history_grid(
    archived_grid: np.ndarray, maximum_history: float
) -> np.ndarray:
    """Restore the exact uniform grid intended before float32 serialization.

    The formal archives store their grids as float32.  For a Nicholson history
    length of 1.6 this changes the first value to approximately
    -1.6000000238, while the original solver checks the configured endpoint with
    a 1e-10 tolerance.  We accept only a grid that agrees with the configured
    uniform grid to float32 precision, then return the exact float64 grid.
    """
    values = np.asarray(archived_grid, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("history grid must be one-dimensional")
    expected = np.linspace(
        -float(maximum_history), 0.0, values.size, dtype=np.float64
    )
    tolerance = 8.0 * np.finfo(np.float32).eps * max(1.0, abs(maximum_history))
    if not np.allclose(values, expected, rtol=0.0, atol=tolerance):
        maximum_error = float(np.max(np.abs(values - expected)))
        raise RuntimeError(
            "archived history grid differs from the configured uniform grid: "
            f"maximum error {maximum_error:.6g}"
        )
    return expected


def select_indices(
    parameters: np.ndarray,
    bounds: np.ndarray,
    finite_difference_levels: list[list[float]],
    count: int,
    seed: int,
    strata: np.ndarray | None = None,
) -> np.ndarray:
    """Choose a fixed interior subset, optionally balanced over history strata."""
    if count < 1:
        raise ValueError("case count must be positive")
    maximum_steps = np.asarray([levels[0] for levels in finite_difference_levels])
    eligible = np.flatnonzero(_interior_mask(parameters, bounds, maximum_steps))
    rng = np.random.default_rng(seed)
    if strata is None:
        if eligible.size < count:
            raise RuntimeError(f"only {eligible.size} interior cases are available")
        return np.sort(rng.choice(eligible, size=count, replace=False))

    strata_values = np.asarray(strata)
    unique = np.unique(strata_values)
    if count % unique.size:
        raise ValueError("balanced case count must be divisible by the stratum count")
    per_stratum = count // unique.size
    selected: list[np.ndarray] = []
    for value in unique:
        candidates = eligible[strata_values[eligible] == value]
        if candidates.size < per_stratum:
            raise RuntimeError(
                f"stratum {value!r} has only {candidates.size} interior cases"
            )
        selected.append(rng.choice(candidates, size=per_stratum, replace=False))
    return np.sort(np.concatenate(selected))


def load_selected_system(
    name: str,
    specification: dict[str, Any],
    selection_seed: int,
    count_override: int | None = None,
) -> SystemData:
    """Load the formal test archive and select one reproducible interior subset."""
    dataset_path = Path(specification["test_dataset"])
    config_path = Path(specification["resolved_config"])
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    resolved = json.loads(config_path.read_text(encoding="utf-8"))
    with np.load(dataset_path, allow_pickle=False) as archive:
        required = {
            "histories",
            "parameters",
            "solutions",
            "history_times",
            "output_times",
        }
        missing = required - set(archive.files)
        if missing:
            raise RuntimeError(f"test archive lacks keys: {sorted(missing)}")
        histories = np.asarray(archive["histories"], dtype=np.float32)
        parameters = np.asarray(archive["parameters"], dtype=np.float64)
        solutions = np.asarray(archive["solutions"], dtype=np.float32)
        history_times = canonical_history_grid(
            archive["history_times"],
            float(resolved["equation"]["maximum_history"]),
        )
        output_times = np.asarray(archive["output_times"], dtype=np.float64)
        strata = None
        strata_key = specification.get("strata_key")
        if specification.get("balanced_strata"):
            if strata_key not in archive.files:
                raise RuntimeError(f"balanced dataset lacks {strata_key!r}")
            strata = np.asarray(archive[strata_key])

    bounds = np.asarray(specification["parameter_bounds"], dtype=np.float64)
    count = int(count_override or specification["case_count"])
    indices = select_indices(
        parameters,
        bounds,
        specification["finite_difference_levels"],
        count,
        selection_seed,
        strata,
    )
    return SystemData(
        name=name,
        display_name=str(specification["display_name"]),
        histories=histories[indices],
        parameters=parameters[indices],
        solutions=solutions[indices],
        history_times=history_times,
        output_times=output_times,
        indices=indices,
        strata=None if strata is None else strata[indices],
        parameter_names=tuple(str(item) for item in specification["parameter_names"]),
        parameter_bounds=bounds,
        equation_values=dict(resolved["equation"]),
    )


def subset_system(data: SystemData, positions: np.ndarray) -> SystemData:
    """Return a position-based subset while preserving original test indices."""
    positions = np.asarray(positions, dtype=np.int64)
    return SystemData(
        name=data.name,
        display_name=data.display_name,
        histories=data.histories[positions],
        parameters=data.parameters[positions],
        solutions=data.solutions[positions],
        history_times=data.history_times,
        output_times=data.output_times,
        indices=data.indices[positions],
        strata=None if data.strata is None else data.strata[positions],
        parameter_names=data.parameter_names,
        parameter_bounds=data.parameter_bounds,
        equation_values=data.equation_values,
    )


def balanced_subset_positions(
    data: SystemData, count: int, seed: int
) -> np.ndarray:
    """Select positions within an already selected set, retaining stratum balance."""
    rng = np.random.default_rng(seed)
    positions = np.arange(data.parameters.shape[0])
    if data.strata is None:
        if count > positions.size:
            raise ValueError("subset is larger than selected data")
        return np.sort(rng.choice(positions, size=count, replace=False))
    unique = np.unique(data.strata)
    if count % unique.size:
        raise ValueError("balanced subset count must be divisible by strata")
    per_stratum = count // unique.size
    result = []
    for value in unique:
        candidates = positions[data.strata == value]
        result.append(rng.choice(candidates, size=per_stratum, replace=False))
    return np.sort(np.concatenate(result))


def save_selection(path: Path, data: SystemData) -> None:
    """Write original test indices and optional strata as a reproducibility record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["selected_position", "test_index"]
        if data.strata is not None:
            fieldnames.append("history_stratum")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for position, test_index in enumerate(data.indices):
            row: dict[str, Any] = {
                "selected_position": position,
                "test_index": int(test_index),
            }
            if data.strata is not None:
                value = data.strata[position]
                row["history_stratum"] = value.item() if hasattr(value, "item") else value
            writer.writerow(row)
