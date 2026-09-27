"""Independent finite-difference sensitivity references with resumable caching."""

from __future__ import annotations

import csv
import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np

from systems import SystemData, array_digest


LOGGER = logging.getLogger(__name__)


def _float_token(value: float) -> str:
    return f"{value:.10g}".replace("-", "m").replace(".", "p")


class ReferenceSolver:
    """Call one audited parallel solver and cache every expensive solve."""

    def __init__(
        self,
        data_module: Any,
        cache_dir: Path,
        workers: int,
        chunk_size: int,
    ) -> None:
        self.data_module = data_module
        self.cache_dir = cache_dir
        self.workers = int(workers)
        self.chunk_size = int(chunk_size)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def solve(
        self,
        data: SystemData,
        parameters: np.ndarray,
        internal_step: float,
        label: str,
    ) -> np.ndarray:
        """Solve or reuse trajectories for an exact input and numerical setting."""
        parameters = np.asarray(parameters, dtype=np.float64)
        digest = array_digest(
            data.indices.astype(np.int64), data.histories, parameters
        )
        path = self.cache_dir / f"solve_dt_{_float_token(internal_step)}_{digest}.npy"
        if path.is_file():
            LOGGER.info("Reusing reference cache: %s", path.name)
            values = np.load(path, allow_pickle=False)
        else:
            LOGGER.info(
                "Solving %s: cases=%d step=%.8g", label, parameters.shape[0], internal_step
            )
            values = self.data_module.solve_parallel(
                data.histories,
                parameters.astype(np.float32),
                data.history_times,
                data.output_times,
                data.equation_values,
                float(internal_step),
                self.workers,
                self.chunk_size,
            )
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.save(handle, np.asarray(values, dtype=np.float32), allow_pickle=False)
            temporary.replace(path)
        expected = (parameters.shape[0], data.output_times.size, data.solutions.shape[-1])
        if values.shape != expected or not np.isfinite(values).all():
            raise RuntimeError(f"invalid cached solution {path}: {values.shape}")
        return np.asarray(values, dtype=np.float64)

    def sensitivity(
        self,
        data: SystemData,
        parameters: np.ndarray,
        parameter_index: int,
        difference_step: float,
        internal_step: float,
        label: str,
    ) -> np.ndarray:
        """Return range-scaled central differences for one physical parameter."""
        bounds = data.parameter_bounds
        parameters = np.asarray(parameters, dtype=np.float64)
        distance = np.minimum(
            parameters[:, parameter_index] - bounds[parameter_index, 0],
            bounds[parameter_index, 1] - parameters[:, parameter_index],
        )
        if np.any(distance < difference_step - 1e-12):
            raise ValueError("central-difference request crosses a parameter bound")
        plus = parameters.copy()
        minus = parameters.copy()
        plus[:, parameter_index] += difference_step
        minus[:, parameter_index] -= difference_step
        plus_solution = self.solve(
            data,
            plus,
            internal_step,
            f"{label}_p{parameter_index}_plus_h_{_float_token(difference_step)}",
        )
        minus_solution = self.solve(
            data,
            minus,
            internal_step,
            f"{label}_p{parameter_index}_minus_h_{_float_token(difference_step)}",
        )
        span = bounds[parameter_index, 1] - bounds[parameter_index, 0]
        return span * (plus_solution - minus_solution) / (2.0 * difference_step)


def per_case_relative(first: np.ndarray, reference: np.ndarray, epsilon: float) -> np.ndarray:
    """Compute one relative Frobenius error for every leading case."""
    axes = tuple(range(1, reference.ndim))
    numerator = np.sqrt(np.sum(np.square(first - reference), axis=axes))
    denominator = np.sqrt(np.sum(np.square(reference), axis=axes))
    return numerator / np.maximum(denominator, epsilon)


def convergence_record(
    system: str,
    parameter: str,
    study: str,
    coarse_value: float,
    fine_value: float,
    coarse: np.ndarray,
    fine: np.ndarray,
    epsilon: float,
) -> dict[str, Any]:
    """Summarize one adjacent convergence-level comparison."""
    errors = per_case_relative(coarse, fine, epsilon)
    aggregate = float(
        np.linalg.norm(coarse - fine) / max(np.linalg.norm(fine), epsilon)
    )
    return {
        "system": system,
        "parameter": parameter,
        "study": study,
        "coarse_value": coarse_value,
        "fine_value": fine_value,
        "aggregate_relative_change": aggregate,
        "median_relative_change": float(np.median(errors)),
        "q25_relative_change": float(np.quantile(errors, 0.25)),
        "q75_relative_change": float(np.quantile(errors, 0.75)),
        "q95_relative_change": float(np.quantile(errors, 0.95)),
    }


def run_convergence_study(
    data: SystemData,
    solver: ReferenceSolver,
    specification: dict[str, Any],
    epsilon: float,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    """Check parameter-step and solver-step convergence and return finest references."""
    rows: list[dict[str, Any]] = []
    finest: list[np.ndarray] = []
    for parameter_index, parameter_name in enumerate(data.parameter_names):
        h_levels = [float(v) for v in specification["finite_difference_levels"][parameter_index]]
        dt_levels = [float(v) for v in specification["solver_step_levels"][parameter_index]]
        final_h = h_levels[-1]
        final_dt = dt_levels[-1]

        h_values = [
            solver.sensitivity(
                data,
                data.parameters,
                parameter_index,
                h,
                final_dt,
                "convergence_h",
            )
            for h in h_levels
        ]
        for comparison_index, (coarse_h, fine_h, coarse, fine) in enumerate(zip(
            h_levels[:-1], h_levels[1:], h_values[:-1], h_values[1:]
        )):
            row = convergence_record(
                    data.name,
                    parameter_name,
                    "finite_difference_step",
                    coarse_h,
                    fine_h,
                    coarse,
                    fine,
                    epsilon,
                )
            row["comparison_index"] = comparison_index
            row["is_final_refinement"] = comparison_index == len(h_levels) - 2
            rows.append(row)

        dt_values = [
            solver.sensitivity(
                data,
                data.parameters,
                parameter_index,
                final_h,
                dt,
                "convergence_dt",
            )
            for dt in dt_levels
        ]
        for comparison_index, (coarse_dt, fine_dt, coarse, fine) in enumerate(zip(
            dt_levels[:-1], dt_levels[1:], dt_values[:-1], dt_values[1:]
        )):
            row = convergence_record(
                    data.name,
                    parameter_name,
                    "solver_step",
                    coarse_dt,
                    fine_dt,
                    coarse,
                    fine,
                    epsilon,
                )
            row["comparison_index"] = comparison_index
            row["is_final_refinement"] = comparison_index == len(dt_levels) - 2
            rows.append(row)
        finest.append(dt_values[-1])
    return rows, finest


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write homogeneous dictionaries to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
