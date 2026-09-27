#!/usr/bin/env python3
"""Benchmark online solution time of one pretrained Nicholson operator and RK4."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from pathlib import Path
import platform
import socket
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401  # registers the SciencePlots styles
import torch


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
PARENT_PROJECT_DIR = BENCHMARK_DIR.parent
SOURCE_PROJECT_DIR = PARENT_PROJECT_DIR / "four_methods_normalized_sensitivity_5seeds"
SOURCE_CODE_DIR = SOURCE_PROJECT_DIR / "code"
if not SOURCE_CODE_DIR.is_dir():
    raise RuntimeError(f"required normalized-sensitivity source is missing: {SOURCE_CODE_DIR}")
for path_value in (str(SOURCE_CODE_DIR), str(SOURCE_PROJECT_DIR)):
    while path_value in sys.path:
        sys.path.remove(path_value)
sys.path.insert(0, str(SOURCE_CODE_DIR))
sys.path.insert(1, str(SOURCE_PROJECT_DIR))

from data import sample_histories, sample_parameters, solve_parallel
from equation import NicholsonConfig
from model import load_operator_checkpoint


LOGGER = logging.getLogger("nicholson_online_speed")
EXPECTED_MODEL_TYPE = "separate_parameter_shared"
METHOD_OPERATOR = "Pretrained Separate-Parameter MIONet"
METHOD_SOLVER = "Direct numerical DDE solver"

plt.style.use(["science", "no-latex", "grid"])
matplotlib.rcParams['font.family'] = 'Source Han Serif'
matplotlib.rcParams["axes.unicode_minus"] = False


def configure_logging(path: Path) -> None:
    """Send benchmark messages to both the terminal and one persistent log.

    ``path`` is the requested log location, for example ``outputs/run/benchmark.log``.
    The function returns ``None``; it creates the parent directory and replaces the
    current process's root logging handlers.  A typical visible line is
    ``Case count 100 completed``.  ``main`` calls it once before loading the model.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(processName)s | %(message)s"
    )
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(logging.INFO)
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(stream)
    root.addHandler(file_handler)


def load_json(path: Path) -> dict[str, Any]:
    """Load one UTF-8 JSON object without modifying the source file.

    ``path`` may be ``configs/benchmark.json``.  The returned mutable dictionary
    contains timing settings; for example ``result['repeats'] == 5`` formally.
    Non-object JSON raises ``ValueError``.  ``main`` calls it before CLI overrides.
    """
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("configuration root must be a JSON object")
    return values


def save_json(path: Path, values: Mapping[str, Any]) -> None:
    """Atomically save a machine-readable JSON mapping.

    ``path`` is an output such as ``summary.json`` and ``values`` is JSON-safe.
    The function returns ``None``; it creates parents and replaces through a hidden
    temporary file.  For example ``{'success': True}`` is written exactly once.
    ``run_stage`` and ``main`` use it for config, status, environment and summaries.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(values), indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def save_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Atomically save nonempty timing rows with a stable header.

    ``path`` is normally ``raw_timings.csv`` or ``timing_summary.csv`` and every
    mapping in ``rows`` has identical keys.  The return is ``None``; parents are
    created and a temporary file is replaced.  A two-row input creates a header and
    two data records.  ``run_stage`` calls it after all repetitions are complete.
    """
    if not rows:
        raise ValueError("cannot save an empty CSV")
    fieldnames = list(rows[0].keys())
    if any(list(row.keys()) != fieldnames for row in rows):
        raise ValueError("all CSV rows must use the same ordered fields")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def validate_config(config: Mapping[str, Any]) -> None:
    """Reject an invalid timing protocol before expensive model or solver work.

    ``config`` contains the formal and smoke case counts, repetitions, warmup,
    sampling seed, batch size and figure formats.  It returns ``None`` for the
    requested formal counts ``[1,10,50,100,500,1000,2048,4096]`` and raises
    ``ValueError`` for duplicates, nonpositive counts or unsupported formats.
    ``main`` calls it after resolving the checkpoint override.
    """
    if str(config.get("model_type")) != EXPECTED_MODEL_TYPE:
        raise ValueError(f"model_type must be {EXPECTED_MODEL_TYPE}")
    counts = [int(value) for value in config.get("case_counts", ())]
    if not counts or counts != sorted(set(counts)) or counts[0] < 1:
        raise ValueError("case_counts must be unique, positive and increasing")
    if counts != [1, 10, 50, 100, 500, 1000, 2048, 4096]:
        raise ValueError("formal case_counts must be [1,10,50,100,500,1000,2048,4096]")
    positive = (
        int(config["repeats"]), int(config["warmup_iterations"]),
        int(config["operator_batch_size"]), int(config["solver_chunk_size"]),
        int(config["accuracy_reference_step_factor"]),
    )
    if any(value < 1 for value in positive):
        raise ValueError("repeats, warmup, batch size and chunk size must be positive")
    if int(config["accuracy_reference_step_factor"]) <= 1:
        raise ValueError("accuracy_reference_step_factor must be greater than one")
    smoke = config["smoke"]
    if (
        not smoke.get("case_counts")
        or any(int(value) < 1 for value in smoke["case_counts"])
        or int(smoke["repeats"]) < 1
        or int(smoke["warmup_iterations"]) < 1
    ):
        raise ValueError("invalid smoke protocol")
    formats = {str(value).lower() for value in config["reporting"]["figure_formats"]}
    if not formats or not formats <= {"png", "jpg", "jpeg", "svg"}:
        raise ValueError("figures are limited to png/jpg/svg")


def prepare_inputs(
    count: int, scientific_config: Mapping[str, Any], sampling_seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample deterministic out-of-training histories and parameter vectors.

    ``count`` is the largest benchmark size, ``scientific_config`` comes from the
    checkpoint, and ``sampling_seed`` controls NumPy draws.  The return is histories
    ``[count,4,M]``, parameters ``[count,2]``, history times ``[M]`` and query times
    ``[Q]``.  For formal input, ``M=101`` and ``Q=401``.  RNG state is local and no
    trajectory is solved here, so preparation is excluded from both timing curves.
    ``run_stage`` calls it once and uses nested prefixes for every case count.
    """
    data = scientific_config["data"]
    equation = NicholsonConfig.from_mapping(scientific_config["equation"])
    history_times = np.linspace(
        -equation.maximum_history,
        0.0,
        int(data["history_sensors"]),
        dtype=np.float64,
    )
    output_times = np.linspace(
        0.0,
        float(data["horizon"]),
        int(data["output_points"]),
        dtype=np.float32,
    )
    rng = np.random.default_rng(int(sampling_seed))
    histories = sample_histories(
        count,
        history_times,
        tuple(float(value) for value in data["history_mean"]),
        float(data["history_sigma"]),
        float(data["history_length_scale"]),
        tuple(float(value) for value in data["history_bounds"]),
        rng,
    )
    parameters = sample_parameters(count, equation, rng)
    return histories, parameters, history_times, output_times


def synchronize(device: torch.device) -> None:
    """Wait for queued CUDA work when and only when the benchmark uses CUDA.

    ``device`` is ``cuda:0`` or ``cpu``.  The return is ``None``; CUDA synchronization
    is its only side effect.  For CPU it is a no-op.  ``warm_up_operator`` and
    ``time_operator`` call it so wall-clock measurements include completed kernels.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def shared_time_operator_forward(
    model: torch.nn.Module,
    histories: torch.Tensor,
    parameters: torch.Tensor,
    times: torch.Tensor,
    shared_trunk: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate the separate-parameter operator with one shared Trunk encoding.

    Benchmark query times are identical for every trajectory, so the Trunk is
    evaluated once as ``[Q,p]`` instead of repeating the same work as ``[B,Q,p]``.
    The checkpoint and mathematical operator are unchanged.  Parameter normalization
    is applied when the loaded project model provides it.
    """
    if getattr(model, "model_type", None) != EXPECTED_MODEL_TYPE:
        raise RuntimeError("shared-time inference requires separate_parameter_shared")
    if times.ndim != 1:
        raise ValueError("shared-time inference requires one common [Q] time grid")
    if histories.ndim != 3 or parameters.ndim != 2:
        raise ValueError("invalid operator input rank")
    batch, state_dim = histories.shape[:2]
    if parameters.shape[0] != batch:
        raise ValueError("history and parameter batch sizes differ")

    trunk = (
        model.time_trunk(model.time_features(times))
        if shared_trunk is None else shared_trunk
    )
    encoded = [
        1.0 + branch(histories[:, component, :]).reshape(
            batch, state_dim, model.latent_dim
        )
        for component, branch in enumerate(model.history_branches)
    ]
    state_features = encoded[0]
    for values in encoded[1:]:
        state_features = state_features * values

    normalizer = getattr(model, "normalize_parameters", None)
    parameter_inputs = normalizer(parameters) if callable(normalizer) else parameters
    parameter_features = model.parameter_branch(parameter_inputs)
    state_features = state_features * (1.0 + parameter_features.unsqueeze(1))
    raw = torch.einsum("bsp,qp->bqs", state_features, trunk)
    return raw * model.latent_scale + model.output_bias


def warm_up_operator(
    model: torch.nn.Module,
    histories: np.ndarray,
    parameters: np.ndarray,
    output_times: np.ndarray,
    device: torch.device,
    iterations: int,
) -> None:
    """Warm CUDA kernels and allocator without producing a reported measurement.

    Inputs contain one representative batch, a frozen operator, query grid, device
    and positive warmup count.  The function returns ``None``; for example 20 passes
    eliminate first-call CUDA initialization from formal timing.  Outputs are checked
    for shape and finiteness, then discarded.  ``run_stage`` calls it before timing.
    """
    model.eval()
    expected_shape = (histories.shape[0], output_times.size, 4)
    with torch.inference_mode():
        validation_count = min(2, histories.shape[0])
        validation_histories = torch.from_numpy(histories[:validation_count]).to(device)
        validation_parameters = torch.from_numpy(parameters[:validation_count]).to(device)
        validation_times = torch.from_numpy(output_times).to(device)
        reference = model(validation_histories, validation_parameters, validation_times)
        optimized = shared_time_operator_forward(
            model, validation_histories, validation_parameters, validation_times
        )
        torch.testing.assert_close(optimized, reference, rtol=1.0e-5, atol=1.0e-6)
        shared_trunk = model.time_trunk(model.time_features(validation_times))
        for _ in range(iterations):
            history_tensor = torch.from_numpy(histories).to(device)
            parameter_tensor = torch.from_numpy(parameters).to(device)
            time_tensor = torch.from_numpy(output_times).to(device)
            prediction = shared_time_operator_forward(
                model, history_tensor, parameter_tensor, time_tensor, shared_trunk
            )
            if tuple(prediction.shape) != expected_shape or not torch.isfinite(prediction).all():
                raise RuntimeError("operator warmup produced invalid output")
    synchronize(device)


def time_operator(
    model: torch.nn.Module,
    histories: np.ndarray,
    parameters: np.ndarray,
    output_times: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float]:
    """Measure one complete online operator evaluation including device transfers.

    Histories ``[N,4,M]``, parameters ``[N,2]`` and times ``[Q]`` are processed in
    batches no larger than ``batch_size``.  The return is ``(seconds, checksum)``;
    for example a valid call may return ``(0.08, 1234.5)``.  The timed region includes
    CPU-to-device inputs, forward inference, synchronization and device-to-CPU output,
    but excludes model loading and sampling.  ``run_stage`` calls it five times per N.
    """
    started = time.perf_counter()
    checksum = 0.0
    with torch.inference_mode():
        time_tensor = torch.from_numpy(output_times).to(device)
        shared_trunk = model.time_trunk(model.time_features(time_tensor))
        for start in range(0, histories.shape[0], batch_size):
            stop = min(start + batch_size, histories.shape[0])
            history_tensor = torch.from_numpy(histories[start:stop]).to(device)
            parameter_tensor = torch.from_numpy(parameters[start:stop]).to(device)
            prediction = shared_time_operator_forward(
                model, history_tensor, parameter_tensor, time_tensor, shared_trunk
            )
            host_prediction = prediction.detach().cpu()
            if tuple(host_prediction.shape) != (stop - start, output_times.size, 4):
                raise RuntimeError("operator returned an unexpected shape")
            if not torch.isfinite(host_prediction).all():
                raise RuntimeError("operator returned NaN or Inf")
            checksum += float(host_prediction.double().sum())
    synchronize(device)
    elapsed = time.perf_counter() - started
    if elapsed <= 0.0 or not math.isfinite(checksum):
        raise RuntimeError("invalid operator timing or checksum")
    return elapsed, checksum


def time_numerical_solver(
    histories: np.ndarray,
    parameters: np.ndarray,
    history_times: np.ndarray,
    output_times: np.ndarray,
    scientific_config: Mapping[str, Any],
    workers: int,
    chunk_size: int,
) -> tuple[float, float]:
    """Measure one direct RK4 solution call including its multiprocessing pool.

    Inputs define exactly the same N histories, parameters and grids used by the
    operator.  ``workers`` is the CPU-process budget and ``chunk_size`` is the parent
    solver batch size.  The return ``(seconds,checksum)`` includes process startup,
    numerical integration and result collection; e.g. ``(12.4,1234.6)``.  Input
    sampling and file output remain excluded.  ``run_stage`` calls this per repeat.
    """
    started = time.perf_counter()
    solution = solve_parallel(
        histories,
        parameters,
        history_times,
        output_times,
        scientific_config["equation"],
        float(scientific_config["data"]["internal_step"]),
        workers,
        chunk_size,
    )
    elapsed = time.perf_counter() - started
    if tuple(solution.shape) != (histories.shape[0], output_times.size, 4):
        raise RuntimeError("numerical solver returned an unexpected shape")
    checksum = float(np.asarray(solution, dtype=np.float64).sum())
    if elapsed <= 0.0 or not math.isfinite(checksum):
        raise RuntimeError("invalid numerical-solver timing or checksum")
    return elapsed, checksum


def evaluate_accuracy(
    model: torch.nn.Module,
    histories: np.ndarray,
    parameters: np.ndarray,
    history_times: np.ndarray,
    output_times: np.ndarray,
    scientific_config: Mapping[str, Any],
    device: torch.device,
    operator_batch_size: int,
    workers: int,
    solver_chunk_size: int,
    reference_step_factor: int,
    case_counts: Sequence[int],
) -> tuple[list[dict[str, float | int]], float]:
    """Compare both online methods with one refined-step RK4 reference.

    Accuracy work is deliberately outside every timed region.  The largest nested
    input set is evaluated once; smaller N values use prefixes of the same arrays.
    The refined RK4 step is the configured online step divided by
    ``reference_step_factor`` and is a numerical reference, not an exact solution.
    """
    started = time.perf_counter()
    operator_parts: list[np.ndarray] = []
    with torch.inference_mode():
        time_tensor = torch.from_numpy(output_times).to(device)
        shared_trunk = model.time_trunk(model.time_features(time_tensor))
        for start in range(0, histories.shape[0], operator_batch_size):
            stop = min(start + operator_batch_size, histories.shape[0])
            prediction = shared_time_operator_forward(
                model,
                torch.from_numpy(histories[start:stop]).to(device),
                torch.from_numpy(parameters[start:stop]).to(device),
                time_tensor,
                shared_trunk,
            )
            operator_parts.append(prediction.detach().cpu().numpy())
    operator_solution = np.concatenate(operator_parts, axis=0)
    online_step = float(scientific_config["data"]["internal_step"])
    solver_solution = solve_parallel(
        histories, parameters, history_times, output_times,
        scientific_config["equation"], online_step, workers, solver_chunk_size,
    )
    reference_solution = solve_parallel(
        histories, parameters, history_times, output_times,
        scientific_config["equation"], online_step / reference_step_factor,
        workers, solver_chunk_size,
    )

    rows: list[dict[str, float | int]] = []
    for count in case_counts:
        reference = np.asarray(reference_solution[:count], dtype=np.float64)
        denominator = max(float(np.linalg.norm(reference.ravel())), np.finfo(float).tiny)
        operator_error = np.asarray(operator_solution[:count], dtype=np.float64) - reference
        solver_error = np.asarray(solver_solution[:count], dtype=np.float64) - reference
        rows.append({
            "case_count": int(count),
            "operator_relative_l2": float(np.linalg.norm(operator_error.ravel()) / denominator),
            "operator_mse": float(np.mean(operator_error ** 2)),
            "rk4_relative_l2": float(np.linalg.norm(solver_error.ravel()) / denominator),
            "rk4_mse": float(np.mean(solver_error ** 2)),
        })
    return rows, time.perf_counter() - started


def summarize_timings(
    raw_rows: Sequence[Mapping[str, Any]], case_counts: Sequence[int]
) -> list[dict[str, Any]]:
    """Compute timing means, sample standard deviations, throughput and speedup.

    ``raw_rows`` contains one method/N/repeat measurement and ``case_counts`` gives
    output order.  The return has one row per N; a row includes operator and solver
    means plus ``speedup = solver_mean/operator_mean``.  With one smoke repeat,
    standard deviations are zero.  The function has no side effect.  ``run_stage``
    calls it before saving the summary and plot.
    """
    result: list[dict[str, Any]] = []
    for count in case_counts:
        by_method: dict[str, list[float]] = {}
        for method in (METHOD_OPERATOR, METHOD_SOLVER):
            values = [
                float(row["wall_time_seconds"])
                for row in raw_rows
                if int(row["case_count"]) == int(count) and row["method"] == method
            ]
            if not values:
                raise RuntimeError(f"missing timings for {method}, N={count}")
            by_method[method] = values
        operator_values = by_method[METHOD_OPERATOR]
        solver_values = by_method[METHOD_SOLVER]
        operator_mean = statistics.fmean(operator_values)
        solver_mean = statistics.fmean(solver_values)
        result.append({
            "case_count": int(count),
            "repeat_count": len(operator_values),
            "operator_time_mean_seconds": operator_mean,
            "operator_time_sample_std_seconds": (
                statistics.stdev(operator_values) if len(operator_values) > 1 else 0.0
            ),
            "solver_time_mean_seconds": solver_mean,
            "solver_time_sample_std_seconds": (
                statistics.stdev(solver_values) if len(solver_values) > 1 else 0.0
            ),
            "operator_cases_per_second": float(count) / operator_mean,
            "solver_cases_per_second": float(count) / solver_mean,
            "speedup_solver_over_operator": solver_mean / operator_mean,
        })
    return result


def plot_timings(
    rows: Sequence[Mapping[str, Any]], output_dir: Path, reporting: Mapping[str, Any]
) -> list[Path]:
    """Plot online time and refined-reference accuracy in two panels.

    ``rows`` is the summarized timing table, ``output_dir`` receives ``figures``, and
    ``reporting`` supplies PNG/SVG formats and DPI.  The returned paths are the files
    written, e.g. ``online_solution_time.png`` and ``.svg``.  Both axes are logarithmic
    and error bars show five-repeat sample standard deviation.  ``run_stage`` calls it.
    """
    counts = np.asarray([int(row["case_count"]) for row in rows])
    operator_mean = np.asarray([float(row["operator_time_mean_seconds"]) for row in rows])
    operator_std = np.asarray([
        float(row["operator_time_sample_std_seconds"]) for row in rows
    ])
    solver_mean = np.asarray([float(row["solver_time_mean_seconds"]) for row in rows])
    solver_std = np.asarray([
        float(row["solver_time_sample_std_seconds"]) for row in rows
    ])
    figure, (time_axis, error_axis) = plt.subplots(1, 2, figsize=(11.2, 4.4))
    time_axis.errorbar(
        counts, operator_mean, yerr=operator_std, marker="o", capsize=3,
        linewidth=1.6, label=METHOD_OPERATOR,
    )
    time_axis.errorbar(
        counts, solver_mean, yerr=solver_std, marker="s", capsize=3,
        linewidth=1.6, label=METHOD_SOLVER,
    )
    time_axis.set_xscale("log")
    time_axis.set_yscale("log")
    time_axis.set_xlabel("Number of trajectories solved")
    time_axis.set_ylabel("Wall-clock time (s)")
    time_axis.set_title("Online solution time")
    time_axis.legend()
    error_axis.plot(
        counts, np.maximum(
            [float(row["operator_relative_l2"]) for row in rows], np.finfo(float).tiny
        ),
        marker="o", linewidth=1.6, label=METHOD_OPERATOR,
    )
    error_axis.plot(
        counts, np.maximum(
            [float(row["rk4_relative_l2"]) for row in rows], np.finfo(float).tiny
        ),
        marker="s", linewidth=1.6, label=METHOD_SOLVER,
    )
    error_axis.set_xscale("log")
    error_axis.set_yscale("log")
    error_axis.set_xlabel("Number of trajectories evaluated")
    error_axis.set_ylabel(r"Relative $L^2$ error")
    error_axis.set_title("Accuracy against refined-step RK4")
    error_axis.legend()
    figure.tight_layout()
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for extension in reporting["figure_formats"]:
        path = figure_dir / f"online_solution_time.{str(extension).lower()}"
        figure.savefig(path, dpi=int(reporting["raster_dpi"]), bbox_inches="tight")
        paths.append(path)
    plt.close(figure)
    return paths


def run_stage(
    stage_dir: Path,
    config: Mapping[str, Any],
    scientific_config: Mapping[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    workers: int,
    smoke: bool,
) -> dict[str, Any]:
    """Execute one smoke or formal benchmark through the identical timing functions.

    ``stage_dir`` receives outputs; configs define protocol/equation; ``model`` is
    the loaded frozen Separate-Parameter MIONet; ``device`` and ``workers`` define
    hardware; ``smoke`` selects two tiny counts or all eight formal counts.  The
    returned summary reports timing only, such as the maximum-N speedup and figure
    paths.  It writes sampled inputs, CSV, JSON and plots.  ``main`` runs smoke first
    and enters formal timing only after smoke succeeds.
    """
    protocol = config["smoke"] if smoke else config
    case_counts = [int(value) for value in protocol["case_counts"]]
    repeats = int(protocol["repeats"])
    warmup_iterations = int(protocol["warmup_iterations"])
    maximum_count = max(case_counts)
    histories, parameters, history_times, output_times = prepare_inputs(
        maximum_count,
        scientific_config,
        int(config["sampling_seed"]),
    )
    stage_dir.mkdir(parents=True, exist_ok=True)
    with (stage_dir / "benchmark_inputs.npz").open("wb") as handle:
        np.savez_compressed(
            handle,
            histories=histories,
            parameters=parameters,
            history_times=history_times,
            output_times=output_times,
        )
    warmup_count = min(int(config["operator_batch_size"]), maximum_count)
    warm_up_operator(
        model,
        histories[:warmup_count],
        parameters[:warmup_count],
        output_times,
        device,
        warmup_iterations,
    )
    raw_rows: list[dict[str, Any]] = []
    for count in case_counts:
        selected_histories = histories[:count]
        selected_parameters = parameters[:count]
        for repeat in range(1, repeats + 1):
            operator_seconds, operator_checksum = time_operator(
                model,
                selected_histories,
                selected_parameters,
                output_times,
                device,
                int(config["operator_batch_size"]),
            )
            raw_rows.append({
                "method": METHOD_OPERATOR,
                "case_count": count,
                "repeat": repeat,
                "wall_time_seconds": operator_seconds,
                "cases_per_second": count / operator_seconds,
                "output_checksum": operator_checksum,
            })
            solver_seconds, solver_checksum = time_numerical_solver(
                selected_histories,
                selected_parameters,
                history_times,
                output_times,
                scientific_config,
                workers,
                int(config["solver_chunk_size"]),
            )
            raw_rows.append({
                "method": METHOD_SOLVER,
                "case_count": count,
                "repeat": repeat,
                "wall_time_seconds": solver_seconds,
                "cases_per_second": count / solver_seconds,
                "output_checksum": solver_checksum,
            })
            LOGGER.info(
                "N=%d | repeat=%d/%d | operator=%.6f s | solver=%.6f s | speedup=%.2fx",
                count, repeat, repeats, operator_seconds, solver_seconds,
                solver_seconds / operator_seconds,
            )
    summary_rows = summarize_timings(raw_rows, case_counts)
    accuracy_rows, accuracy_seconds = evaluate_accuracy(
        model, histories, parameters, history_times, output_times, scientific_config,
        device, int(config["operator_batch_size"]), workers,
        int(config["solver_chunk_size"]),
        int(config["accuracy_reference_step_factor"]), case_counts,
    )
    for timing_row, accuracy_row in zip(summary_rows, accuracy_rows, strict=True):
        if int(timing_row["case_count"]) != int(accuracy_row["case_count"]):
            raise RuntimeError("timing and accuracy case counts differ")
        timing_row.update({key: value for key, value in accuracy_row.items() if key != "case_count"})
    save_csv(stage_dir / "raw_timings.csv", raw_rows)
    save_csv(stage_dir / "accuracy_summary.csv", accuracy_rows)
    save_csv(stage_dir / "timing_summary.csv", summary_rows)
    figure_paths = plot_timings(summary_rows, stage_dir, config["reporting"])
    summary = {
        "stage": "smoke" if smoke else "full",
        "accuracy_reported": True,
        "accuracy_reference": "refined-step causal RK4",
        "accuracy_reference_step_factor": int(config["accuracy_reference_step_factor"]),
        "accuracy_evaluation_time_included": False,
        "accuracy_evaluation_seconds": float(accuracy_seconds),
        "offline_training_time_included": False,
        "checkpoint_loading_time_included": False,
        "input_sampling_time_included": False,
        "operator_input_output_transfer_included": True,
        "operator_shared_time_trunk": True,
        "operator_batch_size": int(config["operator_batch_size"]),
        "solver_pool_startup_included": True,
        "case_counts": case_counts,
        "repeats": repeats,
        "warmup_iterations": warmup_iterations,
        "maximum_case_speedup": float(summary_rows[-1]["speedup_solver_over_operator"]),
        "figure_paths": [str(path) for path in figure_paths],
        "timings": summary_rows,
    }
    save_json(stage_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Create the benchmark's single explicit command-line interface.

    The returned parser accepts config, placeholder-or-real checkpoint, output,
    device, CPU budget and smoke-only mode.  For example parsing
    ``--checkpoint A --output-dir outputs/run --device cpu`` returns a namespace
    without touching files.  ``main`` is its sole caller.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=BENCHMARK_DIR / "configs" / "benchmark.json",
        help="Benchmark JSON configuration",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Best Separate-Parameter MIONet checkpoint; overrides config placeholder A",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Unique output directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="Logical cuda:0 or cpu")
    parser.add_argument("--cpus", type=int, default=1, help="Direct solver CPU-process budget")
    parser.add_argument(
        "--operator-batch-size", type=int, default=None,
        help="Override the configured operator inference batch size",
    )
    parser.add_argument("--smoke-only", action="store_true", help="Stop after the smoke stage")
    return parser


def main() -> int:
    """Run validated smoke and formal online-speed stages and return a shell status.

    CLI values identify the checkpoint, output, logical GPU and CPU budget.  The
    return is zero only after required CSV/JSON/PNG/SVG outputs exist; exceptions
    preserve a failed ``pipeline_status.json`` and full log traceback.  A successful
    formal example creates ``full/timing_summary.csv`` for eight case counts.
    ``run_speed_nohup.sh`` starts this function in the background.
    """
    args = build_parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir / "benchmark.log")
    config = load_json(args.config.expanduser().resolve())
    if args.checkpoint is not None:
        config["checkpoint_path"] = str(args.checkpoint)
    if args.operator_batch_size is not None:
        if args.operator_batch_size < 1:
            raise ValueError("--operator-batch-size must be positive")
        config["operator_batch_size"] = int(args.operator_batch_size)
    validate_config(config)
    checkpoint_text = str(config.get("checkpoint_path", ""))
    if not checkpoint_text or checkpoint_text == "A":
        raise FileNotFoundError(
            "checkpoint is still placeholder A; pass --checkpoint /absolute/path/best_model.pt"
        )
    checkpoint_path = Path(checkpoint_text).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if args.cpus < 1 or args.cpus > (os.cpu_count() or 1):
        raise ValueError(f"--cpus must be in [1,{os.cpu_count()}]")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    allowed_existing = {"nohup.log", "pipeline.pid", "benchmark.log"}
    existing = {path.name for path in output_dir.iterdir()}
    if existing - allowed_existing:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    status: dict[str, Any] = {"success": False, "stage": "loading_checkpoint"}
    save_json(output_dir / "pipeline_status.json", status)
    started = time.perf_counter()
    try:
        model, checkpoint = load_operator_checkpoint(checkpoint_path, device)
        if str(checkpoint.get("model_type")) != EXPECTED_MODEL_TYPE:
            raise RuntimeError(
                f"checkpoint model_type is {checkpoint.get('model_type')}, "
                f"required {EXPECTED_MODEL_TYPE}"
            )
        scientific_config = checkpoint.get("resolved_config")
        if not isinstance(scientific_config, Mapping):
            raise RuntimeError("checkpoint does not contain resolved_config")
        model.eval()
        config["checkpoint_path"] = str(checkpoint_path)
        config["checkpoint_model_type"] = str(checkpoint.get("model_type"))
        config["checkpoint_training_seed"] = int(checkpoint.get("training_seed", -1))
        config["checkpoint_best_iteration"] = int(checkpoint.get("best_iteration", -1))
        config["runtime"] = {
            "device": str(device),
            "cpu_processes": int(args.cpus),
            "output_dir": str(output_dir),
        }
        save_json(output_dir / "config.json", config)
        save_json(output_dir / "scientific_config.json", scientific_config)
        save_json(output_dir / "environment.json", {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "logical_cpu_count": os.cpu_count(),
        })
        LOGGER.info("Checkpoint: %s", checkpoint_path)
        LOGGER.info("Model: %s | device: %s | solver workers: %d", METHOD_OPERATOR, device, args.cpus)
        status["stage"] = "smoke"
        save_json(output_dir / "pipeline_status.json", status)
        smoke_summary = run_stage(
            output_dir / "smoke", config, scientific_config, model, device,
            int(args.cpus), True,
        )
        if args.smoke_only:
            status = {
                "success": True,
                "stage": "smoke_complete",
                "runtime_seconds": time.perf_counter() - started,
                "summary": smoke_summary,
            }
            save_json(output_dir / "pipeline_status.json", status)
            return 0
        status["stage"] = "full"
        save_json(output_dir / "pipeline_status.json", status)
        full_summary = run_stage(
            output_dir / "full", config, scientific_config, model, device,
            int(args.cpus), False,
        )
        status = {
            "success": True,
            "stage": "complete",
            "runtime_seconds": time.perf_counter() - started,
            "summary": full_summary,
        }
        save_json(output_dir / "pipeline_status.json", status)
        LOGGER.info("Benchmark complete in %.2f seconds", status["runtime_seconds"])
        return 0
    except Exception as error:
        status.update({
            "success": False,
            "error": repr(error),
            "runtime_seconds": time.perf_counter() - started,
        })
        save_json(output_dir / "pipeline_status.json", status)
        LOGGER.exception("Benchmark failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
