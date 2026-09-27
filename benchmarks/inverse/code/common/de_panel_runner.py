"""Direct causal-solver differential-evolution baseline."""

from __future__ import annotations

from pathlib import Path
import platform
import time
import traceback
from typing import Any, Mapping

import numpy as np

from common.io_utils import (
    read_json, sha256_file, sha256_mapping, write_csv, write_json,
)
from common.metrics import evaluate_estimate
from common.panel_archive import load_panel, select_panel_histories
from common.parameters import latin_hypercube
from systems import build_adapter


def _cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _observations(
    panel: Mapping[str, np.ndarray], case_index: int, sigma: float
) -> np.ndarray:
    indices = np.asarray(panel["observation_indices"], dtype=np.int64)
    reference = np.asarray(panel["reference"][case_index][:, indices, :])
    noise = np.asarray(panel["standard_normal_noise"][case_index])
    return (reference + float(sigma) * noise).astype(np.float64)


def _direct_objectives(
    adapter: Any,
    unit_candidates: np.ndarray,
    panel: Mapping[str, np.ndarray],
    observations: np.ndarray,
    internal_step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unit_candidates = np.asarray(unit_candidates, dtype=np.float64)
    if unit_candidates.ndim != 2 or unit_candidates.shape[1] != 2:
        raise ValueError("DE candidates must have shape (count, 2)")
    physical = np.asarray(
        adapter.unit_to_physical_numpy(unit_candidates), dtype=np.float64
    )
    histories = np.asarray(panel["histories"], dtype=np.float64)
    history_count = histories.shape[0]
    repeated_histories = np.tile(histories, (physical.shape[0], 1, 1))
    repeated_parameters = np.repeat(physical, history_count, axis=0)
    indices = np.asarray(panel["observation_indices"], dtype=np.int64)
    observation_times = np.asarray(panel["output_times"], dtype=np.float64)[indices]
    predicted = adapter.solve_batch(
        repeated_histories,
        repeated_parameters,
        np.asarray(panel["history_grid"], dtype=np.float64),
        observation_times,
        float(internal_step),
    ).reshape(
        physical.shape[0],
        history_count,
        observation_times.size,
        adapter.state_dim,
    )
    residuals = predicted.astype(np.float64) - observations[None]
    objectives = np.mean(residuals**2, axis=(1, 2, 3))
    if not np.isfinite(objectives).all():
        raise FloatingPointError("non-finite Direct DE objective")
    return objectives, predicted, physical


def _reflect_unit(values: np.ndarray) -> np.ndarray:
    """Reflect arbitrary real values into the closed unit interval."""
    wrapped = np.mod(np.asarray(values, dtype=np.float64), 2.0)
    return np.where(wrapped <= 1.0, wrapped, 2.0 - wrapped)


def _population_diameter(population: np.ndarray) -> float:
    differences = population[:, None, :] - population[None, :, :]
    return float(np.max(np.sqrt(np.sum(differences**2, axis=-1))))


def invert_de(
    adapter: Any,
    panel: Mapping[str, np.ndarray],
    observations: np.ndarray,
    config: Mapping[str, Any],
    seed: int,
) -> tuple[
    np.ndarray,
    float,
    list[dict[str, Any]],
    dict[str, Any],
    np.ndarray,
    np.ndarray,
]:
    if str(config["strategy"]) != "rand1bin":
        raise ValueError("only DE/rand/1/bin is supported")
    if str(config["coordinate_system"]) != "normalized":
        raise ValueError("Direct DE must search normalized coordinates")
    if str(config["initialization"]) != "latin_hypercube":
        raise ValueError("Direct DE requires Latin-hypercube initialization")
    if str(config["bound_handling"]) != "reflection":
        raise ValueError("Direct DE requires reflection bound handling")
    if bool(config["polish"]):
        raise ValueError("Direct DE must not use local polishing")

    population_size = int(config["population_size"])
    maximum_generations = int(config["maximum_generations"])
    maximum_evaluations = int(config["maximum_candidate_evaluations"])
    mutation = float(config["mutation"])
    recombination = float(config["recombination"])
    internal_step = float(config["internal_step"])
    expected_evaluations = population_size * (maximum_generations + 1)
    if population_size < 4:
        raise ValueError("DE/rand/1/bin requires at least four population members")
    if maximum_generations < 0 or maximum_evaluations != expected_evaluations:
        raise ValueError(
            "maximum_candidate_evaluations must equal population_size * "
            "(maximum_generations + 1)"
        )
    if not 0.0 < mutation <= 2.0 or not 0.0 <= recombination <= 1.0:
        raise ValueError("invalid DE mutation or recombination constant")

    rng = np.random.default_rng(seed)
    population = latin_hypercube(population_size, 2, rng)
    trace: list[dict[str, Any]] = []
    solver_batch_calls = 0
    candidate_evaluations = 0
    history_count = int(panel["histories"].shape[0])
    started = time.perf_counter()

    objectives, _, _ = _direct_objectives(
        adapter, population, panel, observations, internal_step
    )
    solver_batch_calls += 1
    candidate_evaluations += population_size

    def append_trace(generation: int) -> None:
        best_index = int(np.argmin(objectives))
        trace.append(
            {
                "generation": int(generation),
                "best_member": best_index,
                "best_observation_mse": float(objectives[best_index]),
                "mean_observation_mse": float(np.mean(objectives)),
                "median_observation_mse": float(np.median(objectives)),
                "objective_standard_deviation": float(np.std(objectives)),
                "normalized_population_diameter": _population_diameter(population),
                "candidate_objective_evaluations": int(candidate_evaluations),
                "elapsed_seconds": float(time.perf_counter() - started),
            }
        )

    append_trace(0)
    all_indices = np.arange(population_size)
    for generation in range(1, maximum_generations + 1):
        donors = np.empty_like(population)
        trials = np.empty_like(population)
        for target in range(population_size):
            eligible = np.delete(all_indices, target)
            first, second, third = rng.choice(eligible, size=3, replace=False)
            donors[target] = _reflect_unit(
                population[first]
                + mutation * (population[second] - population[third])
            )
            crossover = rng.random(2) < recombination
            crossover[int(rng.integers(0, 2))] = True
            trials[target] = np.where(crossover, donors[target], population[target])
        trial_objectives, _, _ = _direct_objectives(
            adapter, trials, panel, observations, internal_step
        )
        solver_batch_calls += 1
        candidate_evaluations += population_size
        accepted = trial_objectives < objectives
        population[accepted] = trials[accepted]
        objectives[accepted] = trial_objectives[accepted]
        append_trace(generation)

    if candidate_evaluations != maximum_evaluations:
        raise RuntimeError("Direct DE candidate-evaluation accounting mismatch")
    selected = int(np.argmin(objectives))
    physical_population = np.asarray(
        adapter.unit_to_physical_numpy(population), dtype=np.float64
    )
    elapsed = time.perf_counter() - started
    timing = {
        "warm_online_seconds": float(elapsed),
        "de_generations_completed": int(maximum_generations),
        "de_stop_reason": "maximum_candidate_evaluations",
        "solver_batch_calls": int(solver_batch_calls),
        "candidate_objective_evaluations": int(candidate_evaluations),
        "direct_trajectory_solves": int(candidate_evaluations * history_count),
    }
    return (
        physical_population[selected].copy(),
        float(objectives[selected]),
        trace,
        timing,
        population.copy(),
        objectives.copy(),
    )


def _result_path(panel_dir: Path, case_index: int, sigma: float) -> Path:
    noise_tag = str(float(sigma)).replace(".", "p")
    return (
        panel_dir / "jobs" / "de" / f"case_{case_index:03d}"
        / f"noise_{noise_tag}" / "result.json"
    )


def run_de_panel(
    panel_index: int,
    system_key: str,
    system_config: Mapping[str, Any],
    common_config: Mapping[str, Any],
    benchmark_root_text: str,
    source_root_text: str | None,
    archive_dir_text: str,
    full_dir_text: str,
    de_config: Mapping[str, Any],
    histories_per_request: int,
    resume: bool,
) -> list[dict[str, Any]]:
    print(f"[de] starting panel {panel_index:03d}", flush=True)
    adapter = build_adapter(
        system_key,
        system_config,
        Path(benchmark_root_text),
        Path(source_root_text) if source_root_text else None,
    )
    archive_dir = Path(archive_dir_text)
    full_dir = Path(full_dir_text)
    manifest = read_json(archive_dir / "manifest.json")
    panel = select_panel_histories(
        load_panel(archive_dir, panel_index), histories_per_request
    )
    panel_dir = full_dir / f"panel_{panel_index:03d}"
    panel_dir.mkdir(parents=True, exist_ok=True)
    thresholds = common_config["reporting"]["parameter_error_thresholds"]
    boundary_tolerance = float(
        common_config["reporting"]["boundary_tolerance_normalized"]
    )
    panel_sha = sha256_file(archive_dir / f"panel_{panel_index:03d}.npz")
    inverse_config_sha256 = sha256_mapping(de_config)
    rows: list[dict[str, Any]] = []
    request_index = 0
    for case_index in range(int(manifest["cases_per_panel"])):
        initialization_seed = (
            int(common_config["data"]["seed"])
            + panel_index * 100_003
            + case_index * 101
        )
        for sigma in manifest["noise_standard_deviations"]:
            result_path = _result_path(panel_dir, case_index, float(sigma))
            if resume and result_path.is_file():
                result = read_json(result_path)
                expected = {
                    "protocol_version": common_config["protocol_version"],
                    "system_key": system_key,
                    "method_key": "de",
                    "panel_index": int(panel_index),
                    "case_index": int(case_index),
                    "noise_standard_deviation": float(sigma),
                    "panel_archive_sha256": panel_sha,
                    "inverse_config_sha256": inverse_config_sha256,
                    "history_count": int(histories_per_request),
                }
                if any(result.get(key) != value for key, value in expected.items()):
                    raise RuntimeError(
                        f"incompatible completed Direct DE result: {result_path}"
                    )
                rows.append(result)
                request_index += 1
                continue
            result_path.parent.mkdir(parents=True, exist_ok=True)
            observations = _observations(panel, case_index, float(sigma))
            try:
                estimate, objective, trace, timing, population, objectives = invert_de(
                    adapter, panel, observations, de_config, initialization_seed
                )
                metrics = evaluate_estimate(
                    adapter,
                    panel,
                    case_index,
                    estimate,
                    float(manifest["truth_internal_step"]),
                    thresholds,
                    boundary_tolerance,
                )
                status = "completed"
                failure = None
            except Exception as error:
                status = "failed"
                failure = {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                }
                trace = []
                population = np.empty((0, 2), dtype=np.float64)
                objectives = np.empty((0,), dtype=np.float64)
                objective = float("nan")
                timing = {
                    "warm_online_seconds": float("nan"),
                    "de_generations_completed": 0,
                    "de_stop_reason": "numerical_failure",
                    "solver_batch_calls": 0,
                    "candidate_objective_evaluations": 0,
                    "direct_trajectory_solves": 0,
                }
                metrics = {
                    "true_parameters": np.asarray(
                        panel["parameters"][case_index], dtype=np.float64
                    ).tolist(),
                    "estimated_parameters": [float("nan"), float("nan")],
                    "parameter_normalized_rmse": float("nan"),
                    "parameter_physical_rmse": float("nan"),
                    "trajectory_relative_l2": float("nan"),
                    "boundary_estimate": 0,
                }
            result = {
                "protocol_version": common_config["protocol_version"],
                "system_key": system_key,
                "method_key": "de",
                "status": status,
                "failure": failure,
                "panel_index": int(panel_index),
                "case_index": int(case_index),
                "request_index_within_panel": int(request_index),
                "noise_standard_deviation": float(sigma),
                "history_count": int(panel["histories"].shape[0]),
                "archive_history_count": int(manifest["histories_per_panel"]),
                "history_indices_used": np.asarray(
                    panel["history_indices_used"], dtype=np.int64
                ).tolist(),
                "observation_count_per_history": int(manifest["observation_count"]),
                "scalar_observation_count": int(
                    panel["histories"].shape[0]
                    * manifest["observation_count"]
                    * manifest["state_dim"]
                ),
                "selected_observation_mse": float(objective),
                "panel_archive_sha256": panel_sha,
                "inverse_config_sha256": inverse_config_sha256,
                "compute_device": "cpu",
                "cpu_model": _cpu_name(),
                "de_initialization_seed": int(initialization_seed),
                "de_strategy": str(de_config["strategy"]),
                "de_population_size": int(de_config["population_size"]),
                "de_mutation": float(de_config["mutation"]),
                "de_recombination": float(de_config["recombination"]),
                "de_polish_used": 0,
                **timing,
                **metrics,
            }
            write_csv(result_path.parent / "de_trace.csv", trace)
            physical_population = (
                np.asarray(adapter.unit_to_physical_numpy(population), dtype=np.float64)
                if population.size else np.empty((0, 2), dtype=np.float64)
            )
            write_csv(
                result_path.parent / "final_population.csv",
                [
                    {
                        "member": int(index),
                        "normalized_parameter_0": float(population[index, 0]),
                        "normalized_parameter_1": float(population[index, 1]),
                        "physical_parameter_0": float(physical_population[index, 0]),
                        "physical_parameter_1": float(physical_population[index, 1]),
                        "observation_mse": float(objectives[index]),
                    }
                    for index in range(population.shape[0])
                ],
            )
            write_json(result_path, result)
            rows.append(result)
            request_index += 1
    write_csv(panel_dir / "de_per_case_results.csv", rows)
    print(
        f"[de] completed panel {panel_index:03d} ({len(rows)} requests)",
        flush=True,
    )
    return rows
