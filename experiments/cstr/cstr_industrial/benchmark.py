"""Twenty-event inverse/forward industrial benchmark for the delayed CSTR."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import numpy as np
import torch

from cstr_forward.data import latin_hypercube
from cstr_forward.equation import CSTRConfig, constant_histories, solve_batch
from cstr_forward.model import CSTRParaFDEONet, load_operator_checkpoint
from cstr_pilot.pilot import (
    choose_control,
    control_grid,
    direct_screen,
    feasible_mask,
    inverse_direct_lm,
    inverse_para_lm,
    normalized_parameter_error,
    operator_screen,
    trajectory_features,
    verify_recommendation,
)


LOGGER = logging.getLogger(__name__)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, values: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(values), indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def save_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot save an empty CSV")
    fieldnames: list[str] = []
    for row in rows:
        fieldnames.extend(key for key in row if key not in fieldnames)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sample_parameter_events(
    equation: CSTRConfig,
    old_parameters: np.ndarray,
    count: int,
    seed: int,
    minimum_normalized_jump: float,
) -> np.ndarray:
    """Generate a fixed space-filling event list before looking at any outcomes."""
    if count < 1:
        raise ValueError("event count must be positive")
    bounds = equation.bounds_array[:2]
    span = bounds[:, 1] - bounds[:, 0]
    rng = np.random.default_rng(seed)
    accepted: list[np.ndarray] = []
    batch = max(4 * count, 32)
    while len(accepted) < count:
        unit = latin_hypercube(batch, 2, rng)
        candidates = bounds[:, 0] + unit * span
        for candidate in candidates:
            jump = float(np.linalg.norm((candidate - old_parameters) / span))
            if jump >= minimum_normalized_jump:
                accepted.append(candidate.copy())
                if len(accepted) == count:
                    break
    return np.asarray(accepted, dtype=np.float64)


def inverse_direct_de(
    history: np.ndarray,
    control: np.ndarray,
    observations: np.ndarray,
    config: Mapping[str, Any],
    equation: CSTRConfig,
    residual_scale: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """ThreeSystems-aligned DE/rand/1/bin using the same DDE objective as LM."""
    bounds = equation.bounds_array[:2]
    scale = np.asarray(residual_scale, dtype=np.float64)
    if scale.shape != (2,) or np.any(scale <= 0.0):
        raise ValueError("residual_scale must have shape [2] and be positive")
    if str(config["de_strategy"]) != "rand1bin":
        raise ValueError("only DE/rand/1/bin is supported")
    if str(config["de_coordinate_system"]) != "normalized":
        raise ValueError("Direct DE must search normalized coordinates")
    if str(config["de_initialization"]) != "latin_hypercube":
        raise ValueError("Direct DE requires Latin-hypercube initialization")
    if str(config["de_bound_handling"]) != "reflection":
        raise ValueError("Direct DE requires reflection bound handling")
    if bool(config["de_polish"]):
        raise ValueError("Direct DE must not use local polishing")
    population_size = int(config["de_population"])
    generations = int(config["de_generations"])
    maximum_evaluations = int(config["de_maximum_candidate_evaluations"])
    expected_evaluations = population_size * (generations + 1)
    if population_size < 4 or generations < 0:
        raise ValueError("DE requires population >= 4 and nonnegative generations")
    if maximum_evaluations != expected_evaluations:
        raise ValueError(
            "de_maximum_candidate_evaluations must equal de_population * "
            "(de_generations + 1)"
        )
    mutation = float(config["de_mutation"])
    crossover = float(config["de_crossover"])
    if not 0.0 < mutation <= 2.0 or not 0.0 <= crossover <= 1.0:
        raise ValueError("invalid DE mutation or crossover constant")
    rng = np.random.default_rng(seed)
    population = latin_hypercube(population_size, 2, rng)
    solver_batch_calls = 0
    trajectory_solves = 0

    def reflect_unit(values: np.ndarray) -> np.ndarray:
        wrapped = np.mod(np.asarray(values, dtype=np.float64), 2.0)
        return np.where(wrapped <= 1.0, wrapped, 2.0 - wrapped)

    def objective(candidates: np.ndarray) -> np.ndarray:
        nonlocal solver_batch_calls, trajectory_solves
        physical = bounds[:, 0] + candidates * (bounds[:, 1] - bounds[:, 0])
        conditions = np.column_stack(
            (
                physical,
                np.full(candidates.shape[0], control[0]),
                np.full(candidates.shape[0], control[1]),
            )
        )
        prediction = solve_batch(
            np.repeat(history, candidates.shape[0], axis=0),
            conditions,
            float(config["observation_horizon"]),
            int(config["observation_points"]),
            float(config["reference_step"]),
            equation,
        ).astype(np.float64)
        solver_batch_calls += 1
        trajectory_solves += candidates.shape[0]
        residual = (prediction - observations[None]) / scale[None, None, :]
        return np.mean(residual**2, axis=(1, 2))

    started = time.perf_counter()
    values = objective(population)
    for generation in range(1, generations + 1):
        trials = np.empty_like(population)
        for index in range(population_size):
            choices = np.delete(np.arange(population_size), index)
            a, b, c = rng.choice(choices, size=3, replace=False)
            mutant = reflect_unit(
                population[a] + mutation * (population[b] - population[c])
            )
            mask = rng.random(2) < crossover
            mask[rng.integers(0, 2)] = True
            trials[index] = np.where(mask, mutant, population[index])
        trial_values = objective(trials)
        accepted = trial_values < values
        population[accepted] = trials[accepted]
        values[accepted] = trial_values[accepted]
    if trajectory_solves != maximum_evaluations:
        raise RuntimeError("Direct DE candidate-evaluation accounting mismatch")
    selected = int(np.argmin(values))
    estimate = bounds[:, 0] + population[selected] * (bounds[:, 1] - bounds[:, 0])
    elapsed = time.perf_counter() - started
    return estimate.copy(), {
        "time_seconds": elapsed,
        "warm_online_seconds": elapsed,
        "forward_model_calls": int(solver_batch_calls),
        "forward_condition_evaluations": int(trajectory_solves),
        "solver_batch_calls": int(solver_batch_calls),
        "direct_trajectory_solves": int(trajectory_solves),
        "candidate_objective_evaluations": int(trajectory_solves),
        "de_generations_completed": int(generations),
        "de_stop_reason": "maximum_candidate_evaluations",
        "de_early_stopped": False,
        "final_loss": float(values[selected]),
    }


def median_timed_call(
    callback: Callable[[], tuple[Any, Mapping[str, Any]]], repeats: int
) -> tuple[Any, dict[str, Any]]:
    """Repeat a deterministic inversion and retain its median measured timing."""
    if repeats < 1:
        raise ValueError("timing repeats must be positive")
    outputs: list[Any] = []
    records: list[dict[str, Any]] = []
    for _ in range(repeats):
        value, record = callback()
        outputs.append(value)
        records.append(dict(record))
    times = np.asarray([float(row["time_seconds"]) for row in records])
    selected = int(np.argsort(times)[len(times) // 2])
    result = records[selected]
    result["timing_repeats"] = int(repeats)
    result["time_seconds_samples"] = times.tolist()
    result["time_seconds"] = float(np.median(times))
    warm = [float(row.get("warm_online_seconds", row["time_seconds"])) for row in records]
    result["warm_online_seconds"] = float(np.median(warm))
    return outputs[selected], result


def build_old_context(
    config: Mapping[str, Any], equation: CSTRConfig
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    controls = control_grid(equation, int(config["control_grid_points"]))
    old_parameters = np.asarray(config["old_parameters"], dtype=np.float64)
    initial = constant_histories(1, int(round(equation.delay / float(config["reference_step"]))) + 1)
    trajectories, screen_seconds = direct_screen(
        initial, old_parameters, controls, config, equation
    )
    features = trajectory_features(trajectories)
    index, productivity = choose_control(controls, features, config, margin=True)
    control = controls[index].astype(np.float64)
    burn = solve_batch(
        initial,
        np.asarray([[*old_parameters, *control]], dtype=np.float64),
        float(config["old_burn_horizon"]),
        int(round(float(config["old_burn_horizon"]) / float(config["reference_step"]))) + 1,
        float(config["reference_step"]),
        equation,
        return_internal=True,
    )
    sensors = initial.shape[-1]
    history = burn[:, -sensors:, :].transpose(0, 2, 1)
    details = {
        "old_screen_seconds": float(screen_seconds),
        "old_control_index": int(index),
        "old_productivity_predicted": float(productivity[index]),
        "old_maximum_temperature": float(features["maximum_temperature"][index]),
        "old_conversion": float(features["conversion"][index]),
        "old_temperature_std": float(features["temperature_std"][index]),
    }
    return history, control, details


def generate_event_observation(
    old_history: np.ndarray,
    old_control: np.ndarray,
    new_parameters: np.ndarray,
    config: Mapping[str, Any],
    equation: CSTRConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    step = float(config["reference_step"])
    internal = solve_batch(
        old_history,
        np.asarray([[*new_parameters, *old_control]], dtype=np.float64),
        float(config["observation_horizon"]),
        int(round(float(config["observation_horizon"]) / step)) + 1,
        step,
        equation,
        return_internal=True,
    )
    indices = np.linspace(
        0, internal.shape[1] - 1, int(config["observation_points"])
    ).round().astype(int)
    clean = internal[0, indices].astype(np.float64)
    planning_history = internal[:, -old_history.shape[-1] :, :].transpose(0, 2, 1)
    times = np.linspace(
        0.0,
        float(config["observation_horizon"]),
        int(config["observation_points"]),
    )
    return clean, planning_history, times


def run_inversions(
    model: CSTRParaFDEONet,
    old_history: np.ndarray,
    old_control: np.ndarray,
    observations: np.ndarray,
    truth: np.ndarray,
    config: Mapping[str, Any],
    equation: CSTRConfig,
    device: torch.device,
    request_seed: int,
    repeats: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    scale = model.output_std.detach().cpu().numpy().astype(np.float64)
    callbacks = {
        "ParaFDEONet LM": lambda: inverse_para_lm(
            model, old_history, old_control, observations, config, equation, device
        ),
        "Direct LM": lambda: inverse_direct_lm(
            old_history, old_control, observations, config, equation, scale
        ),
        "Direct DE": lambda: inverse_direct_de(
            old_history,
            old_control,
            observations,
            config,
            equation,
            scale,
            request_seed + 700,
        ),
    }
    estimates: dict[str, np.ndarray] = {}
    records: dict[str, dict[str, Any]] = {}
    for method, callback in callbacks.items():
        estimate, record = median_timed_call(callback, repeats)
        estimate = np.asarray(estimate, dtype=np.float64)
        record["normalized_parameter_error"] = normalized_parameter_error(
            estimate, truth, equation
        )
        record["estimated_k"] = float(estimate[0])
        record["estimated_kappa"] = float(estimate[1])
        estimates[method] = estimate
        records[method] = record
    return estimates, records


def screen_and_decide(
    model: CSTRParaFDEONet,
    planning_history: np.ndarray,
    old_control: np.ndarray,
    true_parameters: np.ndarray,
    para_estimate: np.ndarray,
    direct_estimate: np.ndarray,
    config: Mapping[str, Any],
    equation: CSTRConfig,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    controls = control_grid(equation, int(config["control_grid_points"]))
    para_trajectories, para_screen_seconds, para_cache = operator_screen(
        model, planning_history, para_estimate, controls, config, device
    )
    direct_trajectories, direct_screen_seconds = direct_screen(
        planning_history, direct_estimate, controls, config, equation
    )
    oracle_trajectories, oracle_screen_seconds = direct_screen(
        planning_history, true_parameters, controls, config, equation
    )
    para_features = trajectory_features(para_trajectories)
    direct_features = trajectory_features(direct_trajectories)
    oracle_features = trajectory_features(oracle_trajectories)
    decision_started = time.perf_counter()
    para_index, _ = choose_control(controls, para_features, config, margin=True)
    direct_index, _ = choose_control(controls, direct_features, config, margin=True)
    oracle_index, _ = choose_control(controls, oracle_features, config, margin=False)
    decision_seconds = time.perf_counter() - decision_started
    recommendations = {
        "Static": verify_recommendation(
            planning_history, true_parameters, old_control, config, equation
        ),
        "Direct physics": verify_recommendation(
            planning_history, true_parameters, controls[direct_index], config, equation
        ),
        "ParaFDEONet": verify_recommendation(
            planning_history, true_parameters, controls[para_index], config, equation
        ),
        "Oracle": verify_recommendation(
            planning_history, true_parameters, controls[oracle_index], config, equation
        ),
    }
    oracle_productivity = recommendations["Oracle"]["valid_productivity"]
    for values in recommendations.values():
        values["production_loss"] = float(
            1.0
            - values["valid_productivity"] / max(oracle_productivity, 1.0e-12)
        )
    true_safe = feasible_mask(oracle_features, config, margin=False)
    para_safe = feasible_mask(para_features, config, margin=False)
    direct_safe = feasible_mask(direct_features, config, margin=False)
    difference = para_trajectories - oracle_trajectories
    relative = np.linalg.norm(difference.reshape(len(controls), -1), axis=1) / np.maximum(
        np.linalg.norm(oracle_trajectories.reshape(len(controls), -1), axis=1),
        1.0e-12,
    )
    summary = {
        "screen_time_seconds": {
            "ParaFDEONet": float(para_screen_seconds),
            "Direct physics": float(direct_screen_seconds),
            "Oracle": float(oracle_screen_seconds),
        },
        "decision_time_seconds": float(decision_seconds),
        "para_screen_cache_seconds": para_cache,
        "map_metrics": {
            "para_safe_accuracy": float(np.mean(para_safe == true_safe)),
            "para_false_safe_count": int(np.sum(para_safe & ~true_safe)),
            "para_false_safe_rate_all": float(np.mean(para_safe & ~true_safe)),
            "direct_safe_accuracy": float(np.mean(direct_safe == true_safe)),
            "direct_false_safe_count": int(np.sum(direct_safe & ~true_safe)),
            "direct_false_safe_rate_all": float(np.mean(direct_safe & ~true_safe)),
            "para_trajectory_relative_l2_mean": float(relative.mean()),
            "para_trajectory_relative_l2_q95": float(np.quantile(relative, 0.95)),
        },
        "recommendations": recommendations,
    }
    maps = {
        "controls": controls,
        "true_safe": true_safe,
        "para_safe": para_safe,
        "direct_safe": direct_safe,
        "para_features": np.column_stack(tuple(para_features.values())),
        "direct_features": np.column_stack(tuple(direct_features.values())),
        "oracle_features": np.column_stack(tuple(oracle_features.values())),
    }
    return summary, maps


def run_event(
    event_index: int,
    parameters: np.ndarray,
    model: CSTRParaFDEONet,
    old_history: np.ndarray,
    old_control: np.ndarray,
    config: Mapping[str, Any],
    equation: CSTRConfig,
    device: torch.device,
    repeats: int,
    run_pipeline: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray] | None]:
    clean, planning_history, observation_times = generate_event_observation(
        old_history, old_control, parameters, config, equation
    )
    direction = np.random.default_rng(int(config["seed"]) + 1009 * (event_index + 1)).normal(
        size=clean.shape
    )
    output_std = model.output_std.detach().cpu().numpy().astype(np.float64)
    inversion_rows: list[dict[str, Any]] = []
    pipeline_estimates: dict[str, np.ndarray] | None = None
    pipeline_records: dict[str, dict[str, Any]] | None = None
    for noise_index, noise_fraction in enumerate(config["noise_levels"]):
        observations = clean + float(noise_fraction) * output_std[None, :] * direction
        estimates, records = run_inversions(
            model,
            old_history,
            old_control,
            observations,
            parameters,
            config,
            equation,
            device,
            int(config["seed"]) + event_index * 101 + noise_index,
            repeats,
        )
        for method, record in records.items():
            inversion_rows.append(
                {
                    "event": int(event_index),
                    "noise_fraction": float(noise_fraction),
                    "true_k": float(parameters[0]),
                    "true_kappa": float(parameters[1]),
                    "method": method,
                    **record,
                }
            )
        if np.isclose(float(noise_fraction), float(config["pipeline_noise_fraction"])):
            pipeline_estimates = estimates
            pipeline_records = records
    event_summary: dict[str, Any] = {
        "event": int(event_index),
        "true_parameters": parameters.tolist(),
        "observation_times": observation_times.tolist(),
    }
    maps = None
    if run_pipeline:
        if pipeline_estimates is None or pipeline_records is None:
            raise RuntimeError("pipeline noise level is absent from noise_levels")
        pipeline, maps = screen_and_decide(
            model,
            planning_history,
            old_control,
            parameters,
            pipeline_estimates["ParaFDEONet LM"],
            pipeline_estimates["Direct LM"],
            config,
            equation,
            device,
        )
        pipeline["inverse_method"] = {
            "ParaFDEONet": "ParaFDEONet LM",
            "Direct physics": "Direct LM",
        }
        pipeline["cycle_time_seconds"] = {
            "ParaFDEONet": float(
                pipeline_records["ParaFDEONet LM"]["time_seconds"]
                + pipeline["screen_time_seconds"]["ParaFDEONet"]
                + pipeline["decision_time_seconds"]
            ),
            "Direct physics": float(
                pipeline_records["Direct LM"]["time_seconds"]
                + pipeline["screen_time_seconds"]["Direct physics"]
                + pipeline["decision_time_seconds"]
            ),
        }
        event_summary["pipeline"] = pipeline
    return inversion_rows, event_summary, maps


def aggregate_events(events: list[Mapping[str, Any]]) -> dict[str, Any]:
    completed = [event for event in events if "pipeline" in event]
    if not completed:
        return {}
    result: dict[str, Any] = {"event_count": len(completed), "pipelines": {}}
    for name in ("Static", "Direct physics", "ParaFDEONet"):
        recommendations = [event["pipeline"]["recommendations"][name] for event in completed]
        losses = np.asarray([row["production_loss"] for row in recommendations])
        unsafe = np.asarray([not row["true_safe"] for row in recommendations])
        values: dict[str, Any] = {
            "unsafe_recommendation_rate": float(unsafe.mean()),
            "unsafe_recommendation_count": int(unsafe.sum()),
            "production_loss_mean": float(losses.mean()),
            "production_loss_median": float(np.median(losses)),
            "production_loss_q25": float(np.quantile(losses, 0.25)),
            "production_loss_q75": float(np.quantile(losses, 0.75)),
        }
        if name in ("Direct physics", "ParaFDEONet"):
            cycle_name = name
            cycles = np.asarray(
                [event["pipeline"]["cycle_time_seconds"][cycle_name] for event in completed]
            )
            values.update(
                {
                    "cycle_time_mean_seconds": float(cycles.mean()),
                    "cycle_time_median_seconds": float(np.median(cycles)),
                    "cycle_time_q25_seconds": float(np.quantile(cycles, 0.25)),
                    "cycle_time_q75_seconds": float(np.quantile(cycles, 0.75)),
                }
            )
        result["pipelines"][name] = values
    para_cycle = result["pipelines"]["ParaFDEONet"]["cycle_time_median_seconds"]
    direct_cycle = result["pipelines"]["Direct physics"]["cycle_time_median_seconds"]
    result["median_cycle_speedup"] = float(direct_cycle / max(para_cycle, 1.0e-12))
    return result


def apply_smoke_overrides(config: dict[str, Any]) -> None:
    config.update(
        {
            "event_count": 1,
            "noise_levels": [0.02],
            "pipeline_noise_fraction": 0.02,
            "control_grid_points": 5,
            "parameter_grid_points": 5,
            "parameter_grid_topk": 2,
            "para_lm_starts": 2,
            "para_lm_minimum_iterations": 1,
            "para_lm_max_iterations": 2,
            "para_lm_early_stop_patience": 5,
            "lm_starts": 2,
            "lm_minimum_iterations": 1,
            "lm_max_iterations": 2,
            "lm_early_stop_patience": 5,
            "de_population": 6,
            "de_generations": 2,
            "de_maximum_candidate_evaluations": 18,
            "timing_repeats_representative": 1,
            "timing_repeats_events": 1,
            "verification_step": 0.01,
            "temperature_limit": 10.0,
            "conversion_minimum": -10.0,
            "temperature_std_limit": 10.0,
            "decision_temperature_limit": 10.0,
            "decision_conversion_minimum": -10.0,
            "decision_temperature_std_limit": 10.0,
        }
    )


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(args.output_dir / "industrial.log"),
            logging.StreamHandler(),
        ],
    )
    config = load_json(args.industrial_config)
    if args.smoke:
        config = copy.deepcopy(config)
        apply_smoke_overrides(config)
    training_config = load_json(args.experiment_config)
    equation = CSTRConfig.from_mapping(training_config["equation"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_operator_checkpoint(args.checkpoint, device)
    model.eval()
    for weight in model.parameters():
        weight.requires_grad_(False)
    old_history, old_control, old_details = build_old_context(config, equation)
    LOGGER.info("old control D=%.6f Tc=%.6f", *old_control)
    inversion_rows: list[dict[str, Any]] = []
    event_summaries: list[dict[str, Any]] = []

    if args.mode in ("representative", "all"):
        representative = np.asarray(config["representative_parameters"], dtype=np.float64)
        rows, summary, maps = run_event(
            -1,
            representative,
            model,
            old_history,
            old_control,
            config,
            equation,
            device,
            int(config["timing_repeats_representative"]),
            True,
        )
        inversion_rows.extend(rows)
        summary["representative"] = True
        event_summaries.append(summary)
        if maps is not None:
            np.savez_compressed(args.output_dir / "representative_maps.npz", **maps)

    if args.mode in ("events", "all"):
        events = sample_parameter_events(
            equation,
            np.asarray(config["old_parameters"], dtype=np.float64),
            int(config["event_count"]),
            int(config["event_seed"]),
            float(config["event_minimum_normalized_jump"]),
        )
        save_csv(
            args.output_dir / "event_parameters.csv",
            [
                {"event": index, "true_k": row[0], "true_kappa": row[1]}
                for index, row in enumerate(events)
            ],
        )
        for index, parameters in enumerate(events):
            LOGGER.info("event %d/%d parameters=%s", index + 1, len(events), parameters)
            rows, summary, _ = run_event(
                index,
                parameters,
                model,
                old_history,
                old_control,
                config,
                equation,
                device,
                int(config["timing_repeats_events"]),
                True,
            )
            inversion_rows.extend(rows)
            summary["representative"] = False
            event_summaries.append(summary)
            save_json(args.output_dir / "events" / f"event_{index:02d}.json", summary)

    save_csv(args.output_dir / "inversion_results.csv", inversion_rows)
    regular_events = [event for event in event_summaries if not event["representative"]]
    summary = {
        "network_format": checkpoint["network_format"],
        "checkpoint_iteration": int(checkpoint["iteration"]),
        "condition_feature_mode": "shared_p",
        "old_parameters": list(config["old_parameters"]),
        "old_control": old_control.tolist(),
        "old_context": old_details,
        "config": config,
        "aggregate": aggregate_events(regular_events),
        "representative": next(
            (event for event in event_summaries if event["representative"]), None
        ),
    }
    save_json(args.output_dir / "industrial_summary.json", summary)
    LOGGER.info("INDUSTRIAL_BENCHMARK_COMPLETED")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", type=Path, default=Path("configs/experiment.json"))
    parser.add_argument("--industrial-config", type=Path, default=Path("configs/industrial.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--mode", choices=("representative", "events", "all"), default="all"
    )
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())
