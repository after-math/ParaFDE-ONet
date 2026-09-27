#!/usr/bin/env python3
"""Additive eight-history High-75 inverse protocol with scaled L-BFGS-B."""

from __future__ import annotations

import copy
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import torch

import pipeline as _pipeline


_ORIGINAL_VALIDATE_CONFIG = _pipeline.validate_config
_ORIGINAL_RUN_OPERATOR_JOB = _pipeline.run_operator_job
_ORIGINAL_AGGREGATE_STAGE = _pipeline.aggregate_stage


def condition_definitions(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Define the four noise conditions for eight shared histories."""
    section = config["high75_eight_history_noise"]
    return [
        {
            "experiment": "high75_eight_history_40obs_noise",
            "condition": f"high75_eight_histories_all_states_{_pipeline.noise_tag(float(sigma))}",
            "display_name": f"{section['display_name']}, sigma={float(sigma):g}",
            "history_count": int(section["history_count"]),
            "observation_count": int(section["observation_count"]),
            "observed_state_indices": list(section["observed_state_indices"]),
            "noise_standard_deviation": float(sigma),
            "observation_mode": "high75_eight_histories_all_states",
        }
        for sigma in section["noise_standard_deviations"]
    ]


def validate_config(config: Mapping[str, Any]) -> None:
    """Retain original numerical checks and validate the additive protocol."""
    compatibility = copy.deepcopy(dict(config))
    compatibility["histories_per_case"] = 4
    _ORIGINAL_VALIDATE_CONFIG(compatibility)

    if int(config.get("histories_per_case", 0)) != 8:
        raise ValueError("the High-75 protocol requires eight histories per case")
    section = config.get("high75_eight_history_noise")
    if not isinstance(section, Mapping):
        raise ValueError("high75_eight_history_noise is required")
    if int(section.get("history_count", 0)) != 8:
        raise ValueError("each inverse condition must use eight histories")
    if int(section.get("observation_count", 0)) != 40:
        raise ValueError("each history must use forty observation times")
    if list(section.get("observed_state_indices", ())) != [0, 1, 2]:
        raise ValueError("S, E and I must all be observed")
    if [float(value) for value in section.get("noise_standard_deviations", ())] != [
        0.005,
        0.01,
        0.02,
        0.05,
    ]:
        raise ValueError("noise levels must be 0.005,0.01,0.02,0.05")

    strata = np.asarray(config.get("stratified_history_level_ranges"), dtype=np.float64)
    support = np.asarray(config["history_level_ranges"], dtype=np.float64)
    if strata.shape != (8, _pipeline.STATE_DIM, 2):
        raise ValueError("stratified_history_level_ranges must have shape (8,3,2)")
    if np.any(strata[..., 0] > strata[..., 1]):
        raise ValueError("a history interval is reversed")
    if np.any(strata[..., 0] < support[None, :, 0] - 1.0e-12) or np.any(
        strata[..., 1] > support[None, :, 1] + 1.0e-12
    ):
        raise ValueError("every history interval must lie inside checkpoint support")
    names = list(config.get("stratified_history_names", ()))
    if len(names) != 8 or len(set(names)) != 8:
        raise ValueError("eight unique stratified_history_names are required")

    inverse = config["operator_projected_grid"]
    if inverse.get("lbfgsb_objective_scaling") != "inverse_noise_variance":
        raise ValueError("L-BFGS-B must use inverse_noise_variance scaling")
    if float(inverse.get("lbfgsb_noise_variance_floor", 0.0)) <= 0.0:
        raise ValueError("lbfgsb_noise_variance_floor must be positive")


def prepare_shared_cases(
    directory: Path,
    config: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    checkpoint_identity: Mapping[str, Any],
    experiment_seed: int,
    smoke: bool,
    resume: bool,
) -> Path:
    """Generate one low, one medium and six high-infection histories per case."""
    directory.mkdir(parents=True, exist_ok=True)
    archive_path = directory / "cases.npz"
    manifest_path = directory / "manifest.json"
    signature = _pipeline.scientific_signature(config, checkpoint_identity, smoke)
    expected_manifest = {
        "signature": signature,
        "experiment_seed": int(experiment_seed),
        "checkpoint_sha256": checkpoint_identity["sha256"],
        "smoke": bool(smoke),
    }
    if archive_path.exists() or manifest_path.exists():
        if not (archive_path.exists() and manifest_path.exists() and resume):
            raise RuntimeError("shared case files are partial or require --resume")
        manifest = _pipeline.read_json(manifest_path)
        if any(manifest.get(key) != value for key, value in expected_manifest.items()):
            raise RuntimeError("shared case manifest is incompatible with this run")
        return archive_path

    resolved = checkpoint["resolved_config"]
    data = resolved["data"]
    equation = _pipeline.DelayedSEIConfig.from_mapping(resolved["equation"])
    case_count = int(config["smoke"]["case_count"] if smoke else config["case_count"])
    history_count = int(config["histories_per_case"])
    history_grid = np.linspace(
        -equation.maximum_history,
        0.0,
        int(checkpoint["model_config"]["history_sensors"]),
        dtype=np.float64,
    )
    output_times = np.linspace(
        0.0,
        float(checkpoint["model_config"]["horizon"]),
        int(data["output_points"]),
        dtype=np.float64,
    )
    seed_values = config["randomness"]
    history_rng = np.random.default_rng(
        int(experiment_seed) + int(seed_values["history_offset"])
    )
    histories_by_stratum = []
    for ranges in config["stratified_history_level_ranges"]:
        histories_by_stratum.append(
            _pipeline.sample_histories(
                case_count,
                history_grid,
                tuple(tuple(float(value) for value in pair) for pair in ranges),
                tuple(float(value) for value in config["history_sigmas"]),
                float(config["history_length_scale"]),
                tuple(
                    tuple(float(value) for value in pair)
                    for pair in config["history_bounds"]
                ),
                float(config["history_total_upper"]),
                history_rng,
            )
        )
    histories = np.stack(histories_by_stratum, axis=1)
    flat_histories = histories.reshape(
        case_count * history_count,
        _pipeline.STATE_DIM,
        history_grid.size,
    )

    bounds = np.asarray(equation.parameter_bounds, dtype=np.float64)
    parameter_rng = np.random.default_rng(
        int(experiment_seed) + int(seed_values["parameter_case_offset"])
    )
    true_parameters = np.empty((case_count, 2), dtype=np.float64)
    for dimension in range(2):
        fractions = (np.arange(case_count, dtype=np.float64) + 0.5) / case_count
        parameter_rng.shuffle(fractions)
        true_parameters[:, dimension] = (
            bounds[dimension, 0]
            + fractions * (bounds[dimension, 1] - bounds[dimension, 0])
        )
    truth_step = float(data["internal_step"] if smoke else config["truth_internal_step"])
    flat_reference = _pipeline.solve_batch(
        flat_histories,
        np.repeat(true_parameters, history_count, axis=0),
        history_grid,
        output_times,
        equation,
        truth_step,
    )
    reference = flat_reference.reshape(
        case_count,
        history_count,
        output_times.size,
        _pipeline.STATE_DIM,
    )
    noise_rng = np.random.default_rng(
        int(experiment_seed) + int(seed_values["observation_offset"])
    )
    standard_normal_noise = noise_rng.standard_normal(reference.shape).astype(np.float32)
    temporary = archive_path.with_name(".cases.npz")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            histories=histories,
            history_grid=history_grid,
            output_times=output_times,
            true_parameters=true_parameters,
            reference=reference,
            standard_normal_noise=standard_normal_noise,
            history_stratum_names=np.asarray(config["stratified_history_names"]),
        )
    os.replace(temporary, archive_path)
    _pipeline.write_json(
        manifest_path,
        {
            **expected_manifest,
            "case_count": case_count,
            "histories_per_case": history_count,
            "observation_count_per_history": 40,
            "history_stratum_names": list(config["stratified_history_names"]),
            "history_stratum_level_ranges": copy.deepcopy(
                config["stratified_history_level_ranges"]
            ),
            "truth_internal_step": truth_step,
            "parameter_bounds": bounds,
        },
    )
    return archive_path


def refine_operator_lbfgsb_scaled(
    model: _pipeline.Operator3D,
    starts: torch.Tensor,
    history: torch.Tensor,
    history_grid: torch.Tensor,
    observation_times: torch.Tensor,
    observations: torch.Tensor,
    observed_states: Sequence[int],
    collocation_times: torch.Tensor,
    equation: _pipeline.DelayedSEIConfig,
    inverse_config: Mapping[str, Any],
    maximum_iterations: int,
    output_path: Path,
    deadline: float | None = None,
) -> tuple[torch.Tensor, bool]:
    """Refine candidates after scaling objective and gradient by 1/sigma^2."""
    objective_scale = float(inverse_config.get("lbfgsb_objective_scale", 1.0))
    if not np.isfinite(objective_scale) or objective_scale <= 0.0:
        raise ValueError("lbfgsb_objective_scale must be finite and positive")
    refined = starts.detach().cpu().numpy().astype(np.float64)
    diagnostics: list[dict[str, Any]] = []
    budget_exhausted = False
    for index in range(refined.shape[0]):
        if deadline is not None and time.perf_counter() >= deadline:
            budget_exhausted = True
            break
        initial = refined[index].copy()

        def raw_value_and_gradient(values: np.ndarray) -> tuple[float, np.ndarray]:
            candidate = torch.as_tensor(
                values, dtype=torch.float32, device=history.device
            ).reshape(1, 2).requires_grad_(True)
            objective, _, _ = _pipeline.operator_objective(
                model,
                candidate,
                history,
                history_grid,
                observation_times,
                observations,
                observed_states,
                collocation_times[index : index + 1],
                equation,
                float(inverse_config["physics_weight"]),
            )
            gradient = torch.autograd.grad(objective.sum(), candidate)[0]
            _pipeline.synchronize(history.device)
            return (
                float(objective.detach().cpu()[0]),
                gradient.detach().cpu().numpy().reshape(2).astype(np.float64),
            )

        def scaled_value_and_gradient(values: np.ndarray) -> tuple[float, np.ndarray]:
            raw_value, raw_gradient = raw_value_and_gradient(values)
            return raw_value * objective_scale, raw_gradient * objective_scale

        initial_raw, initial_gradient = raw_value_and_gradient(initial)
        result = minimize(
            scaled_value_and_gradient,
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=((-1.0, 1.0), (-1.0, 1.0)),
            options={
                "maxiter": int(maximum_iterations),
                "maxcor": int(inverse_config["lbfgsb_history_size"]),
                "ftol": float(inverse_config["lbfgsb_ftol"]),
                "gtol": float(inverse_config["lbfgsb_gtol"]),
                "maxls": 30,
            },
        )
        refined[index] = np.clip(np.asarray(result.x), -1.0, 1.0)
        final_raw, final_gradient = raw_value_and_gradient(refined[index])
        diagnostics.append(
            {
                "restart_index": index,
                "success": int(bool(result.success)),
                "status": int(result.status),
                "message": str(result.message),
                "iterations": int(result.nit),
                "function_evaluations": int(result.nfev),
                "objective_scale": objective_scale,
                "initial_raw_objective": initial_raw,
                "final_raw_objective": final_raw,
                "initial_scaled_objective": initial_raw * objective_scale,
                "final_scaled_objective": final_raw * objective_scale,
                "initial_raw_gradient_inf": float(np.max(np.abs(initial_gradient))),
                "final_raw_gradient_inf": float(np.max(np.abs(final_gradient))),
                "final_normalized_transmission_b": float(refined[index, 0]),
                "final_normalized_convexity_a": float(refined[index, 1]),
            }
        )
    _pipeline.write_csv(output_path, diagnostics)
    return (
        torch.as_tensor(refined, dtype=torch.float32, device=history.device),
        budget_exhausted,
    )


def run_operator_job_scaled(
    scenario_key: str,
    case_index: int,
    experiment_seed: int,
    config: Mapping[str, Any],
    checkpoint_path: Path,
    checkpoint_identity: Mapping[str, Any],
    archive_path: Path,
    device_name: str,
    output_dir: Path,
    smoke: bool,
    resume: bool,
) -> dict[str, Any]:
    """Inject the condition-specific 1/sigma^2 scale into the original job."""
    local_config = copy.deepcopy(dict(config))
    condition = _pipeline.condition_by_key(local_config, scenario_key)
    sigma = float(condition["noise_standard_deviation"])
    floor = float(
        local_config["operator_projected_grid"]["lbfgsb_noise_variance_floor"]
    )
    scale = 1.0 / max(sigma * sigma, floor)
    local_config["operator_projected_grid"]["lbfgsb_objective_scale"] = scale
    result = _ORIGINAL_RUN_OPERATOR_JOB(
        scenario_key,
        case_index,
        experiment_seed,
        local_config,
        checkpoint_path,
        checkpoint_identity,
        archive_path,
        device_name,
        output_dir,
        smoke,
        resume,
    )
    result["lbfgsb_objective_scaling"] = "inverse_noise_variance"
    result["lbfgsb_objective_scale"] = scale
    _pipeline.write_json(output_dir / "result.json", result)
    return result


def plot_stage_results(
    rows: Sequence[Mapping[str, Any]],
    stage_dir: Path,
    config: Mapping[str, Any],
    smoke: bool,
) -> None:
    """Plot noise trends for the selected method subset."""
    methods = _pipeline.active_methods(config)
    conditions = _pipeline.active_condition_definitions(config)
    for metric, label, suffix in (
        ("parameter_normalized_rmse", "Parameter NRMSE", "parameter_error"),
        ("optimization_seconds", "Optimization seconds", "runtime"),
    ):
        figure, axis = plt.subplots(figsize=(8, 4.5))
        for method in methods:
            levels = []
            medians = []
            for condition in conditions:
                values = [
                    float(row[metric])
                    for row in rows
                    if row["condition"] == condition["condition"]
                    and row["method_key"] == method
                ]
                levels.append(float(condition["noise_standard_deviation"]))
                medians.append(float(np.median(values)))
            axis.plot(
                levels,
                medians,
                marker="o",
                label=_pipeline.METHOD_LABELS[method],
            )
        axis.set_xlabel("Noise standard deviation")
        axis.set_ylabel(label)
        axis.set_yscale("log")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
        figure.tight_layout()
        _pipeline.save_figure(
            figure,
            stage_dir / "figures" / f"high75_eight_history_40obs_{suffix}",
            config["reporting"],
            smoke,
        )


def aggregate_stage(
    stage_dir: Path,
    config: Mapping[str, Any],
    experiment_seed: int,
    case_count: int,
    smoke: bool,
) -> dict[str, Any]:
    """Use original aggregation and correct its protocol descriptions."""
    summary = _ORIGINAL_AGGREGATE_STAGE(
        stage_dir, config, experiment_seed, case_count, smoke
    )
    summary["shared_data_rule"] = (
        "All methods share eight High-75 histories per case, truth parameters, "
        "forty observation times per history, paired noise arrays and physical bounds."
    )
    summary["operator_implementation"] = (
        "Normalized Cartesian screen, top-K projected Adam, then independently "
        "bounded L-BFGS-B with objective and gradient scaled by inverse noise variance."
    )
    _pipeline.write_json(stage_dir / "comparison.json", summary)
    return summary


# Apply at import time so multiprocessing spawn workers receive identical behavior.
_pipeline.condition_definitions = condition_definitions
_pipeline.validate_config = validate_config
_pipeline.prepare_shared_cases = prepare_shared_cases
_pipeline.refine_operator_lbfgsb = refine_operator_lbfgsb_scaled
_pipeline.run_operator_job = run_operator_job_scaled
_pipeline.plot_stage_results = plot_stage_results
_pipeline.aggregate_stage = aggregate_stage


def main(argv: Sequence[str] | None = None) -> int:
    return _pipeline.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
