#!/usr/bin/env python3
"""Run the existing inverse methods on four informative history strata.

This is an additive entry point.  It reuses the original operator, projected LM,
PINN-DDE, aggregation and command-line method filtering without modifying them.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

import pipeline as _pipeline


_ORIGINAL_VALIDATE_CONFIG = _pipeline.validate_config
_ORIGINAL_READ_JSON = _pipeline.read_json
_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "stratified_four_history.json"
)

HISTORY_PROFILES: dict[str, dict[str, Any]] = {
    "balanced": {
        "names": ["low", "medium_low", "medium_high", "high"],
        "infection_ranges": [
            [0.004, 0.03],
            [0.03, 0.08],
            [0.08, 0.15],
            [0.15, 0.25],
        ],
    },
    "high50": {
        "names": ["low", "medium", "high_lower", "high_upper"],
        "infection_ranges": [
            [0.004, 0.03],
            [0.03, 0.10],
            [0.10, 0.175],
            [0.175, 0.25],
        ],
    },
    "high75": {
        "names": ["low", "high_lower", "high_middle", "high_upper"],
        "infection_ranges": [
            [0.004, 0.03],
            [0.10, 0.15],
            [0.15, 0.20],
            [0.20, 0.25],
        ],
    },
}


def config_with_history_profile(
    config: Mapping[str, Any], profile: str
) -> dict[str, Any]:
    """Return an isolated config with one command-line-selected history mix."""
    if profile not in HISTORY_PROFILES:
        raise ValueError(
            f"unknown history profile {profile!r}; allowed values are "
            f"{list(HISTORY_PROFILES)}"
        )
    result = copy.deepcopy(dict(config))
    definition = HISTORY_PROFILES[profile]
    base_ranges = np.asarray(result["history_level_ranges"], dtype=np.float64)
    strata = []
    for infection_range in definition["infection_ranges"]:
        ranges = base_ranges.copy()
        ranges[2] = np.asarray(infection_range, dtype=np.float64)
        strata.append(ranges.tolist())
    result["history_profile"] = profile
    result["experiment_name"] = (
        f"delayed_sei3d_stratified_four_history_{profile}_inverse"
    )
    result["stratified_history_names"] = list(definition["names"])
    result["stratified_history_level_ranges"] = strata
    return result


def extract_history_profile(
    argv: Sequence[str] | None,
) -> tuple[str, list[str], Path]:
    """Remove the additive argument before delegating to the original parser."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    profile = "balanced"
    cleaned: list[str] = []
    config_path: Path | None = None
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--history-profile":
            if index + 1 >= len(arguments):
                raise ValueError("--history-profile requires a value")
            profile = arguments[index + 1]
            index += 2
            continue
        if argument.startswith("--history-profile="):
            profile = argument.split("=", 1)[1]
            index += 1
            continue
        if argument == "--config":
            if index + 1 >= len(arguments):
                raise ValueError("--config requires a value")
            config_path = Path(arguments[index + 1]).expanduser().resolve()
            cleaned.extend((argument, str(config_path)))
            index += 2
            continue
        if argument.startswith("--config="):
            config_path = Path(argument.split("=", 1)[1]).expanduser().resolve()
            cleaned.extend(("--config", str(config_path)))
            index += 1
            continue
        cleaned.append(argument)
        index += 1
    if profile not in HISTORY_PROFILES:
        raise ValueError(
            f"--history-profile must be one of {','.join(HISTORY_PROFILES)}"
        )
    if config_path is None:
        config_path = _DEFAULT_CONFIG_PATH.resolve()
        cleaned.extend(("--config", str(config_path)))
    return profile, cleaned, config_path


def condition_definitions(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Define four all-state, four-history noise conditions."""
    section = config["stratified_four_history_noise"]
    definitions: list[dict[str, Any]] = []
    for sigma in section["noise_standard_deviations"]:
        definitions.append(
            {
                "experiment": "stratified_four_history_noise",
                "condition": (
                    "stratified_four_histories_all_states_"
                    f"{_pipeline.noise_tag(float(sigma))}"
                ),
                "display_name": (
                    f"{section['display_name']}, sigma={float(sigma):g}"
                ),
                "history_count": int(section["history_count"]),
                "observation_count": int(section["observation_count"]),
                "observed_state_indices": list(section["observed_state_indices"]),
                "noise_standard_deviation": float(sigma),
                "observation_mode": "stratified_four_histories_all_states",
            }
        )
    return definitions


def validate_config(config: Mapping[str, Any]) -> None:
    """Retain original numerical checks and validate the added protocol."""
    _ORIGINAL_VALIDATE_CONFIG(config)
    section = config.get("stratified_four_history_noise")
    if not isinstance(section, Mapping):
        raise ValueError("stratified_four_history_noise is required")
    if int(section.get("history_count", 0)) != 4:
        raise ValueError("the stratified experiment requires four histories")
    if int(section.get("observation_count", 0)) != 20:
        raise ValueError("each history requires twenty observation times")
    if list(section.get("observed_state_indices", ())) != [0, 1, 2]:
        raise ValueError("the stratified experiment must observe S, E and I")
    if [float(value) for value in section.get("noise_standard_deviations", ())] != [
        0.005,
        0.01,
        0.02,
        0.05,
    ]:
        raise ValueError("noise levels must be 0.005,0.01,0.02,0.05")

    strata = np.asarray(config.get("stratified_history_level_ranges"), dtype=np.float64)
    support = np.asarray(config["history_level_ranges"], dtype=np.float64)
    if strata.shape != (4, _pipeline.STATE_DIM, 2):
        raise ValueError("stratified_history_level_ranges must have shape (4,3,2)")
    if np.any(strata[..., 0] > strata[..., 1]):
        raise ValueError("a stratified history interval is reversed")
    if np.any(strata[..., 0] < support[None, :, 0] - 1.0e-12) or np.any(
        strata[..., 1] > support[None, :, 1] + 1.0e-12
    ):
        raise ValueError("every history stratum must lie inside checkpoint support")
    names = list(config.get("stratified_history_names", ()))
    if len(names) != 4 or len(set(names)) != 4:
        raise ValueError("four unique stratified_history_names are required")


def prepare_shared_cases(
    directory: Path,
    config: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    checkpoint_identity: Mapping[str, Any],
    experiment_seed: int,
    smoke: bool,
    resume: bool,
) -> Path:
    """Generate one low-to-high infection history from each stratum per case."""
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
    strata = config["stratified_history_level_ranges"]
    histories_by_stratum = []
    for ranges in strata:
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
        case_count * int(config["histories_per_case"]),
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
        np.repeat(true_parameters, int(config["histories_per_case"]), axis=0),
        history_grid,
        output_times,
        equation,
        truth_step,
    )
    reference = flat_reference.reshape(
        case_count,
        int(config["histories_per_case"]),
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
            "histories_per_case": int(config["histories_per_case"]),
            "history_stratum_names": list(config["stratified_history_names"]),
            "history_stratum_level_ranges": copy.deepcopy(strata),
            "truth_internal_step": truth_step,
            "parameter_bounds": bounds,
        },
    )
    return archive_path


def plot_stage_results(
    rows: Sequence[Mapping[str, Any]],
    stage_dir: Path,
    config: Mapping[str, Any],
    smoke: bool,
) -> None:
    """Plot the four-noise comparison for whichever methods were selected."""
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
            stage_dir / "figures" / f"stratified_four_history_{suffix}",
            config["reporting"],
            smoke,
        )


# Apply patches at import time so multiprocessing spawn workers receive them.
_pipeline.condition_definitions = condition_definitions
_pipeline.validate_config = validate_config
_pipeline.prepare_shared_cases = prepare_shared_cases
_pipeline.plot_stage_results = plot_stage_results


def main(argv: Sequence[str] | None = None) -> int:
    profile, cleaned_argv, config_path = extract_history_profile(argv)

    def read_json_with_profile(path: Path) -> dict[str, Any]:
        values = _ORIGINAL_READ_JSON(path)
        if path.expanduser().resolve() == config_path:
            return config_with_history_profile(values, profile)
        return values

    _pipeline.read_json = read_json_with_profile
    try:
        return _pipeline.main(cleaned_argv)
    finally:
        _pipeline.read_json = _ORIGINAL_READ_JSON


if __name__ == "__main__":
    raise SystemExit(main())
