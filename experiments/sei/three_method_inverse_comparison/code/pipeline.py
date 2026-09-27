#!/usr/bin/env python3
"""Compare three inverse methods under paired noise and multi-history protocols."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import gc
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import socket
import sys
import time
import traceback
from typing import Any, Mapping, Sequence
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import torch
from torch import nn

matplotlib.rcParams["axes.unicode_minus"] = False


PROJECT_DIR = Path(__file__).resolve().parents[1]
DELAYED_SEI_ROOT = PROJECT_DIR.parent
SOURCE_PROJECT = DELAYED_SEI_ROOT
SOURCE_CODE = SOURCE_PROJECT / "code"
if not SOURCE_CODE.is_dir():
    raise RuntimeError(f"DelayedSEI3D source directory is missing: {SOURCE_CODE}")
for path_value in (str(SOURCE_PROJECT), str(SOURCE_CODE)):
    while path_value in sys.path:
        sys.path.remove(path_value)
sys.path.insert(0, str(SOURCE_CODE))
sys.path.insert(1, str(SOURCE_PROJECT))

from data import sample_histories  # noqa: E402
from equation import (  # noqa: E402
    PARAMETER_NAMES,
    STATE_DIM,
    DelayedSEIConfig,
    rhs_torch,
    solve_batch,
)
from model import (  # noqa: E402
    MODEL_DISPLAY_NAMES,
    MODEL_TYPES,
    OPERATOR_NETWORK_FORMAT,
    Operator3D,
    load_operator_checkpoint,
)
from training import operator_physics_residual  # noqa: E402


METHODS = ("operator_projected_grid", "projected_lm", "pinndde")
EXPERIMENTS = ("noise_escalation", "multi_history")
METHOD_LABELS = {
    "operator_projected_grid": "Frozen operator + normalized projection",
    "projected_lm": "Direct numerical projected LM",
    "pinndde": "PINN-DDE",
}
METHOD_COLORS = {
    "operator_projected_grid": "#0072B2",
    "projected_lm": "#009E73",
    "pinndde": "#D55E00",
}
LOGGER = logging.getLogger("delayed_sei3d_inverse_comparison")


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object."""
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return values


def write_json(path: Path, values: Mapping[str, Any]) -> None:
    """Atomically write JSON with NumPy-safe scalar conversion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")

    def default(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value).__name__)

    temporary.write_text(
        json.dumps(dict(values), indent=2, sort_keys=True, default=default),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Atomically write heterogeneous result rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(row) for row in rows]
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically save resumable Torch state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def setup_logger(path: Path) -> None:
    """Configure stdout and one UTF-8 log file for the current process."""
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


def synchronize(device: torch.device) -> None:
    """Synchronize CUDA timing without affecting CPU execution."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def normalized_to_physical(
    normalized: torch.Tensor,
    bounds: Sequence[Sequence[float]],
) -> torch.Tensor:
    """Linearly map projected ``[-1,1]`` variables to physical parameters."""
    if normalized.ndim < 1 or normalized.shape[-1] != 2:
        raise ValueError("normalized parameters must have final dimension 2")
    box = torch.as_tensor(bounds, dtype=normalized.dtype, device=normalized.device)
    if box.shape != (2, 2) or torch.any(box[:, 1] <= box[:, 0]):
        raise ValueError("parameter bounds must be two increasing pairs")
    center = box.mean(dim=1)
    half_span = 0.5 * (box[:, 1] - box[:, 0])
    return center + half_span * normalized


def physical_to_normalized(
    physical: torch.Tensor,
    bounds: Sequence[Sequence[float]],
) -> torch.Tensor:
    """Map physical parameters into the common projected coordinates."""
    box = torch.as_tensor(bounds, dtype=physical.dtype, device=physical.device)
    center = box.mean(dim=1)
    half_span = 0.5 * (box[:, 1] - box[:, 0])
    return (physical - center) / half_span


def bounded_parameters(
    raw_parameters: torch.Tensor,
    bounds: Sequence[Sequence[float]],
) -> torch.Tensor:
    """Sigmoid-map PINN raw parameters to the admissible physical box."""
    box = torch.as_tensor(
        bounds, dtype=raw_parameters.dtype, device=raw_parameters.device
    )
    return box[:, 0] + (box[:, 1] - box[:, 0]) * torch.sigmoid(raw_parameters)


def lhs_unit(count: int, seed: int) -> np.ndarray:
    """Generate a reproducible two-dimensional Latin hypercube design."""
    if count < 1:
        raise ValueError("LHS count must be positive")
    rng = np.random.default_rng(int(seed))
    values = np.empty((count, 2), dtype=np.float64)
    for dimension in range(2):
        values[:, dimension] = (np.arange(count) + rng.random(count)) / count
        rng.shuffle(values[:, dimension])
    return values


def lhs_physical(count: int, seed: int, bounds: np.ndarray) -> np.ndarray:
    """Generate physical LHS points inside the two-parameter box."""
    unit = lhs_unit(count, seed)
    return bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])


def physical_to_logits(physical: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Convert bounded physical starts to sigmoid logits without saturation."""
    fractions = (physical - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    fractions = np.clip(fractions, 1.0e-6, 1.0 - 1.0e-6)
    return np.log(fractions) - np.log1p(-fractions)


def normalized_parameter_rmse(
    estimates: np.ndarray, truth: np.ndarray, bounds: np.ndarray
) -> np.ndarray:
    """Return per-row parameter RMSE after division by each physical span."""
    values = np.asarray(estimates, dtype=np.float64)
    return np.sqrt(
        np.mean(
            ((values - np.asarray(truth)) / (bounds[:, 1] - bounds[:, 0])) ** 2,
            axis=-1,
        )
    )


def observation_indices(time_count: int, observation_count: int) -> np.ndarray:
    """Choose deterministic approximately uniform indices including endpoints."""
    if not 2 <= observation_count <= time_count:
        raise ValueError("observation_count must lie in [2,time_count]")
    result = np.rint(
        np.linspace(0, time_count - 1, observation_count)
    ).astype(np.int64)
    if np.unique(result).size != observation_count:
        raise ValueError("observation indices are not unique")
    return result


def trajectory_relative_l2(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Compute aggregate trajectory relative L2 error."""
    truth = np.asarray(reference, dtype=np.float64)
    error = np.asarray(prediction, dtype=np.float64) - truth
    return float(np.linalg.norm(error) / max(np.linalg.norm(truth), 1.0e-12))


def file_sha256(path: Path) -> str:
    """Hash a checkpoint once for strict runtime and resume identity."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_metadata(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one trained normalized-sensitivity operator checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("network_format") != OPERATOR_NETWORK_FORMAT:
        raise RuntimeError(
            "checkpoint network_format is incompatible with this inverse experiment"
        )
    model_type = str(checkpoint.get("model_type"))
    if model_type not in MODEL_TYPES:
        raise RuntimeError("checkpoint model_type is not one of the four trained operators")
    resolved = checkpoint.get("resolved_config")
    if not isinstance(resolved, Mapping):
        raise RuntimeError("checkpoint lacks resolved_config")
    for key in ("data", "equation", "operator"):
        if not isinstance(resolved.get(key), Mapping):
            raise RuntimeError(f"checkpoint resolved_config lacks {key}")
    # This exact network format implements physical-to-[-1,1] normalization in
    # Operator3D.normalize_parameters; older unversioned checkpoints are rejected.
    equation = DelayedSEIConfig.from_mapping(resolved["equation"])
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, Mapping):
        raise RuntimeError("checkpoint lacks model_config")
    expected_bounds = np.asarray(equation.parameter_bounds, dtype=np.float64)
    saved_bounds = np.asarray(model_config.get("parameter_bounds"), dtype=np.float64)
    if saved_bounds.shape != (2, 2) or not np.allclose(
        saved_bounds, expected_bounds, rtol=0.0, atol=1.0e-7
    ):
        raise RuntimeError("checkpoint model/equation parameter bounds disagree")
    identity = {
        "path": str(path),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
        "network_format": checkpoint["network_format"],
        "model_type": model_type,
        "model_display_name": MODEL_DISPLAY_NAMES[model_type],
        "training_seed": int(checkpoint.get("training_seed", -1)),
        "best_iteration": int(checkpoint.get("best_iteration", -1)),
        "best_validation": float(checkpoint.get("best_validation", float("nan"))),
    }
    return dict(checkpoint), identity


def validate_checkpoint_protocol(
    checkpoint: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    """Cross-check checkpoint grids against truth, LM and observation protocols."""
    resolved = checkpoint["resolved_config"]
    data = resolved["data"]
    for key in (
        "history_level_ranges", "history_sigmas", "history_length_scale",
        "history_bounds", "history_total_upper",
    ):
        if not np.allclose(
            np.asarray(config[key], dtype=np.float64),
            np.asarray(data[key], dtype=np.float64),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError(f"inverse {key} differs from the checkpoint training distribution")
    model_config = checkpoint["model_config"]
    equation = DelayedSEIConfig.from_mapping(resolved["equation"])
    horizon = float(model_config["horizon"])
    if not math.isclose(horizon, float(data.get("horizon", horizon)), abs_tol=1.0e-12):
        raise RuntimeError("checkpoint model and data horizons disagree")
    if int(model_config["history_sensors"]) != int(
        data.get("history_sensors", model_config["history_sensors"])
    ):
        raise RuntimeError("checkpoint model and data history sensor counts disagree")
    output_points = int(data["output_points"])
    for condition in condition_definitions(config):
        if int(condition["observation_count"]) > output_points:
            raise ValueError(
                f"{condition['condition']} requests more observations than output points"
            )
    for name, step in (
        ("truth_internal_step", float(config["truth_internal_step"])),
        ("projected_lm.internal_step", float(config["projected_lm"]["internal_step"])),
        ("checkpoint data.internal_step", float(data["internal_step"])),
    ):
        count = round(horizon / step)
        if step <= 0.0 or not math.isclose(count * step, horizon, abs_tol=1.0e-10):
            raise ValueError(f"{name} must divide the checkpoint horizon exactly")
        if step >= equation.delay - 1.0e-12:
            raise ValueError(f"{name} must be smaller than the fixed SEI delay")


def validate_config(config: Mapping[str, Any]) -> None:
    """Fail before expensive work when the comparison protocol is inconsistent."""
    if tuple(config.get("methods", ())) != METHODS:
        raise ValueError(f"methods must be exactly {METHODS}")
    seeds = config["randomness"].get("experiment_seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) != 5
        or len(set(seeds)) != 5
        or any(not isinstance(seed, int) or seed < 0 for seed in seeds)
    ):
        raise ValueError("five unique nonnegative experiment seeds are required")
    if int(config.get("case_count", 0)) != 10:
        raise ValueError("formal case_count must equal 10")
    if int(config.get("histories_per_case", 0)) != 4:
        raise ValueError("histories_per_case must equal 4")
    required_randomness = {
        "parameter_case_offset", "history_offset", "observation_offset",
        "initialization_offset", "collocation_offset",
    }
    if not required_randomness <= set(config["randomness"]):
        raise ValueError("randomness offsets are incomplete")
    noise = config.get("noise_escalation")
    multi = config.get("multi_history")
    if not isinstance(noise, Mapping) or not isinstance(multi, Mapping):
        raise ValueError("noise_escalation and multi_history are required")
    if int(noise["history_count"]) != 1 or int(noise["observation_count"]) != 20:
        raise ValueError("noise experiment must use one history and twenty times")
    if list(noise["observation_modes"]) != ["all_states", "two_states"]:
        raise ValueError("noise observation modes must be all_states then two_states")
    if [float(value) for value in noise["noise_standard_deviations"]] != [
        0.005, 0.01, 0.02, 0.05
    ]:
        raise ValueError("noise levels must be 0.005,0.01,0.02,0.05")
    if list(multi["conditions"]) != ["one_history", "four_histories"]:
        raise ValueError("multi-history conditions must be one_history then four_histories")
    expected_histories = {"one_history": 1, "four_histories": 4}
    for key, condition in multi["conditions"].items():
        if int(condition["history_count"]) != expected_histories[key]:
            raise ValueError(f"invalid history count for {key}")
        if int(condition["observation_count_per_history"]) != 20:
            raise ValueError(f"{key} must use twenty observation times per history")
    for condition in condition_definitions(config):
        states = list(condition["observed_state_indices"])
        if not states or len(states) != len(set(states)) or any(
            state not in range(STATE_DIM) for state in states
        ):
            raise ValueError(f"invalid observed_state_indices for {condition['condition']}")
        if float(condition["noise_standard_deviation"]) < 0.0:
            raise ValueError("noise cannot be negative")
    if float(config["truth_internal_step"]) <= 0.0:
        raise ValueError("truth_internal_step must be positive")
    thresholds = np.asarray(config["parameter_error_thresholds"], dtype=np.float64)
    if thresholds.ndim != 1 or np.any(thresholds <= 0.0):
        raise ValueError("parameter error thresholds must be positive")
    if config["pinndde"].get("activation") != "tanh":
        raise ValueError("the referenced PINN-DDE protocol requires tanh")
    if len(config["pinndde"].get("hidden_widths", [])) != 3:
        raise ValueError("PINN-DDE requires exactly three hidden widths")
    if int(config["pinndde"]["random_initializations"]) != int(
        config["projected_lm"]["random_initializations"]
    ):
        raise ValueError("formal LM and PINN-DDE must use the same LHS restart count")
    if float(config["projected_lm"]["internal_step"]) <= float(
        config["truth_internal_step"]
    ):
        raise ValueError("LM internal_step must exceed truth step to avoid inverse crime")


def noise_tag(value: float) -> str:
    """Return a stable directory-safe noise label."""
    return f"noise_{float(value):g}".replace(".", "p")


def condition_definitions(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand the fixed protocol into eight noise and two history conditions."""
    definitions: list[dict[str, Any]] = []
    noise = config["noise_escalation"]
    for standard_deviation in noise["noise_standard_deviations"]:
        for mode_key, mode in noise["observation_modes"].items():
            definitions.append(
                {
                    "experiment": "noise_escalation",
                    "condition": f"{mode_key}_{noise_tag(float(standard_deviation))}",
                    "display_name": f"{mode['display_name']}, sigma={float(standard_deviation):g}",
                    "history_count": int(noise["history_count"]),
                    "observation_count": int(noise["observation_count"]),
                    "observed_state_indices": list(mode["observed_state_indices"]),
                    "noise_standard_deviation": float(standard_deviation),
                    "observation_mode": mode_key,
                }
            )
    multi = config["multi_history"]
    for condition_key, condition in multi["conditions"].items():
        definitions.append(
            {
                "experiment": "multi_history",
                "condition": condition_key,
                "display_name": condition_key.replace("_", " ").title(),
                "history_count": int(condition["history_count"]),
                "observation_count": int(condition["observation_count_per_history"]),
                "observed_state_indices": list(multi["observed_state_indices"]),
                "noise_standard_deviation": float(multi["noise_standard_deviation"]),
                "observation_mode": "two_states",
            }
        )
    if len(definitions) != 10:
        raise RuntimeError("the formal protocol must expand to ten conditions")
    return definitions


def condition_by_key(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    """Resolve one unique condition key."""
    matches = [item for item in condition_definitions(config) if item["condition"] == key]
    if len(matches) != 1:
        raise KeyError(f"unknown or ambiguous condition: {key}")
    return matches[0]


def active_methods(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the runtime-selected method subset, preserving canonical order."""
    selected = set(config.get("runtime_selection", {}).get("methods", METHODS))
    return tuple(method for method in METHODS if method in selected)


def active_condition_definitions(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the runtime-selected condition subset in canonical protocol order."""
    definitions = condition_definitions(config)
    selected = set(
        config.get("runtime_selection", {}).get(
            "conditions", [item["condition"] for item in definitions]
        )
    )
    return [item for item in definitions if item["condition"] in selected]


def parse_csv_subset(
    value: str | None, allowed: Sequence[str], argument_name: str
) -> tuple[str, ...]:
    """Parse a nonempty comma-separated subset and reject unknown or repeated keys."""
    if value is None:
        return tuple(allowed)
    requested = tuple(piece.strip() for piece in value.split(",") if piece.strip())
    if not requested:
        raise ValueError(f"{argument_name} must select at least one value")
    if len(requested) != len(set(requested)):
        raise ValueError(f"{argument_name} contains duplicate values")
    unknown = [item for item in requested if item not in allowed]
    if unknown:
        raise ValueError(
            f"{argument_name} contains unknown values {unknown}; allowed values are {list(allowed)}"
        )
    requested_set = set(requested)
    return tuple(item for item in allowed if item in requested_set)


def scientific_signature(
    config: Mapping[str, Any], checkpoint_identity: Mapping[str, Any], smoke: bool
) -> str:
    """Hash scientific settings and checkpoint identity, excluding runtime paths."""
    payload = {
        "config": config,
        "checkpoint_sha256": checkpoint_identity["sha256"],
        "smoke": bool(smoke),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def prepare_shared_cases(
    directory: Path,
    config: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    checkpoint_identity: Mapping[str, Any],
    experiment_seed: int,
    smoke: bool,
    resume: bool,
) -> Path:
    """Generate independent histories, truth parameters, trajectories and noise."""
    directory.mkdir(parents=True, exist_ok=True)
    archive_path = directory / "cases.npz"
    manifest_path = directory / "manifest.json"
    signature = scientific_signature(config, checkpoint_identity, smoke)
    expected_manifest = {
        "signature": signature,
        "experiment_seed": int(experiment_seed),
        "checkpoint_sha256": checkpoint_identity["sha256"],
        "smoke": bool(smoke),
    }
    if archive_path.exists() or manifest_path.exists():
        if not (archive_path.exists() and manifest_path.exists() and resume):
            raise RuntimeError("shared case files are partial or require --resume")
        manifest = read_json(manifest_path)
        if any(manifest.get(key) != value for key, value in expected_manifest.items()):
            raise RuntimeError("shared case manifest is incompatible with this run")
        return archive_path

    resolved = checkpoint["resolved_config"]
    data = resolved["data"]
    equation = DelayedSEIConfig.from_mapping(resolved["equation"])
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
    histories_per_case = int(config["histories_per_case"])
    history_rng = np.random.default_rng(
        int(experiment_seed) + int(seed_values["history_offset"])
    )
    flat_histories = sample_histories(
        case_count * histories_per_case,
        history_grid,
        tuple(
            tuple(float(value) for value in pair)
            for pair in config["history_level_ranges"]
        ),
        tuple(float(value) for value in config["history_sigmas"]),
        float(config["history_length_scale"]),
        tuple(
            tuple(float(value) for value in pair)
            for pair in config["history_bounds"]
        ),
        float(config["history_total_upper"]),
        history_rng,
    )
    histories = flat_histories.reshape(
        case_count, histories_per_case, STATE_DIM, history_grid.size
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
    truth_step = float(
        data["internal_step"] if smoke else config["truth_internal_step"]
    )
    flat_reference = solve_batch(
        flat_histories,
        np.repeat(true_parameters, histories_per_case, axis=0),
        history_grid,
        output_times,
        equation,
        truth_step,
    )
    reference = flat_reference.reshape(
        case_count, histories_per_case, output_times.size, STATE_DIM
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
            # Delayed SEI's solver validates both grid endpoints at tight tolerance;
            # retain float64 so -maximum_history is not rounded by NPZ storage.
            history_grid=history_grid,
            output_times=output_times,
            true_parameters=true_parameters,
            reference=reference,
            standard_normal_noise=standard_normal_noise,
        )
    os.replace(temporary, archive_path)
    write_json(
        manifest_path,
        {
            **expected_manifest,
            "case_count": case_count,
            "histories_per_case": histories_per_case,
            "truth_internal_step": truth_step,
            "parameter_bounds": bounds,
        },
    )
    return archive_path


def load_case(archive_path: Path, case_index: int) -> dict[str, np.ndarray]:
    """Load one immutable inverse case from the shared archive."""
    with np.load(archive_path, allow_pickle=False) as values:
        return {
            "histories": values["histories"][case_index],
            "history_grid": values["history_grid"],
            "output_times": values["output_times"],
            "true_parameters": values["true_parameters"][case_index],
            "reference": values["reference"][case_index],
            "standard_normal_noise": values["standard_normal_noise"][case_index],
        }


def condition_observations(
    case: Mapping[str, np.ndarray], condition: Mapping[str, Any]
) -> dict[str, Any]:
    """Construct observations shared by all three methods."""
    indices = observation_indices(
        len(case["output_times"]), int(condition["observation_count"])
    )
    history_count = int(condition["history_count"])
    sigma = float(condition["noise_standard_deviation"])
    observations = (
        case["reference"][:history_count, indices, :]
        + sigma * case["standard_normal_noise"][:history_count, indices, :]
    ).astype(np.float32)
    return {
        "histories": np.asarray(case["histories"][:history_count], dtype=np.float32),
        "reference": np.asarray(case["reference"][:history_count], dtype=np.float32),
        "indices": indices,
        "times": np.asarray(case["output_times"])[indices],
        "values": observations,
        "states": tuple(int(value) for value in condition["observed_state_indices"]),
        "history_count": history_count,
        "scalar_observation_count": (
            history_count * len(indices) * len(condition["observed_state_indices"])
        ),
    }


def job_signature(
    method: str,
    scenario: str,
    case_index: int,
    experiment_seed: int,
    config: Mapping[str, Any],
    checkpoint_identity: Mapping[str, Any],
    smoke: bool,
) -> str:
    """Hash every scientific value that makes one completed job reusable."""
    payload = {
        "method": method,
        "scenario": scenario,
        "case_index": int(case_index),
        "experiment_seed": int(experiment_seed),
        "config": config,
        "checkpoint_sha256": checkpoint_identity["sha256"],
        "smoke": bool(smoke),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def finalize_result(
    method: str,
    scenario: str,
    case_index: int,
    experiment_seed: int,
    signature: str,
    status: str,
    estimates: np.ndarray,
    objectives: np.ndarray,
    true_parameters: np.ndarray,
    bounds: np.ndarray,
    case: Mapping[str, np.ndarray],
    equation: DelayedSEIConfig,
    config: Mapping[str, Any],
    optimization_seconds: float,
    output_dir: Path,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Select by visible objective and evaluate parameter/direct-trajectory error."""
    estimates = np.asarray(estimates, dtype=np.float64)
    objectives = np.asarray(objectives, dtype=np.float64)
    if estimates.ndim != 2 or estimates.shape[1] != 2 or objectives.shape != (
        estimates.shape[0],
    ):
        raise ValueError("final candidate arrays are inconsistent")
    selected = int(np.argmin(objectives))
    estimate = estimates[selected]
    truth = np.asarray(true_parameters, dtype=np.float64)
    errors = normalized_parameter_rmse(estimates, truth, bounds)
    condition = condition_by_key(config, scenario)
    history_count = int(condition["history_count"])
    evaluation_histories = np.asarray(case["histories"][:history_count])
    reconstruction = solve_batch(
        evaluation_histories,
        np.repeat(estimate[None, ...], history_count, axis=0),
        np.asarray(case["history_grid"], dtype=np.float64),
        np.asarray(case["output_times"], dtype=np.float64),
        equation,
        float(config["truth_internal_step"]),
    )
    np.save(output_dir / "direct_reconstruction.npy", reconstruction)
    thresholds = [float(value) for value in config["parameter_error_thresholds"]]
    result: dict[str, Any] = {
        "signature": signature,
        "status": status,
        "method_key": method,
        "method": METHOD_LABELS[method],
        "scenario": scenario,
        "case_index": int(case_index),
        "experiment_seed": int(experiment_seed),
        "selected_initialization_index": selected,
        "true_transmission_b": float(truth[0]),
        "true_convexity_a": float(truth[1]),
        "estimated_transmission_b": float(estimate[0]),
        "estimated_convexity_a": float(estimate[1]),
        "transmission_b_absolute_error": float(abs(estimate[0] - truth[0])),
        "convexity_a_absolute_error": float(abs(estimate[1] - truth[1])),
        "parameter_physical_rmse": float(np.sqrt(np.mean((estimate - truth) ** 2))),
        "parameter_normalized_rmse": float(errors[selected]),
        "selected_visible_objective": float(objectives[selected]),
        "direct_reconstruction_relative_l2": trajectory_relative_l2(
            case["reference"][:history_count], reconstruction
        ),
        "optimization_seconds": float(optimization_seconds),
        "candidate_count": int(estimates.shape[0]),
        "experiment": str(condition["experiment"]),
        "condition": str(condition["condition"]),
        "history_count": history_count,
        "observation_count_per_history": int(condition["observation_count"]),
        "observed_state_indices": list(condition["observed_state_indices"]),
        "noise_standard_deviation": float(condition["noise_standard_deviation"]),
        "scalar_observation_count": int(
            history_count
            * int(condition["observation_count"])
            * len(condition["observed_state_indices"])
        ),
    }
    for threshold in thresholds:
        result[f"success_at_{threshold:g}"] = int(errors[selected] <= threshold)
    if extra:
        result.update(dict(extra))
    write_json(output_dir / "result.json", result)
    return result


def parameter_grid(first_points: int, second_points: int, device: torch.device) -> torch.Tensor:
    """Build a deterministic Cartesian grid in normalized coordinates."""
    if min(first_points, second_points) < 2:
        raise ValueError("each grid dimension requires at least two points")
    return torch.cartesian_prod(
        torch.linspace(-1.0, 1.0, first_points, device=device),
        torch.linspace(-1.0, 1.0, second_points, device=device),
    )


def operator_data_objective(
    model: Operator3D,
    normalized_parameters: torch.Tensor,
    history: torch.Tensor,
    observation_times: torch.Tensor,
    observations: torch.Tensor,
    observed_states: Sequence[int],
    bounds: Sequence[Sequence[float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate joint observation MSE over candidates and shared histories."""
    physical = normalized_to_physical(normalized_parameters, bounds)
    candidate_count = physical.shape[0]
    history_count = history.shape[0]
    histories = history.unsqueeze(0).expand(candidate_count, -1, -1, -1).reshape(
        candidate_count * history_count, STATE_DIM, history.shape[-1]
    )
    parameters = physical.unsqueeze(1).expand(-1, history_count, -1).reshape(
        candidate_count * history_count, 2
    )
    states = torch.as_tensor(
        observed_states, dtype=torch.long, device=normalized_parameters.device
    )
    predicted = model(histories, parameters, observation_times).reshape(
        candidate_count, history_count, observation_times.numel(), STATE_DIM
    ).index_select(3, states)
    target = observations.index_select(2, states).unsqueeze(0)
    loss = torch.mean((predicted - target) ** 2, dim=(1, 2, 3))
    if not torch.all(torch.isfinite(loss)):
        raise FloatingPointError("non-finite operator grid objective")
    return loss, physical


def screen_parameter_grid(
    model: Operator3D,
    history: torch.Tensor,
    observation_times: torch.Tensor,
    observations: torch.Tensor,
    observed_states: Sequence[int],
    equation: DelayedSEIConfig,
    first_points: int,
    second_points: int,
    selected_count: int,
    batch_size: int,
    output_path: Path,
) -> torch.Tensor:
    """Rank a full normalized grid by visible observation loss and retain top K."""
    grid = parameter_grid(first_points, second_points, history.device)
    if not 1 <= selected_count <= grid.shape[0] or batch_size < 1:
        raise ValueError("invalid grid selection configuration")
    pieces: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, grid.shape[0], batch_size):
            objective, _ = operator_data_objective(
                model,
                grid[start : start + batch_size],
                history,
                observation_times,
                observations,
                observed_states,
                equation.parameter_bounds,
            )
            pieces.append(objective.cpu())
    objectives = torch.cat(pieces).numpy().astype(np.float64)
    normalized = grid.cpu().numpy().astype(np.float64)
    physical = normalized_to_physical(
        grid.cpu(), equation.parameter_bounds
    ).numpy().astype(np.float64)
    order = np.argsort(objectives, kind="stable")
    selected_indices = order[:selected_count]
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, order.size + 1)
    selected_set = set(int(value) for value in selected_indices)
    write_csv(
        output_path,
        [
            {
                "grid_index": index,
                "objective_rank": int(ranks[index]),
                "selected": int(index in selected_set),
                "normalized_transmission_b": float(normalized[index, 0]),
                "normalized_convexity_a": float(normalized[index, 1]),
                "transmission_b": float(physical[index, 0]),
                "convexity_a": float(physical[index, 1]),
                "observation_mse": float(objectives[index]),
            }
            for index in range(grid.shape[0])
        ],
    )
    return torch.as_tensor(
        normalized[selected_indices], dtype=torch.float32, device=history.device
    )


def operator_objective(
    model: Operator3D,
    normalized_parameters: torch.Tensor,
    history: torch.Tensor,
    history_grid: torch.Tensor,
    observation_times: torch.Tensor,
    observations: torch.Tensor,
    observed_states: Sequence[int],
    collocation_times: torch.Tensor,
    equation: DelayedSEIConfig,
    physics_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Compute data plus constant-delay Delayed SEI physics objective per start."""
    data_loss, physical = operator_data_objective(
        model,
        normalized_parameters,
        history,
        observation_times,
        observations,
        observed_states,
        equation.parameter_bounds,
    )
    candidate_count = physical.shape[0]
    history_count = history.shape[0]
    histories = history.unsqueeze(0).expand(candidate_count, -1, -1, -1).reshape(
        candidate_count * history_count, STATE_DIM, history.shape[-1]
    )
    parameters = physical.unsqueeze(1).expand(-1, history_count, -1).reshape(
        candidate_count * history_count, 2
    )
    if physics_weight > 0.0:
        physics_times = collocation_times.unsqueeze(1).expand(
            -1, history_count, -1
        ).reshape(candidate_count * history_count, -1).detach().clone().requires_grad_(True)
        residual = operator_physics_residual(
            model,
            histories,
            parameters,
            physics_times,
            equation,
        ).reshape(candidate_count, history_count, physics_times.shape[1], STATE_DIM)
        physics_loss = torch.mean(residual**2, dim=(1, 2, 3))
    else:
        physics_loss = torch.zeros_like(data_loss)
    objective = data_loss + float(physics_weight) * physics_loss
    if not torch.all(torch.isfinite(objective)):
        raise FloatingPointError("non-finite projected operator objective")
    return objective, physical, {
        "data_loss": data_loss,
        "physics_loss": physics_loss,
    }


def refine_operator_lbfgsb(
    model: Operator3D,
    starts: torch.Tensor,
    history: torch.Tensor,
    history_grid: torch.Tensor,
    observation_times: torch.Tensor,
    observations: torch.Tensor,
    observed_states: Sequence[int],
    collocation_times: torch.Tensor,
    equation: DelayedSEIConfig,
    inverse_config: Mapping[str, Any],
    maximum_iterations: int,
    output_path: Path,
    deadline: float | None = None,
) -> tuple[torch.Tensor, bool]:
    """Independently refine projected Adam candidates with bounded L-BFGS-B."""
    refined = starts.detach().cpu().numpy().astype(np.float64)
    diagnostics: list[dict[str, Any]] = []
    budget_exhausted = False
    for index in range(refined.shape[0]):
        if deadline is not None and time.perf_counter() >= deadline:
            budget_exhausted = True
            break
        initial = refined[index].copy()

        def value_and_gradient(values: np.ndarray) -> tuple[float, np.ndarray]:
            candidate = torch.as_tensor(
                values, dtype=torch.float32, device=history.device
            ).reshape(1, 2).requires_grad_(True)
            objective, _, _ = operator_objective(
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
            synchronize(history.device)
            return (
                float(objective.detach().cpu()[0]),
                gradient.detach().cpu().numpy().reshape(2).astype(np.float64),
            )

        initial_objective, _ = value_and_gradient(initial)
        result = minimize(
            value_and_gradient,
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
        final_objective, _ = value_and_gradient(refined[index])
        diagnostics.append(
            {
                "restart_index": index,
                "success": int(bool(result.success)),
                "status": int(result.status),
                "iterations": int(result.nit),
                "function_evaluations": int(result.nfev),
                "initial_objective": float(initial_objective),
                "final_objective": float(final_objective),
                "final_normalized_transmission_b": float(refined[index, 0]),
                "final_normalized_convexity_a": float(refined[index, 1]),
            }
        )
    write_csv(output_path, diagnostics)
    return (
        torch.as_tensor(refined, dtype=torch.float32, device=history.device),
        budget_exhausted,
    )


def run_operator_job(
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
    """Run normalized grid screening, projected Adam and bounded L-BFGS-B."""
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = job_signature(
        "operator_projected_grid",
        scenario_key,
        case_index,
        experiment_seed,
        config,
        checkpoint_identity,
        smoke,
    )
    result_path = output_dir / "result.json"
    if resume and result_path.exists():
        result = read_json(result_path)
        if result.get("signature") != signature:
            raise RuntimeError("completed operator result signature mismatch")
        return result
    online_started = time.perf_counter()
    setup_logger(output_dir / "job.log")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, checkpoint = load_operator_checkpoint(checkpoint_path, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    equation = DelayedSEIConfig.from_mapping(
        checkpoint["resolved_config"]["equation"]
    )
    case = load_case(archive_path, case_index)
    observed = condition_observations(case, condition_by_key(config, scenario_key))
    history = torch.as_tensor(observed["histories"], dtype=torch.float32, device=device)
    history_grid = torch.as_tensor(
        case["history_grid"], dtype=torch.float32, device=device
    )
    observation_times = torch.as_tensor(
        observed["times"], dtype=torch.float32, device=device
    )
    observations = torch.as_tensor(
        observed["values"], dtype=torch.float32, device=device
    )
    inverse = config["operator_projected_grid"]
    smoke_config = config["smoke"]
    first_points = int(
        smoke_config["grid_transmission_points"]
        if smoke
        else inverse["grid_transmission_points"]
    )
    second_points = int(
        smoke_config["grid_convexity_points"]
        if smoke
        else inverse["grid_convexity_points"]
    )
    selected_count = int(
        smoke_config["selected_grid_starts"]
        if smoke
        else inverse["selected_grid_starts"]
    )
    batch_size = int(
        smoke_config["grid_evaluation_batch_size"]
        if smoke
        else inverse["grid_evaluation_batch_size"]
    )
    adam_steps = int(
        smoke_config["operator_adam_steps"] if smoke else inverse["adam_steps"]
    )
    collocation_count = int(
        smoke_config["operator_collocation_points"]
        if smoke
        else inverse["collocation_points"]
    )
    lbfgsb_iterations = int(
        smoke_config["operator_lbfgsb_iterations"]
        if smoke
        else inverse["lbfgsb_max_iterations"]
    )
    started = time.perf_counter()
    normalized = screen_parameter_grid(
        model,
        history,
        observation_times,
        observations,
        observed["states"],
        equation,
        first_points,
        second_points,
        selected_count,
        batch_size,
        output_dir / "grid_screening.csv",
    ).requires_grad_(True)
    collocation_rng = np.random.default_rng(
        experiment_seed
        + int(config["randomness"]["collocation_offset"])
        + case_index * 100
    )
    collocation_times = torch.as_tensor(
        np.sort(
            collocation_rng.uniform(
                0.0,
                float(case["output_times"][-1]),
                size=(selected_count, collocation_count),
            ),
            axis=1,
        ),
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.Adam(
        [normalized], lr=float(inverse["parameter_learning_rate"])
    )
    trace: list[dict[str, Any]] = []
    progress_path = output_dir / "progress.pt"
    completed = 0
    elapsed_before = 0.0
    if resume and progress_path.exists():
        progress = torch.load(progress_path, map_location=device, weights_only=False)
        if progress.get("signature") != signature:
            raise RuntimeError("operator progress signature mismatch")
        normalized.data.copy_(progress["normalized_parameters"])
        optimizer.load_state_dict(progress["optimizer_state_dict"])
        completed = int(progress["completed_steps"])
        elapsed_before = float(progress["elapsed_seconds"])
        trace = [dict(row) for row in progress["trace"]]
    run_start = time.perf_counter()
    deadline = run_start + max(
        0.0, float(config["maximum_seconds_per_job"]) - elapsed_before
    )
    budget_exhausted = False
    for step in range(completed + 1, adam_steps + 1):
        fraction = (step - 1) / max(adam_steps - 1, 1)
        learning_rate = float(inverse["minimum_learning_rate"]) + 0.5 * (
            float(inverse["parameter_learning_rate"])
            - float(inverse["minimum_learning_rate"])
        ) * (1.0 + math.cos(math.pi * fraction))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        objective, physical, components = operator_objective(
            model,
            normalized,
            history,
            history_grid,
            observation_times,
            observations,
            observed["states"],
            collocation_times,
            equation,
            float(inverse["physics_weight"]),
        )
        objective.sum().backward()
        optimizer.step()
        with torch.no_grad():
            normalized.clamp_(-1.0, 1.0)
        if step % int(inverse["record_interval"]) == 0 or step == adam_steps:
            selected = int(torch.argmin(objective.detach()).cpu())
            trace.append(
                {
                    "phase": "adam",
                    "step": step,
                    "elapsed_seconds": elapsed_before
                    + time.perf_counter()
                    - run_start,
                    "selected_restart": selected,
                    "selected_objective": float(objective.detach().cpu()[selected]),
                    "selected_data_loss": float(
                        components["data_loss"].detach().cpu()[selected]
                    ),
                    "selected_physics_loss": float(
                        components["physics_loss"].detach().cpu()[selected]
                    ),
                    "selected_transmission_b": float(physical.detach().cpu()[selected, 0]),
                    "selected_convexity_a": float(physical.detach().cpu()[selected, 1]),
                }
            )
        if step % int(inverse["checkpoint_interval"]) == 0 or step == adam_steps:
            atomic_torch_save(
                progress_path,
                {
                    "signature": signature,
                    "normalized_parameters": normalized.detach(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "completed_steps": step,
                    "elapsed_seconds": elapsed_before
                    + time.perf_counter()
                    - run_start,
                    "trace": trace,
                },
            )
        if time.perf_counter() >= deadline:
            budget_exhausted = True
            break
    if not budget_exhausted:
        normalized, budget_exhausted = refine_operator_lbfgsb(
            model,
            normalized,
            history,
            history_grid,
            observation_times,
            observations,
            observed["states"],
            collocation_times,
            equation,
            inverse,
            lbfgsb_iterations,
            output_dir / "lbfgsb_results.csv",
            deadline,
        )
    objective, physical, components = operator_objective(
        model,
        normalized,
        history,
        history_grid,
        observation_times,
        observations,
        observed["states"],
        collocation_times,
        equation,
        float(inverse["physics_weight"]),
    )
    optimization_seconds = elapsed_before + time.perf_counter() - started
    write_csv(output_dir / "trace.csv", trace)
    np.save(output_dir / "candidate_parameters.npy", physical.detach().cpu().numpy())
    selected_index = int(torch.argmin(objective.detach()).cpu().item())
    return finalize_result(
        "operator_projected_grid",
        scenario_key,
        case_index,
        experiment_seed,
        signature,
        "budget_exhausted" if budget_exhausted else "completed",
        physical.detach().cpu().numpy(),
        objective.detach().cpu().numpy(),
        case["true_parameters"],
        np.asarray(equation.parameter_bounds),
        case,
        equation,
        config,
        optimization_seconds,
        output_dir,
        {
            "selected_data_loss": float(
                components["data_loss"].detach().cpu()[selected_index]
            ),
            "selected_physics_loss": float(
                components["physics_loss"].detach().cpu()[selected_index]
            ),
            "grid_candidate_count": first_points * second_points,
            "checkpoint_model_type": checkpoint_identity["model_type"],
            "scalar_observation_count": int(observed["scalar_observation_count"]),
            "online_end_to_end_seconds": time.perf_counter() - online_started,
        },
    )


def direct_solver_residuals(
    parameters: np.ndarray,
    case: Mapping[str, np.ndarray],
    observed: Mapping[str, Any],
    equation: DelayedSEIConfig,
    internal_step: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized observation residuals and full direct trajectories."""
    candidates = np.asarray(parameters, dtype=np.float64)
    base_histories = np.asarray(observed["histories"], dtype=np.float64)
    history_count = base_histories.shape[0]
    histories = np.broadcast_to(
        base_histories[None, ...],
        (candidates.shape[0],) + base_histories.shape,
    ).reshape(candidates.shape[0] * history_count, STATE_DIM, base_histories.shape[-1])
    repeated_parameters = np.repeat(candidates, history_count, axis=0)
    trajectories = solve_batch(
        histories,
        repeated_parameters,
        np.asarray(case["history_grid"], dtype=np.float64),
        np.asarray(case["output_times"], dtype=np.float64),
        equation,
        float(internal_step),
    ).astype(np.float64).reshape(
        candidates.shape[0], history_count, len(case["output_times"]), STATE_DIM
    )
    states = np.asarray(observed["states"], dtype=np.int64)
    predicted = trajectories[:, :, np.asarray(observed["indices"]), :][:, :, :, states]
    target = np.asarray(observed["values"], dtype=np.float64)[:, :, states]
    residuals = predicted - target[None, ...]
    return residuals.reshape(candidates.shape[0], -1) / math.sqrt(
        float(np.prod(residuals.shape[1:]))
    ), trajectories


def shared_initialization_seed(
    config: Mapping[str, Any], experiment_seed: int, case_index: int
) -> int:
    """Return the identical LHS seed used by LM and PINN-DDE."""
    return (
        int(experiment_seed)
        + int(config["randomness"]["initialization_offset"])
        + int(case_index) * 1000
    )


def run_lm_job(
    scenario_key: str,
    case_index: int,
    experiment_seed: int,
    config: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    checkpoint_identity: Mapping[str, Any],
    archive_path: Path,
    output_dir: Path,
    smoke: bool,
    resume: bool,
) -> dict[str, Any]:
    """Run direct numerical box-projected Levenberg--Marquardt."""
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = job_signature(
        "projected_lm",
        scenario_key,
        case_index,
        experiment_seed,
        config,
        checkpoint_identity,
        smoke,
    )
    result_path = output_dir / "result.json"
    if resume and result_path.exists():
        result = read_json(result_path)
        if result.get("signature") != signature:
            raise RuntimeError("completed LM result signature mismatch")
        return result
    online_started = time.perf_counter()
    setup_logger(output_dir / "job.log")
    equation = DelayedSEIConfig.from_mapping(
        checkpoint["resolved_config"]["equation"]
    )
    case = load_case(archive_path, case_index)
    observed = condition_observations(case, condition_by_key(config, scenario_key))
    lm = config["projected_lm"]
    restart_count = int(
        config["smoke"]["pinndde_random_initializations"]
        if smoke
        else lm["random_initializations"]
    )
    maximum_iterations = int(
        config["smoke"]["lm_iterations"] if smoke else lm["max_iterations"]
    )
    bounds = np.asarray(equation.parameter_bounds, dtype=np.float64)
    parameters = lhs_physical(
        restart_count,
        shared_initialization_seed(config, experiment_seed, case_index),
        bounds,
    )
    write_csv(
        output_dir / "initial_lhs_points.csv",
        [
            {
                "restart_index": index,
                "transmission_b": float(row[0]),
                "convexity_a": float(row[1]),
            }
            for index, row in enumerate(parameters)
        ],
    )
    damping = np.full(restart_count, float(lm["initial_damping"]), dtype=np.float64)
    completed = 0
    elapsed_before = 0.0
    trace: list[dict[str, Any]] = []
    progress_path = output_dir / "progress.pt"
    solver_batch_calls = 0
    trajectory_solves = 0
    if resume and progress_path.exists():
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        if progress.get("signature") != signature:
            raise RuntimeError("LM progress signature mismatch")
        parameters = np.asarray(progress["parameters"], dtype=np.float64)
        damping = np.asarray(progress["damping"], dtype=np.float64)
        completed = int(progress["completed_iterations"])
        elapsed_before = float(progress["elapsed_seconds"])
        trace = [dict(row) for row in progress["trace"]]
        solver_batch_calls = int(progress["solver_batch_calls"])
        trajectory_solves = int(progress["trajectory_solves"])
    internal_step = float(lm["internal_step"])
    finite_step = float(lm["finite_difference_step"])
    started = time.perf_counter()
    budget_exhausted = False
    while completed < maximum_iterations:
        residuals, _ = direct_solver_residuals(
            parameters, case, observed, equation, internal_step
        )
        solver_batch_calls += 1
        trajectory_solves += restart_count * int(observed["history_count"])
        objectives = np.sum(residuals**2, axis=1)
        if completed % int(lm["record_interval"]) == 0:
            selected = int(np.argmin(objectives))
            trace.append(
                {
                    "phase": "lm",
                    "iteration": completed,
                    "elapsed_seconds": elapsed_before + time.perf_counter() - started,
                    "selected_restart": selected,
                    "selected_objective": float(objectives[selected]),
                    "selected_transmission_b": float(parameters[selected, 0]),
                    "selected_convexity_a": float(parameters[selected, 1]),
                    "mean_damping": float(np.mean(damping)),
                }
            )
        perturbed = np.repeat(parameters[None, :, :], 4, axis=0)
        for parameter_index in range(2):
            perturbed[2 * parameter_index, :, parameter_index] = np.minimum(
                bounds[parameter_index, 1],
                parameters[:, parameter_index] + finite_step,
            )
            perturbed[2 * parameter_index + 1, :, parameter_index] = np.maximum(
                bounds[parameter_index, 0],
                parameters[:, parameter_index] - finite_step,
            )
        perturbed_residuals, _ = direct_solver_residuals(
            perturbed.reshape(4 * restart_count, 2),
            case,
            observed,
            equation,
            internal_step,
        )
        solver_batch_calls += 1
        trajectory_solves += 4 * restart_count * int(observed["history_count"])
        perturbed_residuals = perturbed_residuals.reshape(
            4, restart_count, residuals.shape[1]
        )
        jacobians = np.empty(
            (restart_count, residuals.shape[1], 2), dtype=np.float64
        )
        for parameter_index in range(2):
            denominator = (
                perturbed[2 * parameter_index, :, parameter_index]
                - perturbed[2 * parameter_index + 1, :, parameter_index]
            )
            jacobians[:, :, parameter_index] = (
                perturbed_residuals[2 * parameter_index]
                - perturbed_residuals[2 * parameter_index + 1]
            ) / denominator[:, None]
        proposed = parameters.copy()
        for restart in range(restart_count):
            jacobian = jacobians[restart]
            normal = jacobian.T @ jacobian
            scaling = np.maximum(np.diag(normal), 1.0e-12)
            system = normal + damping[restart] * np.diag(scaling)
            gradient = jacobian.T @ residuals[restart]
            try:
                delta = np.linalg.solve(system, -gradient)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(system, -gradient, rcond=None)[0]
            proposed[restart] = np.clip(
                parameters[restart] + delta, bounds[:, 0], bounds[:, 1]
            )
        candidate_residuals, _ = direct_solver_residuals(
            proposed, case, observed, equation, internal_step
        )
        solver_batch_calls += 1
        trajectory_solves += restart_count * int(observed["history_count"])
        candidate_objectives = np.sum(candidate_residuals**2, axis=1)
        accepted = candidate_objectives < objectives
        parameters[accepted] = proposed[accepted]
        damping[accepted] = np.maximum(
            float(lm["minimum_damping"]),
            damping[accepted] / float(lm["damping_decrease"]),
        )
        damping[~accepted] = np.minimum(
            float(lm["maximum_damping"]),
            damping[~accepted] * float(lm["damping_increase"]),
        )
        completed += 1
        elapsed = elapsed_before + time.perf_counter() - started
        if completed % int(lm["checkpoint_interval"]) == 0 or completed == maximum_iterations:
            atomic_torch_save(
                progress_path,
                {
                    "signature": signature,
                    "parameters": parameters,
                    "damping": damping,
                    "completed_iterations": completed,
                    "elapsed_seconds": elapsed,
                    "trace": trace,
                    "solver_batch_calls": solver_batch_calls,
                    "trajectory_solves": trajectory_solves,
                },
            )
        if elapsed >= float(config["maximum_seconds_per_job"]):
            budget_exhausted = True
            break
    residuals, _ = direct_solver_residuals(
        parameters, case, observed, equation, internal_step
    )
    solver_batch_calls += 1
    trajectory_solves += restart_count * int(observed["history_count"])
    objectives = np.sum(residuals**2, axis=1)
    optimization_seconds = elapsed_before + time.perf_counter() - started
    write_csv(output_dir / "trace.csv", trace)
    np.save(output_dir / "candidate_parameters.npy", parameters)
    return finalize_result(
        "projected_lm",
        scenario_key,
        case_index,
        experiment_seed,
        signature,
        "budget_exhausted" if budget_exhausted else "completed",
        parameters,
        objectives,
        case["true_parameters"],
        bounds,
        case,
        equation,
        config,
        optimization_seconds,
        output_dir,
        {
            "solver_batch_calls": solver_batch_calls,
            "direct_trajectory_solves": trajectory_solves,
            "lm_final_damping_mean": float(np.mean(damping)),
            "inverse_internal_step": internal_step,
            "finite_difference_step": finite_step,
            "scalar_observation_count": int(observed["scalar_observation_count"]),
            "online_end_to_end_seconds": time.perf_counter() - online_started,
        },
    )


class ScalarPINN(nn.Module):
    """One time-normalized scalar Tanh network for one population state."""

    def __init__(self, hidden_widths: Sequence[int], horizon: float, seed: int) -> None:
        super().__init__()
        widths = [1, *(int(value) for value in hidden_widths), 1]
        if len(widths) != 5 or any(value < 1 for value in widths) or horizon <= 0.0:
            raise ValueError("ScalarPINN requires three widths and a positive horizon")
        self.horizon = float(horizon)
        layers: list[nn.Module] = []
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            for index, (input_width, output_width) in enumerate(
                zip(widths[:-1], widths[1:])
            ):
                linear = nn.Linear(input_width, output_width)
                nn.init.xavier_normal_(linear.weight)
                nn.init.zeros_(linear.bias)
                layers.append(linear)
                if index < len(widths) - 2:
                    layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        if times.ndim != 2 or times.shape[1] != 1:
            raise ValueError("ScalarPINN times must have shape [points,1]")
        normalized_time = 2.0 * times / self.horizon - 1.0
        return self.network(normalized_time)


class MultiStartPINNDDE(nn.Module):
    """Independent state networks per restart/history with shared parameters."""

    def __init__(
        self,
        restart_count: int,
        hidden_widths: Sequence[int],
        horizon: float,
        initial_raw_parameters: torch.Tensor,
        parameter_bounds: Sequence[Sequence[float]],
        seed: int,
        history_count: int = 1,
    ) -> None:
        super().__init__()
        if (
            restart_count < 1
            or history_count < 1
            or initial_raw_parameters.shape != (restart_count, 2)
        ):
            raise ValueError("PINN restart and parameter shapes disagree")
        self.restart_count = int(restart_count)
        self.history_count = int(history_count)
        self.parameter_bounds = tuple(
            tuple(float(value) for value in pair) for pair in parameter_bounds
        )
        self.networks = nn.ModuleList(
            nn.ModuleList(
                nn.ModuleList(
                    ScalarPINN(
                        hidden_widths,
                        horizon,
                        int(seed) + restart * 10_000 + history * 100 + state,
                    )
                    for state in range(STATE_DIM)
                )
                for history in range(history_count)
            )
            for restart in range(restart_count)
        )
        self.raw_parameters = nn.Parameter(initial_raw_parameters.detach().clone())

    def physical_parameters(self) -> torch.Tensor:
        return bounded_parameters(self.raw_parameters, self.parameter_bounds)

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        if (
            times.ndim != 3
            or times.shape[0] != self.restart_count
            or times.shape[1] != self.history_count
        ):
            raise ValueError("PINN-DDE times must have shape [restarts,histories,points]")
        outputs = []
        for restart, history_networks in enumerate(self.networks):
            history_outputs = []
            for history, networks in enumerate(history_networks):
                one_time = times[restart, history].unsqueeze(-1)
                history_outputs.append(
                    torch.cat([network(one_time) for network in networks], dim=-1)
                )
            outputs.append(torch.stack(history_outputs, dim=0))
        return torch.stack(outputs, dim=0)


def interpolate_pinn_history(
    histories: torch.Tensor,
    history_grid: torch.Tensor,
    query_times: torch.Tensor,
) -> torch.Tensor:
    """Linearly interpolate restart/history-specific three-state histories."""
    if histories.ndim != 4 or histories.shape[2] != STATE_DIM:
        raise ValueError("histories must be [R,K,3,M]")
    if query_times.shape[:2] != histories.shape[:2]:
        raise ValueError("query times and histories disagree")
    clipped = query_times.clamp(float(history_grid[0]), float(history_grid[-1]))
    upper = torch.searchsorted(
        history_grid, clipped.contiguous(), right=True
    ).clamp(1, history_grid.numel() - 1)
    lower = upper - 1
    restart_count, history_count, _, sensor_count = histories.shape
    point_count = query_times.shape[2]
    values = histories.permute(0, 1, 3, 2).reshape(
        restart_count * history_count, sensor_count, STATE_DIM
    )
    flat_lower = lower.reshape(restart_count * history_count, point_count)
    flat_upper = upper.reshape(restart_count * history_count, point_count)
    shape = (*lower.shape, STATE_DIM)
    flat_shape = (restart_count * history_count, point_count, STATE_DIM)
    left = torch.gather(values, 1, flat_lower.unsqueeze(-1).expand(flat_shape)).reshape(shape)
    right = torch.gather(values, 1, flat_upper.unsqueeze(-1).expand(flat_shape)).reshape(shape)
    weight = (clipped - history_grid[lower]) / (
        history_grid[upper] - history_grid[lower]
    ).clamp_min(1.0e-12)
    return left + weight.unsqueeze(-1) * (right - left)


def pinn_delayed_state(
    model: MultiStartPINNDDE,
    histories: torch.Tensor,
    history_grid: torch.Tensor,
    times: torch.Tensor,
    equation: DelayedSEIConfig,
) -> torch.Tensor:
    """Evaluate x(t-delay) with the same fixed delay as the RK4 solver."""
    delayed_times = times - float(equation.delay)
    historical = interpolate_pinn_history(histories, history_grid, delayed_times)
    predicted = model(delayed_times.clamp_min(0.0))
    return torch.where(
        (delayed_times <= 0.0).unsqueeze(-1), historical, predicted
    )


def pinndde_losses(
    model: MultiStartPINNDDE,
    histories: torch.Tensor,
    history_grid: torch.Tensor,
    collocation_times: torch.Tensor,
    observation_times: torch.Tensor,
    observations: torch.Tensor,
    observed_states: Sequence[int],
    equation: DelayedSEIConfig,
    adaptive_weights: bool,
    detach_adaptive_weights: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute Delayed SEI constant-delay physics, initial and observation losses."""
    if not collocation_times.requires_grad:
        raise ValueError("PINN collocation times must require gradients")
    current = model(collocation_times)
    derivatives = []
    for state in range(STATE_DIM):
        derivatives.append(
            torch.autograd.grad(
                current[..., state].sum(),
                collocation_times,
                create_graph=True,
                retain_graph=True,
            )[0]
        )
    time_derivative = torch.stack(derivatives, dim=-1)
    delayed = pinn_delayed_state(
        model, histories, history_grid, collocation_times, equation
    )
    restart_count, history_count, point_count, _ = current.shape
    flat_current = current.reshape(restart_count * history_count, point_count, STATE_DIM)
    flat_delayed = delayed.reshape(restart_count * history_count, point_count, STATE_DIM)
    flat_parameters = model.physical_parameters().unsqueeze(1).expand(
        -1, history_count, -1
    ).reshape(restart_count * history_count, 2)
    residual = time_derivative - rhs_torch(
        flat_current, flat_delayed, flat_parameters, equation
    ).reshape_as(current)
    physics_by_state = torch.mean(residual**2, dim=(1, 2))
    zero_times = torch.zeros(
        (model.restart_count, model.history_count, 1),
        dtype=collocation_times.dtype,
        device=collocation_times.device,
    )
    initial_by_state = (
        model(zero_times)[:, :, 0, :] - histories[:, :, :, -1]
    ) ** 2
    initial_by_state = initial_by_state.mean(dim=1)
    predicted_observations = model(observation_times)
    states = torch.as_tensor(
        observed_states, dtype=torch.long, device=collocation_times.device
    )
    data_by_state = torch.mean(
        (
            predicted_observations.index_select(3, states)
            - observations.index_select(3, states)
        )
        ** 2,
        dim=(1, 2),
    )
    components = torch.cat(
        (physics_by_state, initial_by_state, data_by_state), dim=1
    )
    if adaptive_weights:
        weights = components / components.sum(dim=1, keepdim=True).clamp_min(1.0e-15)
        if detach_adaptive_weights:
            weights = weights.detach()
        objective = torch.sum(weights * components, dim=1)
    else:
        objective = torch.sum(components, dim=1)
    summaries = {
        "physics_loss": physics_by_state.mean(dim=1),
        "initial_loss": initial_by_state.mean(dim=1),
        "data_loss": data_by_state.mean(dim=1),
    }
    if not torch.all(torch.isfinite(objective)):
        raise FloatingPointError("non-finite PINN-DDE objective")
    return objective, summaries


def run_pinndde_job(
    scenario_key: str,
    case_index: int,
    experiment_seed: int,
    config: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    checkpoint_identity: Mapping[str, Any],
    archive_path: Path,
    device_name: str,
    output_dir: Path,
    smoke: bool,
    resume: bool,
) -> dict[str, Any]:
    """Train a fresh multi-start constant-delay Delayed SEI PINN for one case."""
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = job_signature(
        "pinndde",
        scenario_key,
        case_index,
        experiment_seed,
        config,
        checkpoint_identity,
        smoke,
    )
    result_path = output_dir / "result.json"
    if resume and result_path.exists():
        result = read_json(result_path)
        if result.get("signature") != signature:
            raise RuntimeError("completed PINN-DDE result signature mismatch")
        return result
    online_started = time.perf_counter()
    setup_logger(output_dir / "job.log")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    equation = DelayedSEIConfig.from_mapping(
        checkpoint["resolved_config"]["equation"]
    )
    case = load_case(archive_path, case_index)
    observed = condition_observations(case, condition_by_key(config, scenario_key))
    pinn = config["pinndde"]
    smoke_config = config["smoke"]
    restart_count = int(
        smoke_config["pinndde_random_initializations"]
        if smoke
        else pinn["random_initializations"]
    )
    collocation_count = int(
        smoke_config["pinndde_collocation_points"]
        if smoke
        else pinn["collocation_points"]
    )
    adam_steps = int(
        smoke_config["pinndde_adam_steps"] if smoke else pinn["adam_steps"]
    )
    lbfgs_steps = int(
        smoke_config["pinndde_lbfgs_steps"] if smoke else pinn["lbfgs_steps"]
    )
    bounds = np.asarray(equation.parameter_bounds, dtype=np.float64)
    initial_physical = lhs_physical(
        restart_count,
        shared_initialization_seed(config, experiment_seed, case_index),
        bounds,
    )
    write_csv(
        output_dir / "initial_lhs_points.csv",
        [
            {
                "restart_index": index,
                "transmission_b": float(row[0]),
                "convexity_a": float(row[1]),
            }
            for index, row in enumerate(initial_physical)
        ],
    )
    raw = torch.as_tensor(
        physical_to_logits(initial_physical, bounds),
        dtype=torch.float32,
        device=device,
    )
    model = MultiStartPINNDDE(
        restart_count,
        pinn["hidden_widths"],
        float(case["output_times"][-1]),
        raw,
        equation.parameter_bounds,
        shared_initialization_seed(config, experiment_seed, case_index) + 10_000,
        int(observed["history_count"]),
    ).to(device)
    histories = torch.as_tensor(
        observed["histories"], dtype=torch.float32, device=device
    ).unsqueeze(0).expand(restart_count, -1, -1, -1).contiguous()
    history_grid = torch.as_tensor(
        case["history_grid"], dtype=torch.float32, device=device
    )
    observation_times = torch.as_tensor(
        observed["times"], dtype=torch.float32, device=device
    ).reshape(1, 1, -1).expand(
        restart_count, int(observed["history_count"]), -1
    ).contiguous()
    observations = torch.as_tensor(
        observed["values"], dtype=torch.float32, device=device
    ).unsqueeze(0).expand(restart_count, -1, -1, -1).contiguous()
    collocation_rng = np.random.default_rng(
        experiment_seed
        + int(config["randomness"]["collocation_offset"])
        + case_index * 100
        + 77
    )
    one_collocation = np.sort(
        collocation_rng.uniform(
            0.0, float(case["output_times"][-1]), size=collocation_count
        ).astype(np.float32)
    )
    collocation_base = torch.as_tensor(
        one_collocation, dtype=torch.float32, device=device
    ).reshape(1, 1, -1).expand(
        restart_count, int(observed["history_count"]), -1
    ).contiguous()
    adam = torch.optim.Adam(model.parameters(), lr=float(pinn["adam_learning_rate"]))
    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=float(pinn["lbfgs_learning_rate"]),
        max_iter=1,
        history_size=int(pinn["lbfgs_history_size"]),
        line_search_fn="strong_wolfe",
    )
    adam_completed = 0
    lbfgs_completed = 0
    elapsed_before = 0.0
    trace: list[dict[str, Any]] = []
    progress_path = output_dir / "progress.pt"
    if resume and progress_path.exists():
        progress = torch.load(progress_path, map_location=device, weights_only=False)
        if progress.get("signature") != signature:
            raise RuntimeError("PINN-DDE progress signature mismatch")
        model.load_state_dict(progress["model_state_dict"], strict=True)
        adam.load_state_dict(progress["adam_optimizer_state_dict"])
        lbfgs.load_state_dict(progress["lbfgs_optimizer_state_dict"])
        adam_completed = int(progress["adam_completed"])
        lbfgs_completed = int(progress["lbfgs_completed"])
        elapsed_before = float(progress["elapsed_seconds"])
        trace = [dict(row) for row in progress["trace"]]

    def evaluate_loss() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        collocation = collocation_base.detach().clone().requires_grad_(True)
        return pinndde_losses(
            model,
            histories,
            history_grid,
            collocation,
            observation_times,
            observations,
            observed["states"],
            equation,
            bool(pinn["adaptive_loss_weights"]),
            bool(pinn["detach_adaptive_weights"]),
        )

    def record(phase: str, step: int, elapsed: float) -> None:
        objective, components = evaluate_loss()
        parameters = model.physical_parameters()
        selected = int(torch.argmin(objective.detach()).cpu())
        trace.append(
            {
                "phase": phase,
                "step": int(step),
                "elapsed_seconds": float(elapsed),
                "selected_restart": selected,
                "selected_objective": float(objective.detach().cpu()[selected]),
                "selected_data_loss": float(
                    components["data_loss"].detach().cpu()[selected]
                ),
                "selected_physics_loss": float(
                    components["physics_loss"].detach().cpu()[selected]
                ),
                "selected_initial_loss": float(
                    components["initial_loss"].detach().cpu()[selected]
                ),
                "selected_transmission_b": float(parameters.detach().cpu()[selected, 0]),
                "selected_convexity_a": float(parameters.detach().cpu()[selected, 1]),
            }
        )

    started = time.perf_counter()
    budget_exhausted = False
    for step in range(adam_completed + 1, adam_steps + 1):
        adam.zero_grad(set_to_none=True)
        objective, _ = evaluate_loss()
        objective.sum().backward()
        adam.step()
        elapsed = elapsed_before + time.perf_counter() - started
        if step % int(pinn["record_interval"]) == 0 or step == adam_steps:
            record("adam", step, elapsed)
        if step % int(pinn["checkpoint_interval"]) == 0 or step == adam_steps:
            atomic_torch_save(
                progress_path,
                {
                    "signature": signature,
                    "model_state_dict": model.state_dict(),
                    "adam_optimizer_state_dict": adam.state_dict(),
                    "lbfgs_optimizer_state_dict": lbfgs.state_dict(),
                    "adam_completed": step,
                    "lbfgs_completed": lbfgs_completed,
                    "elapsed_seconds": elapsed,
                    "trace": trace,
                },
            )
        adam_completed = step
        if elapsed >= float(config["maximum_seconds_per_job"]):
            budget_exhausted = True
            break
    if not budget_exhausted:
        for step in range(lbfgs_completed + 1, lbfgs_steps + 1):

            def closure() -> torch.Tensor:
                lbfgs.zero_grad(set_to_none=True)
                objective, _ = evaluate_loss()
                total = objective.sum()
                total.backward()
                return total

            lbfgs.step(closure)
            elapsed = elapsed_before + time.perf_counter() - started
            if step % int(pinn["record_interval"]) == 0 or step == lbfgs_steps:
                record("lbfgs", step, elapsed)
            if step % int(pinn["checkpoint_interval"]) == 0 or step == lbfgs_steps:
                atomic_torch_save(
                    progress_path,
                    {
                        "signature": signature,
                        "model_state_dict": model.state_dict(),
                        "adam_optimizer_state_dict": adam.state_dict(),
                        "lbfgs_optimizer_state_dict": lbfgs.state_dict(),
                        "adam_completed": adam_completed,
                        "lbfgs_completed": step,
                        "elapsed_seconds": elapsed,
                        "trace": trace,
                    },
                )
            lbfgs_completed = step
            if elapsed >= float(config["maximum_seconds_per_job"]):
                budget_exhausted = True
                break
    objective, components = evaluate_loss()
    physical = model.physical_parameters()
    selected = int(torch.argmin(objective.detach()).cpu())
    all_times = torch.as_tensor(
        case["output_times"], dtype=torch.float32, device=device
    ).reshape(1, 1, -1).expand(
        restart_count, int(observed["history_count"]), -1
    )
    with torch.no_grad():
        pinn_trajectories = model(all_times).cpu().numpy()
    np.save(output_dir / "pinn_reconstruction.npy", pinn_trajectories[selected])
    write_csv(output_dir / "trace.csv", trace)
    optimization_seconds = elapsed_before + time.perf_counter() - started
    result = finalize_result(
        "pinndde",
        scenario_key,
        case_index,
        experiment_seed,
        signature,
        "budget_exhausted" if budget_exhausted else "completed",
        physical.detach().cpu().numpy(),
        objective.detach().cpu().numpy(),
        case["true_parameters"],
        bounds,
        case,
        equation,
        config,
        optimization_seconds,
        output_dir,
        {
            "selected_data_loss": float(
                components["data_loss"].detach().cpu()[selected]
            ),
            "selected_physics_loss": float(
                components["physics_loss"].detach().cpu()[selected]
            ),
            "selected_initial_loss": float(
                components["initial_loss"].detach().cpu()[selected]
            ),
            "pinn_reconstruction_relative_l2": trajectory_relative_l2(
                observed["reference"], pinn_trajectories[selected]
            ),
            "adam_steps_completed": adam_completed,
            "lbfgs_steps_completed": lbfgs_completed,
            "scalar_observation_count": int(observed["scalar_observation_count"]),
            "online_end_to_end_seconds": time.perf_counter() - online_started,
        },
    )
    return result


def lightweight_checkpoint(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only scientific metadata needed by direct LM and PINN workers."""
    return {
        "network_format": checkpoint["network_format"],
        "model_type": checkpoint["model_type"],
        "model_config": dict(checkpoint["model_config"]),
        "resolved_config": {
            key: dict(checkpoint["resolved_config"][key])
            for key in ("data", "equation", "operator")
        },
    }


def job_output_dir(stage_dir: Path, job: Mapping[str, Any]) -> Path:
    """Return the stable experiment/condition/case/method output directory."""
    return (
        stage_dir
        / "jobs"
        / str(job["experiment"])
        / str(job["condition"])
        / f"case_{int(job['case_index']):03d}"
        / str(job["method"])
    )


def queue_worker(
    device_name: str,
    jobs: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    checkpoint_path_text: str,
    checkpoint_identity: Mapping[str, Any],
    checkpoint_science: Mapping[str, Any],
    archive_path_text: str,
    stage_dir_text: str,
    smoke: bool,
    resume: bool,
) -> list[dict[str, Any]]:
    """Run one long-lived deterministic queue on one logical device."""
    checkpoint_path = Path(checkpoint_path_text)
    archive_path = Path(archive_path_text)
    stage_dir = Path(stage_dir_text)
    results: list[dict[str, Any]] = []
    for job in jobs:
        output_dir = job_output_dir(stage_dir, job)
        method = str(job["method"])
        common = (
            str(job["condition"]),
            int(job["case_index"]),
            int(job["experiment_seed"]),
            config,
        )
        if method == "operator_projected_grid":
            result = run_operator_job(
                *common,
                checkpoint_path,
                checkpoint_identity,
                archive_path,
                device_name,
                output_dir,
                smoke,
                resume,
            )
        elif method == "projected_lm":
            result = run_lm_job(
                *common,
                checkpoint_science,
                checkpoint_identity,
                archive_path,
                output_dir,
                smoke,
                resume,
            )
        elif method == "pinndde":
            result = run_pinndde_job(
                *common,
                checkpoint_science,
                checkpoint_identity,
                archive_path,
                device_name,
                output_dir,
                smoke,
                resume,
            )
        else:
            raise ValueError(f"unknown method: {method}")
        results.append(result)
        gc.collect()
        if device_name.startswith("cuda"):
            torch.cuda.empty_cache()
    return results


def build_round_robin_queues(
    jobs: Sequence[Mapping[str, Any]], worker_count: int
) -> list[list[Mapping[str, Any]]]:
    """Distribute ordered jobs over persistent workers without duplication."""
    if not jobs or not 1 <= worker_count <= len(jobs):
        raise ValueError("worker_count must lie in [1,len(jobs)]")
    return [list(jobs[index::worker_count]) for index in range(worker_count)]


def summarize_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize one method/condition group using cases as statistical units."""
    metrics = (
        "parameter_normalized_rmse",
        "parameter_physical_rmse",
        "transmission_b_absolute_error",
        "convexity_a_absolute_error",
        "direct_reconstruction_relative_l2",
        "optimization_seconds",
        "online_end_to_end_seconds",
    )
    result: dict[str, Any] = {"case_count": len(rows)}
    for metric in metrics:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        result[f"{metric}_mean"] = float(np.mean(values))
        result[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        result[f"{metric}_median"] = float(np.median(values))
    success_keys = sorted(
        key for key in rows[0] if str(key).startswith("success_at_")
    )
    for key in success_keys:
        result[f"{key}_rate"] = float(np.mean([float(row[key]) for row in rows]))
    return result


def save_figure(
    figure: plt.Figure, stem: Path, reporting: Mapping[str, Any], smoke: bool
) -> None:
    """Save comparison figures in configured formats."""
    formats = ["png"] if smoke else list(reporting["figure_formats"])
    dpi = 100 if smoke else int(reporting["raster_dpi"])
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in formats:
        figure.savefig(
            stem.with_suffix(f".{suffix}"), dpi=dpi, bbox_inches="tight"
        )
    plt.close(figure)


def plot_stage_results(
    rows: Sequence[Mapping[str, Any]], stage_dir: Path, config: Mapping[str, Any], smoke: bool
) -> None:
    """Plot whichever noise and multi-history conditions were selected."""
    methods = active_methods(config)
    conditions = active_condition_definitions(config)
    selected_keys = {item["condition"] for item in conditions}
    noise_modes = [
        mode
        for mode in config["noise_escalation"]["observation_modes"]
        if any(
            f"{mode}_{noise_tag(float(level))}" in selected_keys
            for level in config["noise_escalation"]["noise_standard_deviations"]
        )
    ]
    multi_conditions = [
        key
        for key in config["multi_history"]["conditions"]
        if key in selected_keys
    ]
    for metric, label, suffix in (
        ("parameter_normalized_rmse", "Parameter NRMSE", "parameter_error"),
        ("optimization_seconds", "Optimization seconds", "runtime"),
    ):
        if noise_modes:
            figure, axes = plt.subplots(
                len(noise_modes), 1, figsize=(8, 4 * len(noise_modes)), squeeze=False
            )
            for axis, mode in zip(axes[:, 0], noise_modes):
                levels = [
                    float(level)
                    for level in config["noise_escalation"]["noise_standard_deviations"]
                    if f"{mode}_{noise_tag(float(level))}" in selected_keys
                ]
                for method in methods:
                    medians = []
                    for noise in levels:
                        condition = f"{mode}_{noise_tag(noise)}"
                        values = [
                            float(row[metric])
                            for row in rows
                            if row["condition"] == condition
                            and row["method_key"] == method
                        ]
                        medians.append(float(np.median(values)))
                    axis.plot(levels, medians, marker="o", label=METHOD_LABELS[method])
                axis.set_title(
                    config["noise_escalation"]["observation_modes"][mode]["display_name"]
                )
                axis.set_ylabel(label)
                axis.set_yscale("log")
                axis.grid(True, alpha=0.25)
            axes[-1, 0].set_xlabel("Noise standard deviation")
            axes[0, 0].legend(fontsize=7)
            figure.tight_layout()
            save_figure(
                figure,
                stage_dir / "figures" / f"noise_{suffix}",
                config["reporting"],
                smoke,
            )

        if multi_conditions:
            figure, axes = plt.subplots(
                1,
                len(multi_conditions),
                figsize=(5.5 * len(multi_conditions), 4),
                squeeze=False,
            )
            for axis, condition in zip(axes[0], multi_conditions):
                values = [
                    [
                        float(row[metric])
                        for row in rows
                        if row["condition"] == condition
                        and row["method_key"] == method
                    ]
                    for method in methods
                ]
                labels = [METHOD_LABELS[method] for method in methods]
                try:
                    axis.boxplot(values, tick_labels=labels)
                except TypeError:
                    axis.boxplot(values, labels=labels)
                axis.set_title(condition.replace("_", " ").title())
                axis.set_ylabel(label)
                axis.set_yscale("log")
                axis.tick_params(axis="x", rotation=20)
                axis.grid(True, alpha=0.25)
            figure.tight_layout()
            save_figure(
                figure,
                stage_dir / "figures" / f"multi_history_{suffix}",
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
    """Require every job result and build per-case and grouped comparisons."""
    rows: list[dict[str, Any]] = []
    conditions = active_condition_definitions(config)
    methods = active_methods(config)
    for condition in conditions:
        for case_index in range(case_count):
            for method in methods:
                path = (
                    stage_dir
                    / "jobs"
                    / str(condition["experiment"])
                    / str(condition["condition"])
                    / f"case_{case_index:03d}"
                    / method
                    / "result.json"
                )
                if not path.exists():
                    raise RuntimeError(f"missing completed job result: {path}")
                rows.append(read_json(path))
    write_csv(stage_dir / "per_case_results.csv", rows)
    summary_rows: list[dict[str, Any]] = []
    for condition in conditions:
        for method in methods:
            group = [
                row
                for row in rows
                if row["condition"] == condition["condition"] and row["method_key"] == method
            ]
            if len(group) != case_count:
                raise RuntimeError("condition/method case count is incomplete")
            summary_rows.append(
                {
                    "experiment_seed": int(experiment_seed),
                    "experiment": condition["experiment"],
                    "condition": condition["condition"],
                    "history_count": condition["history_count"],
                    "noise_standard_deviation": condition["noise_standard_deviation"],
                    "method_key": method,
                    "method": METHOD_LABELS[method],
                    **summarize_group(group),
                }
            )
    write_csv(stage_dir / "comparison.csv", summary_rows)
    summary = {
        "experiment_seed": int(experiment_seed),
        "smoke": bool(smoke),
        "case_count": int(case_count),
        "condition_count": len(conditions),
        "job_count": len(rows),
        "rows": summary_rows,
        "selection_rule": "Each method selects only by its own visible inverse objective; truth is evaluation-only.",
        "shared_data_rule": "All methods share four histories per case, truth parameters, observation indices, paired noise arrays and physical bounds.",
        "operator_implementation": "Normalized Cartesian screen, top-K projected Adam, then independently bounded L-BFGS-B.",
        "lm_implementation": "Direct causal RK4, centered finite-difference residual Jacobian, damped normal equations and box projection.",
        "pinndde_implementation": "Three scalar Tanh networks per history and restart, shared physical parameters, Delayed SEI residual, Adam then PyTorch L-BFGS.",
    }
    write_json(stage_dir / "comparison.json", summary)
    plot_stage_results(rows, stage_dir, config, smoke)
    return summary


def run_seed_stage(
    stage_dir: Path,
    config: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    checkpoint_identity: Mapping[str, Any],
    checkpoint_path: Path,
    experiment_seed: int,
    devices: Sequence[str],
    processes: int,
    smoke: bool,
    resume: bool,
) -> dict[str, Any]:
    """Generate shared cases, dispatch selected methods and aggregate one seed."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    case_count = int(config["smoke"]["case_count"] if smoke else config["case_count"])
    archive = prepare_shared_cases(
        stage_dir / "shared_cases",
        config,
        checkpoint,
        checkpoint_identity,
        experiment_seed,
        smoke,
        resume,
    )
    jobs = [
        {
            "method": method,
            "experiment": condition["experiment"],
            "condition": condition["condition"],
            "case_index": case_index,
            "experiment_seed": int(experiment_seed),
        }
        for condition in active_condition_definitions(config)
        for case_index in range(case_count)
        for method in active_methods(config)
    ]
    worker_count = min(len(jobs), len(devices), int(processes))
    queues = build_round_robin_queues(jobs, worker_count)
    LOGGER.info("Seed %d queues: %s", experiment_seed, [[len(q) for q in queues]])
    science = lightweight_checkpoint(checkpoint)
    if worker_count == 1:
        queue_worker(
            devices[0],
            queues[0],
            config,
            str(checkpoint_path),
            checkpoint_identity,
            science,
            str(archive),
            str(stage_dir),
            smoke,
            resume,
        )
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=worker_count, mp_context=context
        ) as executor:
            futures = [
                executor.submit(
                    queue_worker,
                    devices[index],
                    queue,
                    config,
                    str(checkpoint_path),
                    checkpoint_identity,
                    science,
                    str(archive),
                    str(stage_dir),
                    smoke,
                    resume,
                )
                for index, queue in enumerate(queues)
            ]
            for future in as_completed(futures):
                future.result()
    return aggregate_stage(
        stage_dir, config, experiment_seed, case_count, smoke
    )


def aggregate_across_seeds(
    full_dir: Path, experiment_seeds: Sequence[int], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Summarize seed-level case means without treating cases as extra seeds."""
    seed_rows: list[dict[str, Any]] = []
    for seed in experiment_seeds:
        path = full_dir / f"seed_{int(seed)}" / "comparison.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            seed_rows.extend(dict(row) for row in csv.DictReader(handle))
    write_csv(full_dir / "seed_summary_rows.csv", seed_rows)
    metrics = (
        "parameter_normalized_rmse_mean",
        "parameter_physical_rmse_mean",
        "direct_reconstruction_relative_l2_mean",
        "optimization_seconds_mean",
        "online_end_to_end_seconds_mean",
    )
    across: list[dict[str, Any]] = []
    for condition in active_condition_definitions(config):
        for method in active_methods(config):
            group = [
                row
                for row in seed_rows
                if row["condition"] == condition["condition"] and row["method_key"] == method
            ]
            summary: dict[str, Any] = {
                "experiment": condition["experiment"],
                "condition": condition["condition"],
                "method_key": method,
                "method": METHOD_LABELS[method],
                "experiment_seed_count": len(group),
            }
            for metric in metrics:
                values = np.asarray([float(row[metric]) for row in group])
                summary[f"{metric}_across_seed_mean"] = float(np.mean(values))
                summary[f"{metric}_across_seed_std"] = (
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                )
            across.append(summary)
    write_csv(full_dir / "across_seeds_comparison.csv", across)
    result = {
        "experiment_seeds": [int(seed) for seed in experiment_seeds],
        "statistical_unit": "experiment-seed case mean",
        "rows": across,
    }
    write_json(full_dir / "across_seeds_comparison.json", result)
    return result


def parse_devices(value: str, processes: int) -> list[str]:
    """Validate CPU or comma-separated logical CUDA devices."""
    if value.strip().lower() == "cpu":
        if processes != 1:
            raise ValueError("CPU mode requires --processes 1")
        return ["cpu"]
    pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
    if not pieces or any(not piece.isdigit() for piece in pieces):
        raise ValueError("--devices must be cpu or comma-separated logical GPU ids")
    values = [int(piece) for piece in pieces]
    if len(values) != len(set(values)):
        raise ValueError("logical GPU ids must be unique")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA devices requested but PyTorch reports no CUDA")
    if any(value >= torch.cuda.device_count() for value in values):
        raise ValueError("a requested logical GPU does not exist")
    if not 1 <= processes <= len(values):
        raise ValueError("processes must lie in [1,number of devices]")
    return [f"cuda:{value}" for value in values]


def notify(topic: str | None, title: str, message: str) -> None:
    """Send best-effort lifecycle notification without controlling success."""
    if not topic:
        return
    try:
        request = urllib.request.Request(
            topic,
            data=message.encode("utf-8"),
            headers={"Title": title.encode("ascii", "ignore").decode("ascii")},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read()
    except Exception:
        LOGGER.warning("ntfy notification failed", exc_info=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the runtime CLI; the trained best-model path is intentionally required."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained best_model.pt chosen after forward experiments.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs" / "experiment.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--devices", type=str, default="0")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Optional comma-separated subset of operator_projected_grid,projected_lm,pinndde.",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default=None,
        help="Optional comma-separated subset of the ten formal condition keys.",
    )
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-ntfy", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run mandatory smoke and then the selected formal inverse comparison."""
    args = build_parser().parse_args(argv)
    checkpoint_path = args.checkpoint.expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = (Path.cwd() / checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    config_path = args.config.expanduser().resolve()
    config = read_json(config_path)
    validate_config(config)
    selected_methods = parse_csv_subset(args.methods, METHODS, "--methods")
    all_condition_keys = tuple(
        item["condition"] for item in condition_definitions(config)
    )
    selected_conditions = parse_csv_subset(
        args.conditions, all_condition_keys, "--conditions"
    )
    config = dict(config)
    config["runtime_selection"] = {
        "methods": list(selected_methods),
        "conditions": list(selected_conditions),
    }
    if args.processes < 1:
        raise ValueError("processes must be positive")
    if not 1 <= args.cpus <= (os.cpu_count() or 1):
        raise ValueError(f"cpus must lie in [1,{os.cpu_count()}]")
    devices = parse_devices(args.devices, int(args.processes))
    seeds = [int(seed) for seed in config["randomness"]["experiment_seeds"]]
    if args.seed is not None:
        if int(args.seed) not in seeds:
            raise ValueError(f"seed must be one of {seeds}")
        selected_seeds = [int(args.seed)]
    else:
        selected_seeds = seeds
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(output_dir / "pipeline.log")
    existing = {path.name for path in output_dir.iterdir()}
    allowed = {"nohup.log", "pipeline.pid", "pipeline.log"}
    if existing - allowed and not args.resume:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    checkpoint, checkpoint_id = checkpoint_metadata(checkpoint_path)
    validate_checkpoint_protocol(checkpoint, config)
    runtime = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_identity": checkpoint_id,
        "selected_experiment_seeds": selected_seeds,
        "devices": devices,
        "processes": int(args.processes),
        "cpus": int(args.cpus),
        "smoke_only": bool(args.smoke_only),
        "resume": bool(args.resume),
        "selected_methods": list(selected_methods),
        "selected_conditions": list(selected_conditions),
        "conditions_per_seed": len(selected_conditions),
        "formal_jobs_per_seed": (
            len(selected_conditions) * int(config["case_count"]) * len(selected_methods)
        ),
        "selected_formal_job_count": (
            len(selected_conditions)
            * int(config["case_count"])
            * len(selected_methods)
            * len(selected_seeds)
        ),
    }
    if args.resume:
        saved_runtime = read_json(output_dir / "runtime_config.json")
        immutable_keys = (
            "checkpoint_identity",
            "selected_experiment_seeds",
            "selected_methods",
            "selected_conditions",
            "smoke_only",
        )
        if any(saved_runtime.get(key) != runtime.get(key) for key in immutable_keys):
            raise RuntimeError(
                "resume checkpoint, seed/method/condition selection or smoke mode changed"
            )
    else:
        write_json(output_dir / "runtime_config.json", runtime)
        write_json(output_dir / "experiment_config.json", config)
        write_json(
            output_dir / "environment.json",
            {
                "hostname": socket.gethostname(),
                "python": sys.version,
                "platform": platform.platform(),
                "pytorch": torch.__version__,
                "cuda": torch.version.cuda,
                "logical_cpu_count": os.cpu_count(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
        )
    topic = (
        None
        if args.no_ntfy or not config["ntfy"].get("enabled", True)
        else str(config["ntfy"]["topic"])
    )
    status: dict[str, Any] = {"success": False, "stage": "starting"}
    write_json(output_dir / "pipeline_status.json", status)
    started = time.perf_counter()
    try:
        LOGGER.info("Checkpoint: %s", checkpoint_id)
        LOGGER.info("Seeds: %s | devices=%s", selected_seeds, devices)
        smoke_marker = output_dir / "smoke" / "SMOKE_COMPLETED.json"
        if not (args.resume and smoke_marker.exists()):
            status["stage"] = "smoke"
            write_json(output_dir / "pipeline_status.json", status)
            smoke_summary = run_seed_stage(
                output_dir / "smoke" / f"seed_{selected_seeds[0]}",
                config,
                checkpoint,
                checkpoint_id,
                checkpoint_path,
                selected_seeds[0],
                devices,
                int(args.processes),
                True,
                args.resume,
            )
            write_json(
                smoke_marker,
                {"passed": True, "summary": smoke_summary},
            )
        if args.smoke_only:
            status = {
                "success": True,
                "stage": "smoke_complete",
                "runtime_seconds": time.perf_counter() - started,
            }
            write_json(output_dir / "pipeline_status.json", status)
            return 0
        status["stage"] = "full"
        write_json(output_dir / "pipeline_status.json", status)
        notify(topic, "Three-method inverse comparison started", str(output_dir))
        seed_summaries = []
        for seed in selected_seeds:
            seed_summaries.append(
                run_seed_stage(
                    output_dir / "full" / f"seed_{seed}",
                    config,
                    checkpoint,
                    checkpoint_id,
                    checkpoint_path,
                    seed,
                    devices,
                    int(args.processes),
                    False,
                    args.resume,
                )
            )
        across = aggregate_across_seeds(output_dir / "full", selected_seeds, config)
        status = {
            "success": True,
            "stage": "complete",
            "runtime_seconds": time.perf_counter() - started,
            "seed_summaries": seed_summaries,
            "across_seeds": across,
        }
        write_json(output_dir / "pipeline_status.json", status)
        write_json(output_dir / "INVERSE_COMPARISON_COMPLETED.json", status)
        notify(topic, "Three-method inverse comparison succeeded", str(output_dir))
        return 0
    except Exception as error:
        status = {
            "success": False,
            "stage": status.get("stage"),
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "runtime_seconds": time.perf_counter() - started,
        }
        write_json(output_dir / "pipeline_status.json", status)
        LOGGER.exception("Three-method inverse comparison failed")
        notify(topic, "Three-method inverse comparison failed", repr(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
