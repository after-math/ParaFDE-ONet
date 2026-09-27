#!/usr/bin/env python3
"""Measure history-feature reuse during repeated Nicholson parameter queries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import socket
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import scienceplots
import matplotlib
plt.style.use(['science','no-latex','grid',])
matplotlib.rcParams['font.family'] = 'Source Han Serif'
matplotlib.rcParams['font.family'] = 'Source Han Serif'
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["pdf.fonttype"] = 42

import numpy as np
import torch


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = BENCHMARK_DIR / "configs" / "benchmark.json"
METHOD_ORDER = (
    "single_branch_deeponet",
    "four_branch_mionet",
    "separate_parameter_shared",
)
METHOD_COLORS = {
    "single_branch_deeponet": "#4D4D4D",
    "four_branch_mionet": "#6276B5",
    "separate_parameter_shared": "#C44E52",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark repeated loss-and-parameter-gradient evaluations."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--source-project",
        type=Path,
        default=None,
        help="Path to NicholsonPatch4D_4Methods_1Seed_2026-08-18.",
    )
    parser.add_argument("--ji-checkpoint", type=Path, default=None)
    parser.add_argument("--rp-checkpoint", type=Path, default=None)
    parser.add_argument("--pf-checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return values


def save_json(path: Path, values: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(values), indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def save_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot save an empty CSV")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("CSV rows have inconsistent fields")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def resolve_source_project(config: Mapping[str, Any], override: Path | None) -> Path:
    if override is not None:
        source = override.expanduser().resolve()
    else:
        source = BENCHMARK_DIR.parent / str(config["source_project_name"])
    required = source / str(config["source_experiment"]) / "code" / "model.py"
    if not required.is_file():
        raise FileNotFoundError(
            f"source project was not found at {source}; pass --source-project"
        )
    return source


def install_source_imports(source_project: Path, source_experiment: str) -> None:
    source_root = source_project / source_experiment
    source_code = source_root / "code"
    for value in (str(source_code), str(source_root)):
        while value in sys.path:
            sys.path.remove(value)
    sys.path.insert(0, str(source_code))
    sys.path.insert(1, str(source_root))


def default_checkpoint_paths(
    source_project: Path, config: Mapping[str, Any]
) -> dict[str, Path]:
    root = (
        source_project
        / str(config["source_experiment"])
        / "outputs"
        / str(config["source_run"])
        / "full"
        / "methods"
    )
    if "checkpoint_root" in config:
        configured = Path(str(config["checkpoint_root"])).expanduser()
        root = configured if configured.is_absolute() else BENCHMARK_DIR.parents[1] / configured
    seed_dir = f"seed_{int(config['training_seed'])}"
    return {
        method: root / method / seed_dir / "best_model.pt" for method in METHOD_ORDER
    }


def resolve_checkpoints(
    source_project: Path, config: Mapping[str, Any], args: argparse.Namespace
) -> dict[str, Path]:
    paths = default_checkpoint_paths(source_project, config)
    overrides = {
        "single_branch_deeponet": args.ji_checkpoint,
        "four_branch_mionet": args.rp_checkpoint,
        "separate_parameter_shared": args.pf_checkpoint,
    }
    for method, override in overrides.items():
        if override is not None:
            paths[method] = override.expanduser().resolve()
    missing = [f"{method}: {path}" for method, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "required formal checkpoints are missing:\n  " + "\n  ".join(missing)
        )
    return paths


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def validate_config(config: Mapping[str, Any], smoke: bool) -> None:
    if tuple(config["methods"]) != METHOD_ORDER:
        raise ValueError(f"methods must be exactly {METHOD_ORDER}")
    panel = config["panel"]
    if int(panel["history_count"]) < 1 or int(panel["observation_count"]) < 2:
        raise ValueError("panel sizes must be positive")
    states = [int(value) for value in panel["observed_states"]]
    if not states or len(states) != len(set(states)) or any(v not in range(4) for v in states):
        raise ValueError("observed_states must be a unique nonempty subset of 0,1,2,3")
    counts = [int(value) for value in config["queries"]["counts"]]
    if counts != [1, 10, 100, 1000, 3000, 10000, 100000]:
        raise ValueError(
            "formal query counts must be [1,10,100,1000,3000,10000,100000]"
        )
    if int(config["timing"]["repeats"]) < 2:
        raise ValueError("at least two timing repeats are required for sample SD")
    if str(config["timing"]["dtype"]) != "float32":
        raise ValueError("this benchmark requires float32")
    formats = {str(value).lower() for value in config["reporting"]["figure_formats"]}
    if not formats or not formats <= {"pdf", "svg", "png"}:
        raise ValueError("figure formats must be pdf, svg, or png")
    if smoke and int(config["smoke"]["repeats"]) < 2:
        raise ValueError("smoke repeats must be at least two")


def load_models(
    checkpoint_paths: Mapping[str, Path], device: torch.device, config: Mapping[str, Any]
) -> tuple[dict[str, torch.nn.Module], dict[str, dict[str, Any]]]:
    from model import load_operator_checkpoint

    models: dict[str, torch.nn.Module] = {}
    payloads: dict[str, dict[str, Any]] = {}
    expected_seed = int(config["training_seed"])
    for method in METHOD_ORDER:
        model, payload = load_operator_checkpoint(checkpoint_paths[method], device)
        if str(model.model_type) != method:
            raise RuntimeError(f"checkpoint for {method} contains {model.model_type}")
        if config["timing"].get("require_same_training_seed", True):
            if int(payload.get("training_seed", -1)) != expected_seed:
                raise RuntimeError(f"{method} checkpoint does not use seed {expected_seed}")
        model.eval()
        model.requires_grad_(False)
        models[method] = model
        payloads[method] = payload

    reference = payloads["separate_parameter_shared"]["resolved_config"]
    for method in METHOD_ORDER[:-1]:
        candidate = payloads[method]["resolved_config"]
        for section in ("data", "equation"):
            if candidate[section] != reference[section]:
                raise RuntimeError(f"{method} and PFDEONet use different {section} settings")
    return models, payloads


def prepare_panel(
    scientific_config: Mapping[str, Any], benchmark_config: Mapping[str, Any], smoke: bool
) -> dict[str, np.ndarray]:
    from data import sample_histories, sample_parameters
    from equation import NicholsonConfig, solve_batch

    panel_config = benchmark_config["panel"]
    smoke_config = benchmark_config["smoke"]
    history_count = int(
        smoke_config["history_count"] if smoke else panel_config["history_count"]
    )
    observation_count = int(
        smoke_config["observation_count"] if smoke else panel_config["observation_count"]
    )
    data = scientific_config["data"]
    equation = NicholsonConfig.from_mapping(scientific_config["equation"])
    history_times = np.linspace(
        -equation.maximum_history,
        0.0,
        int(data["history_sensors"]),
        dtype=np.float64,
    )
    observation_times = np.linspace(
        0.0, float(data["horizon"]), observation_count, dtype=np.float64
    )
    history_rng = np.random.default_rng(int(panel_config["history_seed"]))
    histories = sample_histories(
        history_count,
        history_times,
        tuple(float(value) for value in data["history_mean"]),
        float(data["history_sigma"]),
        float(data["history_length_scale"]),
        tuple(float(value) for value in data["history_bounds"]),
        history_rng,
    )
    parameter_rng = np.random.default_rng(int(panel_config["true_parameter_seed"]))
    true_parameter = sample_parameters(1, equation, parameter_rng)[0]
    repeated_parameters = np.repeat(true_parameter[None, :], history_count, axis=0)
    clean = solve_batch(
        histories,
        repeated_parameters,
        history_times,
        observation_times,
        equation,
        float(data["internal_step"]),
    )
    states = np.asarray(panel_config["observed_states"], dtype=np.int64)
    noise_rng = np.random.default_rng(int(panel_config["noise_seed"]))
    observations = clean[:, :, states] + noise_rng.normal(
        loc=0.0,
        scale=float(panel_config["noise_standard_deviation"]),
        size=(history_count, observation_count, states.size),
    ).astype(np.float32)
    return {
        "histories": histories.astype(np.float32),
        "history_times": history_times.astype(np.float32),
        "observation_times": observation_times.astype(np.float32),
        "observed_states": states,
        "true_parameter": true_parameter.astype(np.float32),
        "clean_observations": clean[:, :, states].astype(np.float32),
        "noisy_observations": observations.astype(np.float32),
    }


def prepare_parameter_sequence(
    scientific_config: Mapping[str, Any], benchmark_config: Mapping[str, Any], smoke: bool
) -> np.ndarray:
    from data import sample_parameters
    from equation import NicholsonConfig

    counts = (
        benchmark_config["smoke"]["query_counts"]
        if smoke
        else benchmark_config["queries"]["counts"]
    )
    maximum = max(int(value) for value in counts)
    equation = NicholsonConfig.from_mapping(scientific_config["equation"])
    rng = np.random.default_rng(int(benchmark_config["queries"]["parameter_seed"]))
    return sample_parameters(maximum, equation, rng).astype(np.float32)


def build_cache(
    model: torch.nn.Module, histories: torch.Tensor, times: torch.Tensor
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        trunk = model.time_trunk(model.time_features(times)).detach()
        cache = {"trunk": trunk}
        if model.model_type == "separate_parameter_shared":
            encoded = [
                1.0 + branch(histories[:, component, :]).reshape(
                    histories.shape[0], 4, model.latent_dim
                )
                for component, branch in enumerate(model.history_branches)
            ]
            history_features = encoded[0]
            for values in encoded[1:]:
                history_features = history_features * values
            cache["history"] = history_features.detach()
    return cache


def cached_prediction(
    model: torch.nn.Module,
    histories: torch.Tensor,
    physical_parameter: torch.Tensor,
    cache: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    panel_size = histories.shape[0]
    if model.model_type == "single_branch_deeponet":
        normalized = model.normalize_parameters(
            physical_parameter.unsqueeze(0).expand(panel_size, -1)
        )
        joint = torch.cat((histories.reshape(panel_size, -1), normalized), dim=-1)
        state_features = model.joint_branch(joint).reshape(
            panel_size, 4, model.latent_dim
        )
    elif model.model_type == "four_branch_mionet":
        normalized = model.normalize_parameters(
            physical_parameter.unsqueeze(0).expand(panel_size, -1)
        )
        encoded = [
            1.0
            + branch(torch.cat((histories[:, component, :], normalized), dim=-1)).reshape(
                panel_size, 4, model.latent_dim
            )
            for component, branch in enumerate(model.history_parameter_branches)
        ]
        state_features = encoded[0]
        for values in encoded[1:]:
            state_features = state_features * values
    elif model.model_type == "separate_parameter_shared":
        normalized = model.normalize_parameters(physical_parameter.unsqueeze(0))
        parameter_features = model.parameter_branch(normalized)
        state_features = cache["history"] * (1.0 + parameter_features.unsqueeze(1))
    else:
        raise RuntimeError(f"unsupported benchmark model: {model.model_type}")
    raw = torch.einsum("bsp,qp->bqs", state_features, cache["trunk"])
    return raw * model.latent_scale + model.output_bias


def loss_and_parameter_gradient(
    model: torch.nn.Module,
    histories: torch.Tensor,
    observations: torch.Tensor,
    observed_states: torch.Tensor,
    cache: Mapping[str, torch.Tensor],
    candidate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    parameter = candidate.detach().clone().requires_grad_(True)
    prediction = cached_prediction(model, histories, parameter, cache)
    selected = prediction.index_select(-1, observed_states)
    loss = torch.mean((selected - observations) ** 2)
    gradient = torch.autograd.grad(loss, parameter, create_graph=False)[0]
    return prediction.detach(), loss.detach(), gradient.detach()


def full_loss_and_gradient(
    model: torch.nn.Module,
    histories: torch.Tensor,
    times: torch.Tensor,
    observations: torch.Tensor,
    observed_states: torch.Tensor,
    candidate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    parameter = candidate.detach().clone().requires_grad_(True)
    repeated = parameter.unsqueeze(0).expand(histories.shape[0], -1)
    prediction = model(histories, repeated, times)
    selected = prediction.index_select(-1, observed_states)
    loss = torch.mean((selected - observations) ** 2)
    gradient = torch.autograd.grad(loss, parameter, create_graph=False)[0]
    return prediction.detach(), gradient.detach()


def relative_error(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm((left - right).double())
    denominator = torch.linalg.vector_norm(right.double()).clamp_min(1.0e-30)
    return float((numerator / denominator).cpu())


def correctness_checks(
    models: Mapping[str, torch.nn.Module],
    histories: torch.Tensor,
    times: torch.Tensor,
    observations: torch.Tensor,
    observed_states: torch.Tensor,
    candidate: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        cache = build_cache(models[method], histories, times)
        cached_output, _, cached_gradient = loss_and_parameter_gradient(
            models[method], histories, observations, observed_states, cache, candidate
        )
        full_output, full_gradient = full_loss_and_gradient(
            models[method], histories, times, observations, observed_states, candidate
        )
        prediction_error = relative_error(cached_output, full_output)
        gradient_error = relative_error(cached_gradient, full_gradient)
        if prediction_error > 2.0e-5 or gradient_error > 5.0e-4:
            raise RuntimeError(
                f"cached/full mismatch for {method}: {prediction_error=}, {gradient_error=}"
            )
        rows.append(
            {
                "model_type": method,
                "prediction_relative_error": prediction_error,
                "parameter_gradient_relative_error": gradient_error,
            }
        )
    return rows


def warm_up(
    model: torch.nn.Module,
    histories: torch.Tensor,
    times: torch.Tensor,
    observations: torch.Tensor,
    observed_states: torch.Tensor,
    candidates: torch.Tensor,
    count: int,
    device: torch.device,
) -> None:
    cache = build_cache(model, histories, times)
    for index in range(count):
        loss_and_parameter_gradient(
            model,
            histories,
            observations,
            observed_states,
            cache,
            candidates[index % candidates.shape[0]],
        )
    synchronize(device)


def time_cache_and_first_query(
    model: torch.nn.Module,
    histories: torch.Tensor,
    times: torch.Tensor,
    observations: torch.Tensor,
    observed_states: torch.Tensor,
    candidate: torch.Tensor,
    device: torch.device,
) -> tuple[float, float, float]:
    synchronize(device)
    cache_started = time.perf_counter()
    cache = build_cache(model, histories, times)
    synchronize(device)
    cache_seconds = time.perf_counter() - cache_started
    query_started = time.perf_counter()
    _, loss, gradient = loss_and_parameter_gradient(
        model, histories, observations, observed_states, cache, candidate
    )
    synchronize(device)
    query_seconds = time.perf_counter() - query_started
    checksum = float(loss.double().cpu()) + float(gradient.double().sum().cpu())
    if not math.isfinite(checksum):
        raise RuntimeError("non-finite cold-query checksum")
    return cache_seconds, cache_seconds + query_seconds, checksum


def time_steady_queries(
    model: torch.nn.Module,
    histories: torch.Tensor,
    times: torch.Tensor,
    observations: torch.Tensor,
    observed_states: torch.Tensor,
    candidates: torch.Tensor,
    count: int,
    device: torch.device,
) -> tuple[float, float, int]:
    cache = build_cache(model, histories, times)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    checksum = torch.zeros((), dtype=torch.float64, device=device)
    for index in range(count):
        _, loss, gradient = loss_and_parameter_gradient(
            model,
            histories,
            observations,
            observed_states,
            cache,
            candidates[index],
        )
        checksum = checksum + loss.double() + gradient.double().sum()
    synchronize(device)
    elapsed = time.perf_counter() - started
    checksum_value = float(checksum.cpu())
    peak_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    if elapsed <= 0.0 or not math.isfinite(checksum_value):
        raise RuntimeError("invalid steady timing or checksum")
    return elapsed, checksum_value, peak_bytes


def summarize_timings(
    raw_rows: Sequence[Mapping[str, Any]], display_names: Mapping[str, str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in raw_rows:
        grouped.setdefault((str(row["model_type"]), int(row["query_count"])), []).append(row)
    means: dict[tuple[str, int], float] = {}
    total_means: dict[tuple[str, int], float] = {}
    for key, values in grouped.items():
        means[key] = statistics.fmean(float(value["steady_seconds"]) for value in values)
        total_means[key] = statistics.fmean(
            float(value["cache_seconds"]) + float(value["steady_seconds"])
            for value in values
        )
    for method in METHOD_ORDER:
        for query_count in sorted(key[1] for key in grouped if key[0] == method):
            values = grouped[(method, query_count)]
            cache = [float(value["cache_seconds"]) for value in values]
            first = [float(value["first_query_seconds"]) for value in values]
            steady = [float(value["steady_seconds"]) for value in values]
            pf_mean = means[("separate_parameter_shared", query_count)]
            pf_total_mean = total_means[("separate_parameter_shared", query_count)]
            total = [
                float(value["cache_seconds"]) + float(value["steady_seconds"])
                for value in values
            ]
            rows.append(
                {
                    "model_type": method,
                    "method": display_names[method],
                    "query_count": query_count,
                    "timing_repeats": len(values),
                    "cache_mean_seconds": statistics.fmean(cache),
                    "cache_sample_sd_seconds": statistics.stdev(cache),
                    "first_query_mean_seconds": statistics.fmean(first),
                    "first_query_sample_sd_seconds": statistics.stdev(first),
                    "steady_mean_seconds": statistics.fmean(steady),
                    "steady_sample_sd_seconds": statistics.stdev(steady),
                    "mean_seconds_per_query": statistics.fmean(steady) / query_count,
                    "speedup_over_pfdeonet": statistics.fmean(steady) / pf_mean,
                    "cache_plus_steady_mean_seconds": statistics.fmean(total),
                    "cache_plus_steady_sample_sd_seconds": statistics.stdev(total),
                    "cache_plus_steady_time_ratio_to_pfdeonet": (
                        statistics.fmean(total) / pf_total_mean
                    ),
                }
            )
    return rows


def plot_results(
    summary_rows: Sequence[Mapping[str, Any]],
    display_names: Mapping[str, str],
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    query_counts = sorted({int(row["query_count"]) for row in summary_rows})
    labelled_ticks = [value for value in query_counts if value in {1, 10, 100, 1000, 10000, 100000}]
    tick_labels = {
        1: "1",
        10: "10",
        100: "100",
        1000: "1k",
        10000: "10k",
        100000: "100k",
    }
    for method in METHOD_ORDER:
        rows = [row for row in summary_rows if row["model_type"] == method]
        counts = np.asarray([row["query_count"] for row in rows], dtype=float)
        means = np.asarray([row["steady_mean_seconds"] for row in rows], dtype=float)
        stds = np.asarray([row["steady_sample_sd_seconds"] for row in rows], dtype=float)
        axes[0].errorbar(
            counts,
            means,
            yerr=stds,
            marker="o",
            capsize=2.5,
            linewidth=1.2,
            color=METHOD_COLORS[method],
            label=display_names[method],
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xticks(labelled_ticks)
    axes[0].set_xticklabels([tick_labels[value] for value in labelled_ticks])
    axes[0].set_xlabel("Number of successive parameter queries")
    axes[0].set_ylabel("Steady-state time (s)")
    axes[0].legend(frameon=True, fontsize=7)
    axes[0].text(-0.13, 1.04, "a", transform=axes[0].transAxes, fontweight="bold")

    for method in METHOD_ORDER[:2]:
        rows = [row for row in summary_rows if row["model_type"] == method]
        axes[1].plot(
            [row["query_count"] for row in rows],
            [row["speedup_over_pfdeonet"] for row in rows],
            marker="o",
            linewidth=1.2,
            color=METHOD_COLORS[method],
            label=f"{display_names[method]} / PFDEONet",
        )
    axes[1].axhline(1.0, color="0.45", linestyle="--", linewidth=0.9)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xticks(labelled_ticks)
    axes[1].set_xticklabels([tick_labels[value] for value in labelled_ticks])
    axes[1].set_xlabel("Number of successive parameter queries")
    axes[1].set_ylabel("Time ratio")
    axes[1].legend(frameon=True, fontsize=7)
    axes[1].text(-0.13, 1.04, "b", transform=axes[1].transAxes, fontweight="bold")
    figure.tight_layout()
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        figure.savefig(
            figure_dir / f"feature_reuse_timing.{extension}",
            dpi=dpi if extension.lower() == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)


def environment_summary(device: torch.device) -> dict[str, Any]:
    values: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "pid": os.getpid(),
    }
    if device.type == "cuda":
        values["cuda"] = torch.version.cuda
        values["gpu"] = torch.cuda.get_device_name(device)
    return values


def run(args: argparse.Namespace) -> Path:
    config = load_json(args.config.expanduser().resolve())
    validate_config(config, args.smoke)
    source_project = resolve_source_project(config, args.source_project)
    install_source_imports(source_project, str(config["source_experiment"]))
    device = torch.device(args.device)
    if config["timing"].get("require_cuda", True) and not args.smoke:
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("formal timing requires an available CUDA device")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    checkpoint_paths = resolve_checkpoints(source_project, config, args)
    models, payloads = load_models(checkpoint_paths, device, config)
    scientific_config = payloads["separate_parameter_shared"]["resolved_config"]
    panel = prepare_panel(scientific_config, config, args.smoke)
    parameter_sequence = prepare_parameter_sequence(scientific_config, config, args.smoke)
    query_counts = [
        int(value)
        for value in (
            config["smoke"]["query_counts"]
            if args.smoke
            else config["queries"]["counts"]
        )
    ]
    repeats = int(config["smoke"]["repeats"] if args.smoke else config["timing"]["repeats"])
    warmup_queries = int(
        config["smoke"]["warmup_queries"]
        if args.smoke
        else config["timing"]["warmup_queries"]
    )
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output.expanduser().resolve()
        if args.output is not None
        else BENCHMARK_DIR / "outputs" / f"feature_reuse_{timestamp}"
    )
    stage_dir = output_dir / ("smoke" if args.smoke else "full")
    stage_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config.json", config)
    save_json(output_dir / "environment.json", environment_summary(device))
    save_json(
        output_dir / "checkpoint_paths.json",
        {method: str(path) for method, path in checkpoint_paths.items()},
    )
    np.savez_compressed(stage_dir / "fixed_panel.npz", **panel)
    np.save(stage_dir / "candidate_parameters.npy", parameter_sequence)

    histories = torch.from_numpy(panel["histories"]).to(device)
    times = torch.from_numpy(panel["observation_times"]).to(device)
    observations = torch.from_numpy(panel["noisy_observations"]).to(device)
    observed_states = torch.from_numpy(panel["observed_states"]).to(device)
    candidates = torch.from_numpy(parameter_sequence).to(device)
    checks = correctness_checks(
        models,
        histories,
        times,
        observations,
        observed_states,
        candidates[0],
    )
    save_csv(stage_dir / "correctness_checks.csv", checks)

    raw_rows: list[dict[str, Any]] = []
    display_names = {str(key): str(value) for key, value in config["methods"].items()}
    for method in METHOD_ORDER:
        print(f"Warming up {display_names[method]}...", flush=True)
        warm_up(
            models[method],
            histories,
            times,
            observations,
            observed_states,
            candidates,
            warmup_queries,
            device,
        )

    order_rng = np.random.default_rng(int(config["timing"]["measurement_order_seed"]))
    measurement_index = 0
    for repeat in range(1, repeats + 1):
        schedule = [
            (method, query_count)
            for method in METHOD_ORDER
            for query_count in query_counts
        ]
        order_rng.shuffle(schedule)
        for method, query_count in schedule:
            model = models[method]
            measurement_index += 1
            cache_seconds, first_seconds, cold_checksum = time_cache_and_first_query(
                model,
                histories,
                times,
                observations,
                observed_states,
                candidates[0],
                device,
            )
            steady_seconds, steady_checksum, peak_bytes = time_steady_queries(
                model,
                histories,
                times,
                observations,
                observed_states,
                candidates,
                query_count,
                device,
            )
            raw_rows.append(
                {
                    "model_type": method,
                    "method": display_names[method],
                    "query_count": query_count,
                    "repeat": repeat,
                    "measurement_index": measurement_index,
                    "cache_seconds": cache_seconds,
                    "first_query_seconds": first_seconds,
                    "steady_seconds": steady_seconds,
                    "seconds_per_query": steady_seconds / query_count,
                    "peak_memory_bytes": peak_bytes,
                    "cold_checksum": cold_checksum,
                    "steady_checksum": steady_checksum,
                }
            )
            print(
                f"Completed {display_names[method]} with K={query_count} "
                f"(timing repeat {repeat}/{repeats}).",
                flush=True,
            )
    save_csv(stage_dir / "raw_timings.csv", raw_rows)
    summary_rows = summarize_timings(raw_rows, display_names)
    save_csv(stage_dir / "timing_summary.csv", summary_rows)
    plot_results(
        summary_rows,
        display_names,
        stage_dir,
        [str(value).lower() for value in config["reporting"]["figure_formats"]],
        int(config["reporting"]["raster_dpi"]),
    )
    save_json(
        stage_dir / "summary.json",
        {
            "success": True,
            "stage": "smoke" if args.smoke else "full",
            "history_count": int(histories.shape[0]),
            "observation_count": int(times.numel()),
            "observed_states_zero_based": panel["observed_states"].tolist(),
            "query_counts": query_counts,
            "timing_repeats": repeats,
            "warmup_queries": warmup_queries,
            "measurement_order": "interleaved and shuffled within each timing repeat",
            "measurement_order_seed": int(config["timing"]["measurement_order_seed"]),
            "training_seed": int(config["training_seed"]),
            "timed_work": "prediction + observation loss + parameter gradient",
            "excluded_work": [
                "model loading",
                "panel generation",
                "candidate-parameter generation",
                "optimizer update",
                "line search",
                "file output",
            ],
            "cache_policy": {
                "all_methods": "time-trunk features",
                "PFDEONet_only": "parameter-independent history-branch features",
            },
        },
    )
    save_json(output_dir / "pipeline_status.json", {"success": True, "stage_dir": str(stage_dir)})
    return output_dir


def main() -> None:
    output = run(parse_args())
    print(f"Benchmark completed: {output}")


if __name__ == "__main__":
    main()
