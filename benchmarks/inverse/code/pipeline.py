#!/usr/bin/env python3
"""Unified panel-based Frozen, Direct LM and Direct DE benchmark."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time
import traceback
from typing import Any, Mapping, Sequence


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = BENCHMARK_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common.aggregation import aggregate  # noqa: E402
from common.io_utils import read_json, write_csv, write_json  # noqa: E402
from common.panel_archive import generate_archive  # noqa: E402
from common.panel_scheduler import (  # noqa: E402
    run_de_panels, run_frozen_panels, run_lm_panels,
)
from systems import build_adapter  # noqa: E402


SYSTEM_CONFIGS = {
    "v2d": "variable_delay.json",
    "n4d": "nicholson.json",
    "sei": "delayed_sei.json",
}


def parse_csv_floats(text: str) -> list[float]:
    values = [float(value.strip()) for value in text.split(",") if value.strip()]
    if not values or any(value < 0.0 for value in values):
        raise argparse.ArgumentTypeError("noise levels must be nonnegative")
    return values


def parse_methods(text: str) -> list[str]:
    values = [value.strip() for value in text.split(",") if value.strip()]
    invalid = sorted(set(values) - {"frozen", "lm", "de"})
    if invalid or not values:
        raise argparse.ArgumentTypeError(
            f"methods must be frozen,lm,de; invalid={invalid}"
        )
    return list(dict.fromkeys(values))


def parse_devices(text: str) -> list[str]:
    values = [value.strip() for value in text.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one device is required")
    if values == ["cpu"]:
        return ["cpu"]
    try:
        return [f"cuda:{int(value)}" for value in values]
    except ValueError as error:
        raise argparse.ArgumentTypeError("--gpus must be cpu or comma-separated integers") from error


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--system", choices=sorted(SYSTEM_CONFIGS), required=True)
    result.add_argument(
        "--stage", choices=("generate", "inverse", "aggregate", "all"), default="all"
    )
    result.add_argument("--common-config", type=Path)
    result.add_argument("--system-config", type=Path)
    result.add_argument("--source-root", type=Path)
    result.add_argument("--checkpoint", type=Path)
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--archive-dir", type=Path)
    result.add_argument("--run-metadata-dir", type=Path)
    result.add_argument("--methods", type=parse_methods, default=["frozen", "lm"])
    result.add_argument("--gpus", type=parse_devices, default=["cuda:0"])
    result.add_argument("--processes", type=int, default=1)
    result.add_argument("--lm-processes", type=int, default=1)
    result.add_argument("--de-processes", type=int, default=1)
    result.add_argument("--generation-processes", type=int, default=1)
    result.add_argument("--cpus", type=int, default=1)
    result.add_argument("--panel-count", type=int)
    result.add_argument("--histories-per-panel", type=int)
    result.add_argument(
        "--histories-per-request",
        type=int,
        help="number of archived panel histories used by each inverse request",
    )
    result.add_argument("--case-count", type=int)
    result.add_argument("--noise-levels", type=parse_csv_floats)
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--validate-cache", action="store_true")
    return result


def _mode(args: argparse.Namespace) -> str:
    return "smoke" if args.smoke else "pilot" if args.pilot else "formal"


def _default_output(system: str, mode: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return BENCHMARK_ROOT / "outputs" / f"{system}_{mode}_{timestamp}"


def _resolve_config_path(value: Path | None, default_name: str) -> Path:
    if value is None:
        return BENCHMARK_ROOT / "configs" / default_name
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (Path.cwd() / value).resolve()


def resolve_protocol(
    args: argparse.Namespace,
    common: Mapping[str, Any],
    system: Mapping[str, Any],
) -> dict[str, Any]:
    mode = _mode(args)
    selected = copy.deepcopy(common[mode])
    panel_count = int(args.panel_count or selected["panel_count"])
    histories = int(args.histories_per_panel or selected["histories_per_panel"])
    histories_per_request = int(args.histories_per_request or histories)
    cases = int(args.case_count or selected["cases_per_panel"])
    noise = list(
        args.noise_levels
        if args.noise_levels is not None
        else selected["noise_standard_deviations"]
    )
    observation_count = int(
        selected.get("observation_count", system["observation_count"])
    )
    truth_step = float(
        selected.get("truth_internal_step", common["data"]["truth_internal_step"])
    )
    output_points = int(common["data"]["output_points"])
    frozen = copy.deepcopy(common["frozen"])
    frozen["grid_points"] = list(
        selected.get("grid_points", system["grid_points"])
    )
    frozen["top_k"] = int(selected.get("top_k", frozen["top_k"]))
    frozen["cache_parameter_validation_cases"] = int(
        selected.get("cache_parameter_validation_cases", 0)
    )
    for source, target in (
        ("adam_min_steps", "adam_min_steps"),
        ("adam_max_steps", "adam_max_steps"),
        ("adam_check_interval", "adam_check_interval"),
        ("adam_patience", "adam_patience"),
    ):
        if source in selected:
            frozen[target] = int(selected[source])
    lm = copy.deepcopy(common["lm"])
    for source, target in (
        ("lm_starts", "starts"),
        ("lm_min_iterations", "min_iterations"),
        ("lm_max_iterations", "max_iterations"),
        ("lm_patience", "patience"),
    ):
        if source in selected:
            lm[target] = int(selected[source])
    de = copy.deepcopy(common["de"])
    for source, target in (
        ("de_population_size", "population_size"),
        ("de_maximum_generations", "maximum_generations"),
    ):
        if source in selected:
            de[target] = int(selected[source])
    de["maximum_candidate_evaluations"] = int(de["population_size"]) * (
        int(de["maximum_generations"]) + 1
    )

    if min(panel_count, histories, histories_per_request, cases, observation_count) < 1:
        raise ValueError("panel, history, case and observation counts must be positive")
    if histories_per_request > histories:
        raise ValueError("histories_per_request exceeds histories_per_panel")
    if observation_count > output_points:
        raise ValueError("observation_count exceeds output_points")
    if system["system_key"] == "sei" and histories > 8:
        raise ValueError("High-75 SEI has exactly eight predefined strata")
    if float(frozen["physics_weight"]) != 0.0 or bool(frozen["use_lbfgsb"]):
        raise ValueError("the requested protocol requires data-only Frozen without L-BFGS-B")
    if int(frozen["top_k"]) > int(np.prod(frozen["grid_points"])):
        raise ValueError("top_k exceeds grid size")
    if int(de["population_size"]) < 4 or int(de["maximum_generations"]) < 0:
        raise ValueError("invalid Direct DE population or generation count")
    if (
        str(de["strategy"]) != "rand1bin"
        or str(de["coordinate_system"]) != "normalized"
        or str(de["initialization"]) != "latin_hypercube"
        or str(de["bound_handling"]) != "reflection"
        or bool(de["polish"])
    ):
        raise ValueError("Direct DE protocol configuration was altered")
    return {
        "mode": mode,
        "panel_count": panel_count,
        "histories_per_panel": histories,
        "histories_per_request": histories_per_request,
        "cases_per_panel": cases,
        "noise_standard_deviations": [float(value) for value in noise],
        "observation_count": observation_count,
        "truth_internal_step": truth_step,
        "output_points": output_points,
        "frozen": frozen,
        "lm": lm,
        "de": de,
    }


def environment_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "requested_devices": list(args.gpus),
        "cpu_count": os.cpu_count(),
        "pid": os.getpid(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    pipeline_started = time.perf_counter()
    args = parser().parse_args(argv)
    common_path = _resolve_config_path(args.common_config, "common.json")
    system_path = _resolve_config_path(
        args.system_config, SYSTEM_CONFIGS[args.system]
    )
    common = read_json(common_path)
    system = read_json(system_path)
    if str(system["system_key"]) != args.system:
        raise RuntimeError("--system differs from system config")
    protocol = resolve_protocol(args, common, system)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else _default_output(args.system, protocol["mode"])
    )
    # The nohup launcher creates these bookkeeping files before this process
    # starts.  They are not experiment results and must not make a fresh
    # output directory look occupied.
    launcher_files = {"nohup.log", "pipeline.pid"}
    existing_entries = (
        [path for path in output_dir.iterdir() if path.name not in launcher_files]
        if output_dir.exists()
        else []
    )
    if existing_entries and not args.resume:
        raise FileExistsError(
            f"output directory is not empty; use --resume or a new path: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_metadata_dir = (
        args.run_metadata_dir.expanduser().resolve()
        if args.run_metadata_dir is not None
        else output_dir
    )
    run_metadata_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = (
        args.archive_dir.expanduser().resolve()
        if args.archive_dir is not None
        else output_dir / "shared_panels"
    )
    full_dir = output_dir / "full"
    full_dir.mkdir(parents=True, exist_ok=True)
    source_override = args.source_root.expanduser().resolve() if args.source_root else None
    adapter = build_adapter(args.system, system, BENCHMARK_ROOT, source_override)

    run_config = {
        "protocol_version": common["protocol_version"],
        "system": system,
        "protocol": protocol,
        "methods": list(args.methods),
        "common_config_path": str(common_path),
        "system_config_path": str(system_path),
        "source_project": str(adapter.source_project),
        "output_dir": str(output_dir),
        "archive_dir": str(archive_dir),
    }
    write_json(run_metadata_dir / "experiment_config.json", run_config)
    write_json(run_metadata_dir / "environment.json", environment_snapshot(args))
    write_json(
        run_metadata_dir / "pipeline_status.json",
        {"status": "running", "stage": args.stage, "started_at": time.time()},
    )
    runtime: dict[str, Any] = {}
    print(
        f"System={args.system} mode={protocol['mode']} stage={args.stage} "
        f"panels={protocol['panel_count']} histories/panel={protocol['histories_per_panel']} "
        f"histories/request={protocol['histories_per_request']} "
        f"cases/panel={protocol['cases_per_panel']} observations/history={protocol['observation_count']}",
        flush=True,
    )
    print(f"Output={output_dir}", flush=True)
    print(f"Run metadata={run_metadata_dir}", flush=True)

    try:
        if args.stage in {"generate", "all"}:
            print("Starting immutable panel-data generation", flush=True)
            stage_started = time.perf_counter()
            generate_archive(
                adapter,
                system,
                common,
                BENCHMARK_ROOT,
                source_override,
                archive_dir,
                protocol["panel_count"],
                protocol["histories_per_panel"],
                protocol["cases_per_panel"],
                protocol["noise_standard_deviations"],
                protocol["observation_count"],
                protocol["truth_internal_step"],
                protocol["output_points"],
                max(1, int(args.generation_processes)),
                args.resume,
            )
            runtime["data_generation_wall_seconds"] = float(
                time.perf_counter() - stage_started
            )
            print(
                f"Panel-data generation completed in "
                f"{runtime['data_generation_wall_seconds']:.3f}s",
                flush=True,
            )
        if args.stage in {"inverse", "all"}:
            manifest = read_json(archive_dir / "manifest.json")
            for key in (
                "panel_count", "histories_per_panel", "cases_per_panel",
                "observation_count", "output_points"
            ):
                protocol_key = key
                if int(manifest[key]) != int(protocol[protocol_key]):
                    raise RuntimeError(f"archive {key} differs from requested protocol")
            if [float(v) for v in manifest["noise_standard_deviations"]] != [
                float(v) for v in protocol["noise_standard_deviations"]
            ]:
                raise RuntimeError("archive noise levels differ from requested protocol")

            if "frozen" in args.methods:
                print("Starting Frozen panel inversion", flush=True)
                if args.gpus != ["cpu"] and not torch.cuda.is_available():
                    raise RuntimeError("CUDA devices requested but torch.cuda.is_available() is false")
                checkpoint = adapter.resolve_checkpoint(args.checkpoint)
                devices = list(args.gpus)[: max(1, int(args.processes))]
                stage_started = time.perf_counter()
                frozen_rows = run_frozen_panels(
                    devices,
                    protocol["panel_count"],
                    args.system,
                    system,
                    common,
                    BENCHMARK_ROOT,
                    source_override,
                    checkpoint,
                    archive_dir,
                    full_dir,
                    protocol["frozen"],
                    protocol["histories_per_request"],
                    args.resume,
                    bool(
                        args.validate_cache
                        or protocol["mode"] in {"pilot", "smoke"}
                    ),
                )
                runtime["frozen_multi_gpu_wall_seconds"] = float(
                    time.perf_counter() - stage_started
                )
                write_csv(full_dir / "frozen_per_case_results.csv", frozen_rows)
                print(
                    f"Frozen completed: {len(frozen_rows)} requests in "
                    f"{runtime['frozen_multi_gpu_wall_seconds']:.3f}s",
                    flush=True,
                )

            if "lm" in args.methods:
                print("Starting Direct LM panel inversion", flush=True)
                lm_workers = max(1, min(int(args.lm_processes), int(args.cpus)))
                threads_per_worker = max(1, int(args.cpus) // lm_workers)
                os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
                os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)
                stage_started = time.perf_counter()
                lm_rows = run_lm_panels(
                    lm_workers,
                    protocol["panel_count"],
                    args.system,
                    system,
                    common,
                    BENCHMARK_ROOT,
                    source_override,
                    archive_dir,
                    full_dir,
                    protocol["lm"],
                    protocol["histories_per_request"],
                    args.resume,
                )
                runtime["lm_cpu_wall_seconds"] = float(
                    time.perf_counter() - stage_started
                )
                write_csv(full_dir / "lm_per_case_results.csv", lm_rows)
                print(
                    f"Direct LM completed: {len(lm_rows)} requests in "
                    f"{runtime['lm_cpu_wall_seconds']:.3f}s",
                    flush=True,
                )

            if "de" in args.methods:
                print("Starting Direct DE panel inversion", flush=True)
                de_workers = max(1, min(int(args.de_processes), int(args.cpus)))
                threads_per_worker = max(1, int(args.cpus) // de_workers)
                os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
                os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)
                stage_started = time.perf_counter()
                de_rows = run_de_panels(
                    de_workers,
                    protocol["panel_count"],
                    args.system,
                    system,
                    common,
                    BENCHMARK_ROOT,
                    source_override,
                    archive_dir,
                    full_dir,
                    protocol["de"],
                    protocol["histories_per_request"],
                    args.resume,
                )
                runtime["de_cpu_wall_seconds"] = float(
                    time.perf_counter() - stage_started
                )
                write_csv(full_dir / "de_per_case_results.csv", de_rows)
                print(
                    f"Direct DE completed: {len(de_rows)} requests in "
                    f"{runtime['de_cpu_wall_seconds']:.3f}s",
                    flush=True,
                )

        summary = None
        if args.stage in {"inverse", "aggregate", "all"}:
            print("Aggregating panel-level results", flush=True)
            summary = aggregate(full_dir, common["reporting"])
        runtime["total_pipeline_wall_seconds"] = float(
            time.perf_counter() - pipeline_started
        )
        write_json(run_metadata_dir / "runtime.json", runtime)
        print(
            f"Completed in {runtime['total_pipeline_wall_seconds']:.3f}s", flush=True
        )
        write_json(
            run_metadata_dir / "pipeline_status.json",
            {
                "status": "completed",
                "stage": args.stage,
                "completed_at": time.time(),
                "summary": summary,
            },
        )
        return 0
    except Exception as error:
        write_json(
            run_metadata_dir / "pipeline_status.json",
            {
                "status": "failed",
                "stage": args.stage,
                "failed_at": time.time(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
