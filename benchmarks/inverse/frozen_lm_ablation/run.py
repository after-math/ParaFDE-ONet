#!/usr/bin/env python3
"""Run the Frozen-LM optimizer ablation on existing immutable panel archives."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import gc
import multiprocessing as mp
import os
from pathlib import Path
import platform
import socket
import sys
import time
import traceback
from typing import Any, Mapping, Sequence


ABLATION_ROOT = Path(__file__).resolve().parent
BENCHMARK_ROOT = ABLATION_ROOT.parent
CODE_ROOT = BENCHMARK_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
if str(ABLATION_ROOT) not in sys.path:
    sys.path.insert(0, str(ABLATION_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common.aggregation import noise_summary_rows, panel_macro_rows  # noqa: E402
from common.frozen_panel_runner import _device_name  # noqa: E402
from common.io_utils import (  # noqa: E402
    read_json,
    sha256_file,
    sha256_mapping,
    write_csv,
    write_json,
)
from common.metrics import evaluate_estimate  # noqa: E402
from common.panel_archive import load_panel, select_panel_histories  # noqa: E402
from common.panel_scheduler import panel_queues  # noqa: E402
from systems import build_adapter  # noqa: E402
from runtime import FrozenLMRuntime  # noqa: E402


SYSTEM_CONFIGS = {
    "v2d": "variable_delay.json",
    "n4d": "nicholson.json",
    "d3d": "delayed_sei.json",
}
SYSTEM_ADAPTER_KEYS = {"v2d": "v2d", "n4d": "n4d", "d3d": "sei"}


def resolve_histories_per_request(
    requested_system: str,
    requested_count: int | None,
    archive_history_count: int,
) -> int:
    """Resolve the protocol history count without changing the original baseline."""
    if requested_system == "d3d":
        if archive_history_count != 8:
            raise ValueError(
                "D3D Frozen-LM requires the original eight-history panel archive"
            )
        history_count = 8 if requested_count is None else int(requested_count)
        if history_count != 8:
            raise ValueError("D3D Frozen-LM must use all eight history functions")
        return history_count
    return 1 if requested_count is None else int(requested_count)


def parse_devices(text: str) -> list[str]:
    values = [value.strip() for value in text.split(",") if value.strip()]
    if values == ["cpu"]:
        return ["cpu"]
    if not values:
        raise argparse.ArgumentTypeError("at least one device is required")
    try:
        return [f"cuda:{int(value)}" for value in values]
    except ValueError as error:
        raise argparse.ArgumentTypeError("--gpus must be cpu or comma-separated integers") from error


def parse_noise(text: str) -> list[float]:
    values = [float(value.strip()) for value in text.split(",") if value.strip()]
    if not values or any(value < 0.0 for value in values):
        raise argparse.ArgumentTypeError("noise levels must be nonnegative")
    return values


def parse_grid(text: str) -> list[int]:
    values = [int(value.strip()) for value in text.split(",") if value.strip()]
    if len(values) != 2 or min(values) < 2:
        raise argparse.ArgumentTypeError("--grid-points requires two integers >= 2")
    return values


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--system", choices=sorted(SYSTEM_CONFIGS), required=True)
    result.add_argument("--archive-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path)
    result.add_argument("--source-root", type=Path)
    result.add_argument("--config", type=Path, default=ABLATION_ROOT / "config.json")
    result.add_argument("--gpus", type=parse_devices, default=["cuda:0"])
    result.add_argument("--processes", type=int, default=1)
    result.add_argument(
        "--histories-per-request",
        type=int,
        help="defaults to 8 for D3D and 1 for V2D/N4D",
    )
    result.add_argument("--panel-count", type=int)
    result.add_argument("--case-count", type=int)
    result.add_argument("--noise-levels", type=parse_noise)
    result.add_argument("--grid-points", type=parse_grid)
    result.add_argument("--top-k", type=int)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--validate-cache", action="store_true")
    return result


def _result_path(panel_dir: Path, case_index: int, sigma: float) -> Path:
    noise_tag = str(float(sigma)).replace(".", "p")
    return (
        panel_dir
        / "jobs"
        / "frozen_lm"
        / f"case_{case_index:03d}"
        / f"noise_{noise_tag}"
        / "result.json"
    )


def run_worker(
    worker_index: int,
    device_name: str,
    panel_indices: Sequence[int],
    case_count: int,
    noise_levels: Sequence[float],
    histories_per_request: int,
    system_key: str,
    system_config: Mapping[str, Any],
    common_config: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    source_root_text: str | None,
    checkpoint_text: str,
    archive_dir_text: str,
    full_dir_text: str,
    resume: bool,
    validate_cache: bool,
) -> list[dict[str, Any]]:
    adapter = build_adapter(
        system_key,
        system_config,
        BENCHMARK_ROOT,
        Path(source_root_text) if source_root_text else None,
    )
    archive_dir = Path(archive_dir_text)
    full_dir = Path(full_dir_text)
    manifest = read_json(archive_dir / "manifest.json")
    first_panel = select_panel_histories(
        load_panel(archive_dir, int(panel_indices[0])), histories_per_request
    )
    runtime = FrozenLMRuntime(
        adapter,
        Path(checkpoint_text),
        device_name,
        manifest,
        runtime_config,
        first_panel,
        validate_cache,
    )
    runtime.deployment.update(
        {
            "method_key": "frozen_lm",
            "optimizer": "projected_damped_gauss_newton_exact_jvp",
        }
    )
    write_json(
        full_dir / "deployment_initialization" / f"worker_{worker_index:02d}.json",
        runtime.deployment,
    )
    print(
        f"[Frozen-LM worker {worker_index}] resident on {device_name}; "
        f"panels={list(panel_indices)}",
        flush=True,
    )

    thresholds = common_config["reporting"]["parameter_error_thresholds"]
    boundary_tolerance = float(
        common_config["reporting"]["boundary_tolerance_normalized"]
    )
    config_sha = sha256_mapping(runtime_config)
    rows: list[dict[str, Any]] = []
    request_index = 0
    for panel_index in panel_indices:
        print(f"[Frozen-LM worker {worker_index}] starting panel {panel_index:03d}", flush=True)
        panel = select_panel_histories(
            load_panel(archive_dir, int(panel_index)), histories_per_request
        )
        panel_dir = full_dir / f"panel_{panel_index:03d}"
        panel_dir.mkdir(parents=True, exist_ok=True)
        history_features, panel_seconds, panel_cache = runtime.prepare_panel(panel)
        panel_sha = sha256_file(archive_dir / f"panel_{panel_index:03d}.npz")
        panel_cache.update(
            {
                "panel_index": int(panel_index),
                "worker_index": int(worker_index),
                "device": device_name,
                "checkpoint_sha256": runtime.checkpoint_identity["sha256"],
                "panel_archive_sha256": panel_sha,
                "method_key": "frozen_lm",
            }
        )
        write_json(panel_dir / "panel_cache.json", panel_cache)
        panel_rows: list[dict[str, Any]] = []
        panel_request_index = 0
        for case_index in range(case_count):
            for sigma in noise_levels:
                result_path = _result_path(panel_dir, case_index, float(sigma))
                expected = {
                    "protocol_version": common_config["protocol_version"],
                    "system_key": system_key,
                    "method_key": "frozen_lm",
                    "panel_index": int(panel_index),
                    "case_index": int(case_index),
                    "noise_standard_deviation": float(sigma),
                    "checkpoint_sha256": runtime.checkpoint_identity["sha256"],
                    "panel_archive_sha256": panel_sha,
                    "inverse_config_sha256": config_sha,
                    "history_count": int(histories_per_request),
                }
                if resume and result_path.is_file():
                    result = read_json(result_path)
                    if any(result.get(key) != value for key, value in expected.items()):
                        raise RuntimeError(f"incompatible Frozen-LM result: {result_path}")
                    panel_rows.append(result)
                    rows.append(result)
                    request_index += 1
                    panel_request_index += 1
                    continue

                result_path.parent.mkdir(parents=True, exist_ok=True)
                observations = runtime.observations(panel, case_index, float(sigma))
                estimate, objective, trace, timing, top_rows = runtime.invert(
                    history_features, observations
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
                result = {
                    **expected,
                    "status": "completed",
                    "optimizer": runtime_config["optimizer"],
                    "request_index_within_worker": int(request_index),
                    "request_index_within_panel": int(panel_request_index),
                    "archive_history_count": int(manifest["histories_per_panel"]),
                    "history_indices_used": np.asarray(
                        panel["history_indices_used"], dtype=np.int64
                    ).tolist(),
                    "observation_count_per_history": int(manifest["observation_count"]),
                    "scalar_observation_count": int(
                        histories_per_request
                        * manifest["observation_count"]
                        * manifest["state_dim"]
                    ),
                    "selected_observation_mse": float(objective),
                    "panel_initialization_seconds": float(panel_seconds),
                    "compute_device": device_name,
                    "accelerator_name": _device_name(runtime.device),
                    "grid_candidate_count": int(runtime.grid_normalized.shape[0]),
                    "top_k": int(runtime_config["top_k"]),
                    "physics_weight": 0.0,
                    "lbfgsb_used": 0,
                    "gpu_memory_allocated_after_query_bytes": (
                        int(torch.cuda.memory_allocated(runtime.device))
                        if runtime.device.type == "cuda"
                        else 0
                    ),
                    **timing,
                    **metrics,
                }
                write_csv(result_path.parent / "lm_trace.csv", trace)
                write_csv(result_path.parent / "top_grid_starts.csv", top_rows)
                write_json(result_path, result)
                panel_rows.append(result)
                rows.append(result)
                request_index += 1
                panel_request_index += 1
        write_csv(panel_dir / "frozen_lm_per_case_results.csv", panel_rows)
        del history_features, panel
        gc.collect()
        if runtime.device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            f"[Frozen-LM worker {worker_index}] completed panel {panel_index:03d} "
            f"({len(panel_rows)} requests)",
            flush=True,
        )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    started = time.perf_counter()
    archive_dir = args.archive_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    common = read_json(BENCHMARK_ROOT / "configs" / "common.json")
    system = read_json(BENCHMARK_ROOT / "configs" / SYSTEM_CONFIGS[args.system])
    manifest = read_json(archive_dir / "manifest.json")
    system_key = SYSTEM_ADAPTER_KEYS[args.system]
    if str(system["system_key"]) != system_key:
        raise ValueError("system configuration does not match the requested system")
    if str(manifest["system_key"]) != system_key:
        raise ValueError("panel archive does not match the requested system")
    config = read_json(config_path)
    config["grid_points"] = list(args.grid_points or system["grid_points"])
    if args.top_k is not None:
        config["top_k"] = int(args.top_k)
        config["lm_starts"] = int(args.top_k)

    panel_count = int(args.panel_count or manifest["panel_count"])
    case_count = int(args.case_count or manifest["cases_per_panel"])
    histories_per_request = resolve_histories_per_request(
        args.system,
        args.histories_per_request,
        int(manifest["histories_per_panel"]),
    )
    noise_levels = list(
        args.noise_levels or manifest["noise_standard_deviations"]
    )
    if not 1 <= panel_count <= int(manifest["panel_count"]):
        raise ValueError("panel_count lies outside the archive")
    if not 1 <= case_count <= int(manifest["cases_per_panel"]):
        raise ValueError("case_count lies outside the archive")
    if not 1 <= histories_per_request <= int(manifest["histories_per_panel"]):
        raise ValueError("histories_per_request lies outside the archive")
    archived_noise = {float(value) for value in manifest["noise_standard_deviations"]}
    if any(float(value) not in archived_noise for value in noise_levels):
        raise ValueError("requested noise level is absent from the archive")
    if int(config["lm_starts"]) > int(config["top_k"]):
        raise ValueError("lm_starts exceeds top_k")
    if int(config["top_k"]) > int(np.prod(config["grid_points"])):
        raise ValueError("top_k exceeds grid size")

    allowed_existing = {"nohup.log", "pipeline.pid"}
    existing = (
        [path for path in output_dir.iterdir() if path.name not in allowed_existing]
        if output_dir.exists()
        else []
    )
    if existing and not args.resume:
        raise FileExistsError(f"output is not empty; use --resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    full_dir = output_dir / "full"
    full_dir.mkdir(parents=True, exist_ok=True)

    source_root = args.source_root.expanduser().resolve() if args.source_root else None
    adapter = build_adapter(system_key, system, BENCHMARK_ROOT, source_root)
    checkpoint = adapter.resolve_checkpoint(args.checkpoint)
    write_json(
        output_dir / "experiment_config.json",
        {
            "ablation": "Frozen projected Adam replaced by projected damped LM",
            "created_at": datetime.now().isoformat(),
            "requested_system": args.system,
            "system_key": system_key,
            "system": system,
            "archive_dir": str(archive_dir),
            "output_dir": str(output_dir),
            "checkpoint": str(checkpoint),
            "panel_count": panel_count,
            "case_count": case_count,
            "histories_per_request": histories_per_request,
            "noise_standard_deviations": noise_levels,
            "runtime_config": config,
        },
    )
    write_json(
        output_dir / "environment.json",
        {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "requested_devices": args.gpus,
            "pid": os.getpid(),
        },
    )
    write_json(
        output_dir / "pipeline_status.json",
        {"status": "running", "started_at": time.time()},
    )

    devices = list(args.gpus)[: max(1, int(args.processes))]
    active = min(len(devices), panel_count)
    queues = panel_queues(panel_count, active)
    arguments = [
        (
            worker_index,
            devices[worker_index],
            queues[worker_index],
            case_count,
            noise_levels,
            histories_per_request,
            system_key,
            system,
            common,
            config,
            str(source_root) if source_root else None,
            str(checkpoint),
            str(archive_dir),
            str(full_dir),
            args.resume,
            args.validate_cache,
        )
        for worker_index in range(active)
    ]
    try:
        rows: list[dict[str, Any]] = []
        if active == 1:
            rows.extend(run_worker(*arguments[0]))
        else:
            context = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=active, mp_context=context) as executor:
                futures = [executor.submit(run_worker, *values) for values in arguments]
                for future in as_completed(futures):
                    rows.extend(future.result())
        rows.sort(
            key=lambda row: (
                int(row["panel_index"]),
                int(row["case_index"]),
                float(row["noise_standard_deviation"]),
            )
        )
        write_csv(full_dir / "frozen_lm_per_case_results.csv", rows)
        panel_rows = panel_macro_rows(rows)
        noise_rows = noise_summary_rows(panel_rows)
        write_csv(full_dir / "per_panel_results.csv", panel_rows)
        write_csv(full_dir / "per_noise_summary.csv", noise_rows)
        elapsed = float(time.perf_counter() - started)
        write_json(output_dir / "runtime.json", {"total_wall_seconds": elapsed})
        write_json(
            output_dir / "pipeline_status.json",
            {
                "status": "completed",
                "completed_at": time.time(),
                "result_count": len(rows),
                "per_noise_summary": noise_rows,
            },
        )
        print(f"Frozen-LM completed: {len(rows)} requests in {elapsed:.3f}s", flush=True)
        return 0
    except Exception as error:
        write_json(
            output_dir / "pipeline_status.json",
            {
                "status": "failed",
                "failed_at": time.time(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
