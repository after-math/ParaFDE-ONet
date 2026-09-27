"""Schedule whole panels without splitting a panel across GPU workers."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
from typing import Any, Mapping, Sequence

from common.frozen_panel_runner import run_frozen_worker
from common.de_panel_runner import run_de_panel
from common.lm_panel_runner import run_lm_panel


def panel_queues(panel_count: int, worker_count: int) -> list[list[int]]:
    if panel_count < 1 or worker_count < 1:
        raise ValueError("panel and worker counts must be positive")
    workers = min(panel_count, worker_count)
    return [list(range(index, panel_count, workers)) for index in range(workers)]


def run_frozen_panels(
    devices: Sequence[str],
    panel_count: int,
    system_key: str,
    system_config: Mapping[str, Any],
    common_config: Mapping[str, Any],
    benchmark_root: Path,
    source_root_override: Path | None,
    checkpoint_path: Path,
    archive_dir: Path,
    full_dir: Path,
    frozen_config: Mapping[str, Any],
    histories_per_request: int,
    resume: bool,
    validate_cache: bool,
) -> list[dict[str, Any]]:
    active = min(len(devices), panel_count)
    if active < 1:
        raise ValueError("at least one Frozen device is required")
    queues = panel_queues(panel_count, active)
    arguments = [
        (
            worker_index,
            str(devices[worker_index]),
            queues[worker_index],
            system_key,
            dict(system_config),
            dict(common_config),
            str(benchmark_root),
            str(source_root_override) if source_root_override else None,
            str(checkpoint_path),
            str(archive_dir),
            str(full_dir),
            dict(frozen_config),
            histories_per_request,
            resume,
            validate_cache,
        )
        for worker_index in range(active)
    ]
    rows: list[dict[str, Any]] = []
    if active == 1:
        rows.extend(run_frozen_worker(*arguments[0]))
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=active, mp_context=context) as executor:
            futures = [executor.submit(run_frozen_worker, *values) for values in arguments]
            for future in as_completed(futures):
                rows.extend(future.result())
    return sorted(
        rows,
        key=lambda row: (
            int(row["panel_index"]),
            int(row["case_index"]),
            float(row["noise_standard_deviation"]),
        ),
    )


def run_lm_panels(
    worker_count: int,
    panel_count: int,
    system_key: str,
    system_config: Mapping[str, Any],
    common_config: Mapping[str, Any],
    benchmark_root: Path,
    source_root_override: Path | None,
    archive_dir: Path,
    full_dir: Path,
    lm_config: Mapping[str, Any],
    histories_per_request: int,
    resume: bool,
) -> list[dict[str, Any]]:
    active = max(1, min(int(worker_count), panel_count))
    arguments = [
        (
            panel_index,
            system_key,
            dict(system_config),
            dict(common_config),
            str(benchmark_root),
            str(source_root_override) if source_root_override else None,
            str(archive_dir),
            str(full_dir),
            dict(lm_config),
            histories_per_request,
            resume,
        )
        for panel_index in range(panel_count)
    ]
    rows: list[dict[str, Any]] = []
    if active == 1:
        for values in arguments:
            rows.extend(run_lm_panel(*values))
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=active, mp_context=context) as executor:
            futures = [executor.submit(run_lm_panel, *values) for values in arguments]
            for future in as_completed(futures):
                rows.extend(future.result())
    return sorted(
        rows,
        key=lambda row: (
            int(row["panel_index"]),
            int(row["case_index"]),
            float(row["noise_standard_deviation"]),
        ),
    )


def run_de_panels(
    worker_count: int,
    panel_count: int,
    system_key: str,
    system_config: Mapping[str, Any],
    common_config: Mapping[str, Any],
    benchmark_root: Path,
    source_root_override: Path | None,
    archive_dir: Path,
    full_dir: Path,
    de_config: Mapping[str, Any],
    histories_per_request: int,
    resume: bool,
) -> list[dict[str, Any]]:
    active = max(1, min(int(worker_count), panel_count))
    arguments = [
        (
            panel_index,
            system_key,
            dict(system_config),
            dict(common_config),
            str(benchmark_root),
            str(source_root_override) if source_root_override else None,
            str(archive_dir),
            str(full_dir),
            dict(de_config),
            histories_per_request,
            resume,
        )
        for panel_index in range(panel_count)
    ]
    rows: list[dict[str, Any]] = []
    if active == 1:
        for values in arguments:
            rows.extend(run_de_panel(*values))
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=active, mp_context=context) as executor:
            futures = [executor.submit(run_de_panel, *values) for values in arguments]
            for future in as_completed(futures):
                rows.extend(future.result())
    return sorted(
        rows,
        key=lambda row: (
            int(row["panel_index"]),
            int(row["case_index"]),
            float(row["noise_standard_deviation"]),
        ),
    )
