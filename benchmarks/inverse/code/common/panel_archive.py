"""Generate immutable panel archives shared by Frozen, LM and DE."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from common.io_utils import (
    read_json,
    sha256_arrays,
    sha256_file,
    write_json,
)
from common.parameters import build_parameter_design
from systems import build_adapter


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _build_panel(
    adapter: Any,
    panel_index: int,
    histories_per_panel: int,
    physical_parameters: np.ndarray,
    output_points: int,
    observation_count: int,
    truth_internal_step: float,
    data_seed: int,
    noise_seed: int,
    output_path: Path,
    resume: bool,
) -> dict[str, Any]:
    if resume and output_path.is_file():
        with np.load(output_path, allow_pickle=False) as values:
            required = {
                "histories", "history_grid", "parameters", "output_times",
                "observation_indices", "reference", "standard_normal_noise",
                "history_labels", "panel_index",
            }
            if required.issubset(values.files):
                expected_reference_shape = (
                    physical_parameters.shape[0],
                    histories_per_panel,
                    output_points,
                    adapter.state_dim,
                )
                expected_noise_shape = (
                    physical_parameters.shape[0],
                    histories_per_panel,
                    observation_count,
                    adapter.state_dim,
                )
                compatible = (
                    values["histories"].shape
                    == (histories_per_panel, adapter.state_dim, adapter.history_sensors)
                    and values["reference"].shape == expected_reference_shape
                    and values["standard_normal_noise"].shape == expected_noise_shape
                    and values["parameters"].shape == physical_parameters.shape
                    and np.allclose(
                        values["parameters"], physical_parameters, rtol=0.0, atol=0.0
                    )
                    and int(values["panel_index"]) == panel_index
                )
                if not compatible:
                    raise RuntimeError(
                        f"existing panel is incompatible with requested protocol: {output_path}"
                    )
                return {
                    "panel_index": int(panel_index),
                    "path": output_path.name,
                    "sha256": sha256_file(output_path),
                    "content_sha256": (
                        str(values["content_sha256"])
                        if "content_sha256" in values.files
                        else None
                    ),
                    "reused": True,
                }

    history_rng = np.random.default_rng(data_seed + panel_index * 10_007)
    noise_rng = np.random.default_rng(noise_seed + panel_index * 10_009)
    histories, history_labels = adapter.sample_panel_histories(
        histories_per_panel, history_rng
    )
    history_grid = adapter.history_grid()
    output_times = adapter.output_times(output_points)
    observation_indices = adapter.observation_indices(output_points, observation_count)
    case_count = physical_parameters.shape[0]
    repeated_histories = np.tile(histories, (case_count, 1, 1))
    repeated_parameters = np.repeat(physical_parameters, histories_per_panel, axis=0)
    reference = adapter.solve_batch(
        repeated_histories,
        repeated_parameters,
        history_grid,
        output_times,
        truth_internal_step,
    ).reshape(
        case_count,
        histories_per_panel,
        output_points,
        adapter.state_dim,
    )
    standard_normal_noise = noise_rng.normal(
        size=(
            case_count,
            histories_per_panel,
            observation_count,
            adapter.state_dim,
        )
    ).astype(np.float32)
    arrays = {
        "panel_index": np.asarray(panel_index, dtype=np.int64),
        "histories": histories.astype(np.float32),
        "history_grid": history_grid.astype(np.float64),
        "parameters": physical_parameters.astype(np.float64),
        "output_times": output_times.astype(np.float64),
        "observation_indices": observation_indices.astype(np.int64),
        "reference": reference.astype(np.float32),
        "standard_normal_noise": standard_normal_noise,
        "history_labels": np.asarray(history_labels, dtype="<U32"),
    }
    arrays["content_sha256"] = np.asarray(sha256_arrays(**arrays), dtype="<U64")
    _atomic_savez(output_path, **arrays)
    return {
        "panel_index": int(panel_index),
        "path": output_path.name,
        "sha256": sha256_file(output_path),
        "content_sha256": str(arrays["content_sha256"]),
        "reused": False,
    }


def _generate_panel_chunk(
    system_key: str,
    system_config: Mapping[str, Any],
    benchmark_root_text: str,
    source_root_text: str | None,
    panel_indices: Sequence[int],
    histories_per_panel: int,
    physical_parameters: np.ndarray,
    output_points: int,
    observation_count: int,
    truth_internal_step: float,
    data_seed: int,
    noise_seed: int,
    archive_dir_text: str,
    resume: bool,
) -> list[dict[str, Any]]:
    adapter = build_adapter(
        system_key,
        system_config,
        Path(benchmark_root_text),
        Path(source_root_text) if source_root_text else None,
    )
    archive_dir = Path(archive_dir_text)
    rows = []
    for panel_index in panel_indices:
        print(
            f"[data:{system_key}] generating panel {panel_index:03d}", flush=True
        )
        rows.append(_build_panel(
            adapter,
            panel_index,
            histories_per_panel,
            physical_parameters,
            output_points,
            observation_count,
            truth_internal_step,
            data_seed,
            noise_seed,
            archive_dir / f"panel_{panel_index:03d}.npz",
            resume,
        ))
        print(
            f"[data:{system_key}] completed panel {panel_index:03d}", flush=True
        )
    return rows


def generate_archive(
    adapter: Any,
    system_config: Mapping[str, Any],
    common_config: Mapping[str, Any],
    benchmark_root: Path,
    source_root_override: Path | None,
    archive_dir: Path,
    panel_count: int,
    histories_per_panel: int,
    case_count: int,
    noise_levels: Sequence[float],
    observation_count: int,
    truth_internal_step: float,
    output_points: int,
    workers: int,
    resume: bool,
) -> dict[str, Any]:
    archive_dir.mkdir(parents=True, exist_ok=True)
    unit_parameters, parameter_labels = build_parameter_design(
        case_count, common_config["parameter_design"]
    )
    physical_parameters = adapter.unit_to_physical_numpy(unit_parameters)
    np.save(archive_dir / "parameters_unit.npy", unit_parameters)
    np.save(archive_dir / "parameters_physical.npy", physical_parameters)
    np.save(archive_dir / "parameter_labels.npy", parameter_labels)

    data_seed = int(common_config["data"]["seed"])
    noise_seed = int(common_config["data"]["noise_seed"])
    worker_count = max(1, min(int(workers), panel_count))
    queues = [list(range(index, panel_count, worker_count)) for index in range(worker_count)]
    panel_rows: list[dict[str, Any]] = []
    if worker_count == 1:
        panel_rows.extend(
            _generate_panel_chunk(
                adapter.system_key,
                system_config,
                str(benchmark_root),
                str(source_root_override) if source_root_override else None,
                queues[0],
                histories_per_panel,
                physical_parameters,
                output_points,
                observation_count,
                truth_internal_step,
                data_seed,
                noise_seed,
                str(archive_dir),
                resume,
            )
        )
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _generate_panel_chunk,
                    adapter.system_key,
                    dict(system_config),
                    str(benchmark_root),
                    str(source_root_override) if source_root_override else None,
                    queue,
                    histories_per_panel,
                    physical_parameters,
                    output_points,
                    observation_count,
                    truth_internal_step,
                    data_seed,
                    noise_seed,
                    str(archive_dir),
                    resume,
                )
                for queue in queues
            ]
            for future in as_completed(futures):
                panel_rows.extend(future.result())

    panel_rows.sort(key=lambda row: int(row["panel_index"]))
    manifest = {
        "protocol_version": common_config["protocol_version"],
        "system_key": adapter.system_key,
        "display_name": system_config["display_name"],
        "panel_count": int(panel_count),
        "histories_per_panel": int(histories_per_panel),
        "cases_per_panel": int(case_count),
        "noise_standard_deviations": [float(value) for value in noise_levels],
        "state_dim": int(adapter.state_dim),
        "parameter_names": list(adapter.parameter_names),
        "parameter_bounds": adapter.parameter_bounds.tolist(),
        "observation_count": int(observation_count),
        "output_points": int(output_points),
        "truth_internal_step": float(truth_internal_step),
        "history_sensors": int(adapter.history_sensors),
        "horizon": float(adapter.horizon),
        "parameter_design_sha256": sha256_arrays(
            unit=unit_parameters,
            physical=physical_parameters,
            labels=parameter_labels,
        ),
        "training_resolved_config": str(adapter.training_config_path),
        "training_resolved_config_sha256": sha256_file(adapter.training_config_path),
        "panels": panel_rows,
    }
    write_json(archive_dir / "manifest.json", manifest)
    return manifest


def load_panel(archive_dir: Path, panel_index: int, verify: bool = True) -> dict[str, np.ndarray]:
    path = archive_dir / f"panel_{panel_index:03d}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    if verify:
        manifest = read_json(archive_dir / "manifest.json")
        row = next(
            item for item in manifest["panels"] if int(item["panel_index"]) == panel_index
        )
        if sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"panel archive hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def select_panel_histories(
    panel: Mapping[str, np.ndarray], history_count: int
) -> dict[str, np.ndarray]:
    """Return a deterministic in-memory view using the first histories in a panel.

    The immutable archive remains unchanged.  Reference trajectories and noise are
    sliced on the same history axis so single- and multi-history runs remain paired.
    """
    available = int(np.asarray(panel["histories"]).shape[0])
    requested = int(history_count)
    if requested < 1 or requested > available:
        raise ValueError(
            f"history_count must lie in [1, {available}], received {requested}"
        )
    indices = np.arange(requested, dtype=np.int64)
    selected = dict(panel)
    selected["histories"] = np.asarray(panel["histories"])[indices]
    if "history_labels" in panel:
        selected["history_labels"] = np.asarray(panel["history_labels"])[indices]
    selected["reference"] = np.asarray(panel["reference"])[:, indices, ...]
    selected["standard_normal_noise"] = np.asarray(
        panel["standard_normal_noise"]
    )[:, indices, ...]
    selected["history_indices_used"] = indices
    return selected
