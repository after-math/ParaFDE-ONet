"""Panel-resident Frozen operator inversion with strict warm timing."""

from __future__ import annotations

import gc
import math
import os
from pathlib import Path
import platform
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from common.io_utils import (
    read_json, sha256_file, sha256_mapping, write_csv, write_json,
)
from common.metrics import evaluate_estimate
from common.panel_archive import load_panel, select_panel_histories
from systems import build_adapter
from systems.base import synchronize


def _device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return platform.processor() or platform.machine()


class FrozenPanelRuntime:
    """One model-resident runtime owned by one GPU worker process."""

    def __init__(
        self,
        adapter: Any,
        checkpoint_path: Path,
        device_name: str,
        manifest: Mapping[str, Any],
        config: Mapping[str, Any],
        first_panel: Mapping[str, np.ndarray],
        validate_cache: bool,
    ) -> None:
        self.adapter = adapter
        self.device = torch.device(device_name)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.config = dict(config)
        self.manifest = dict(manifest)

        deployment_started = time.perf_counter()
        self.model, self.checkpoint, self.checkpoint_identity = adapter.load_model(
            checkpoint_path, self.device
        )
        load_seconds = float(self.checkpoint_identity["load_seconds"])
        observation_indices = np.asarray(
            first_panel["observation_indices"], dtype=np.int64
        )
        observation_times_np = np.asarray(
            first_panel["output_times"], dtype=np.float64
        )[observation_indices]
        self.observation_times = torch.as_tensor(
            observation_times_np, dtype=torch.float32, device=self.device
        )

        synchronize(self.device)
        trunk_started = time.perf_counter()
        self.trunk_features = adapter.encode_trunk(
            self.model, self.observation_times
        )
        synchronize(self.device)
        trunk_seconds = time.perf_counter() - trunk_started

        points = tuple(int(value) for value in self.config["grid_points"])
        self.grid_normalized = adapter.parameter_grid(self.device, points)
        synchronize(self.device)
        parameter_started = time.perf_counter()
        self.grid_parameter_features, self.grid_physical = adapter.encode_parameter(
            self.model, self.grid_normalized, detach=True
        )
        synchronize(self.device)
        grid_parameter_seconds = time.perf_counter() - parameter_started

        warm_started = time.perf_counter()
        histories = torch.as_tensor(
            first_panel["histories"], dtype=torch.float32, device=self.device
        )
        history_features = adapter.encode_history(self.model, histories)
        observations = self.observations(first_panel, 0, float(manifest["noise_standard_deviations"][0]))
        with torch.no_grad():
            self._grid_objectives(
                history_features,
                observations,
                batch_size=min(64, int(self.config["grid_batch_size"])),
            )
        probe = torch.zeros((2, 2), dtype=torch.float32, device=self.device)
        probe.requires_grad_(True)
        objective, _ = self._objective(probe, history_features, observations)
        objective.sum().backward()
        synchronize(self.device)
        warmup_seconds = time.perf_counter() - warm_started

        self.cache_validation = (
            self._validate_cached_decomposition(histories, observations)
            if validate_cache
            else {"performed": False}
        )
        del probe, objective, history_features, histories, observations
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
        self.deployment = {
            "process_id": int(os.getpid()),
            "device": str(self.device),
            "accelerator_name": _device_name(self.device),
            "checkpoint_load_seconds": load_seconds,
            "trunk_cache_seconds": float(trunk_seconds),
            "grid_parameter_branch_cache_seconds": float(grid_parameter_seconds),
            "cuda_forward_backward_warmup_seconds": float(warmup_seconds),
            "total_deployment_initialization_seconds": float(
                time.perf_counter() - deployment_started
            ),
            "model_loaded_once": True,
            "trunk_computed_once": True,
            "grid_parameter_features_computed_once": True,
            "excluded_from_warm_query_timing": True,
            "checkpoint": self.checkpoint_identity,
            "cache_validation": self.cache_validation,
        }

    def observations(
        self, panel: Mapping[str, np.ndarray], case_index: int, sigma: float
    ) -> torch.Tensor:
        indices = np.asarray(panel["observation_indices"], dtype=np.int64)
        reference = np.asarray(panel["reference"][case_index][:, indices, :])
        noise = np.asarray(panel["standard_normal_noise"][case_index])
        return torch.as_tensor(
            reference + float(sigma) * noise,
            dtype=torch.float32,
            device=self.device,
        )

    def prepare_panel(
        self, panel: Mapping[str, np.ndarray]
    ) -> tuple[torch.Tensor, float, dict[str, Any]]:
        histories = torch.as_tensor(
            panel["histories"], dtype=torch.float32, device=self.device
        )
        synchronize(self.device)
        started = time.perf_counter()
        history_features = self.adapter.encode_history(self.model, histories)
        synchronize(self.device)
        seconds = time.perf_counter() - started
        memory = {
            "history_branch_computed_once": True,
            "history_branch_cache_seconds": float(seconds),
            "history_count": int(histories.shape[0]),
            "gpu_memory_allocated_bytes": (
                int(torch.cuda.memory_allocated(self.device))
                if self.device.type == "cuda"
                else 0
            ),
            "gpu_peak_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(self.device))
                if self.device.type == "cuda"
                else 0
            ),
        }
        return history_features, seconds, memory

    def _objective(
        self,
        normalized: torch.Tensor,
        history_features: torch.Tensor,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        parameter_features, physical = self.adapter.encode_parameter(
            self.model, normalized, detach=False
        )
        prediction = self.adapter.predict_from_features(
            self.model, history_features, parameter_features, self.trunk_features
        )
        loss = torch.mean((prediction - observations.unsqueeze(0)) ** 2, dim=(1, 2, 3))
        if not torch.all(torch.isfinite(loss)):
            raise FloatingPointError("non-finite Frozen observation objective")
        return loss, physical

    def _grid_objectives(
        self,
        history_features: torch.Tensor,
        observations: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        pieces = []
        for start in range(0, self.grid_normalized.shape[0], batch_size):
            prediction = self.adapter.predict_from_features(
                self.model,
                history_features,
                self.grid_parameter_features[start : start + batch_size],
                self.trunk_features,
            )
            pieces.append(
                torch.mean(
                    (prediction - observations.unsqueeze(0)) ** 2,
                    dim=(1, 2, 3),
                )
            )
        result = torch.cat(pieces)
        if not torch.all(torch.isfinite(result)):
            raise FloatingPointError("non-finite Frozen grid objective")
        return result

    def _validate_cached_decomposition(
        self, histories: torch.Tensor, observations: torch.Tensor
    ) -> dict[str, Any]:
        history_features = self.adapter.encode_history(self.model, histories)
        probe_cached = torch.tensor(
            [[-0.43, 0.27], [0.61, -0.19]],
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        cached_loss, _ = self._objective(probe_cached, history_features, observations)
        cached_gradient = torch.autograd.grad(cached_loss.sum(), probe_cached)[0]

        probe_full = probe_cached.detach().clone().requires_grad_(True)
        full_prediction = self.adapter.full_model_prediction(
            self.model, histories, probe_full, self.observation_times
        )
        full_loss = torch.mean(
            (full_prediction - observations.unsqueeze(0)) ** 2, dim=(1, 2, 3)
        )
        full_gradient = torch.autograd.grad(full_loss.sum(), probe_full)[0]
        cached_features, _ = self.adapter.encode_parameter(
            self.model, probe_cached.detach(), detach=True
        )
        cached_prediction = self.adapter.predict_from_features(
            self.model, history_features, cached_features, self.trunk_features
        )
        prediction_difference = float(
            torch.max(torch.abs(cached_prediction - full_prediction.detach())).cpu()
        )
        loss_difference = float(
            torch.max(torch.abs(cached_loss.detach() - full_loss.detach())).cpu()
        )
        gradient_difference = float(
            torch.max(torch.abs(cached_gradient - full_gradient)).cpu()
        )
        passed = (
            prediction_difference <= 2.0e-5
            and loss_difference <= 2.0e-7
            and gradient_difference <= 2.0e-5
        )
        if not passed:
            raise RuntimeError(
                "cached decomposition differs from the original full forward pass: "
                f"prediction={prediction_difference:.3e}, loss={loss_difference:.3e}, "
                f"gradient={gradient_difference:.3e}"
            )
        return {
            "performed": True,
            "passed": True,
            "maximum_prediction_absolute_difference": prediction_difference,
            "maximum_loss_absolute_difference": loss_difference,
            "maximum_parameter_gradient_absolute_difference": gradient_difference,
        }

    def _uncached_objective(
        self,
        normalized: torch.Tensor,
        histories: torch.Tensor,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prediction = self.adapter.full_model_prediction(
            self.model, histories, normalized, self.observation_times
        )
        physical = self.adapter.normalized_to_physical_tensor(normalized)
        loss = torch.mean(
            (prediction - observations.unsqueeze(0)) ** 2, dim=(1, 2, 3)
        )
        return loss, physical

    def replay_uncached_parameter_estimate(
        self,
        starts: np.ndarray,
        histories: torch.Tensor,
        observations: torch.Tensor,
    ) -> np.ndarray:
        """Replay the same projected Adam through the original full forward path."""
        normalized = torch.as_tensor(
            starts, dtype=torch.float32, device=self.device
        ).clone().requires_grad_(True)
        optimizer = torch.optim.Adam(
            [normalized], lr=float(self.config["learning_rate"])
        )
        maximum_steps = int(self.config["adam_max_steps"])
        minimum_steps = int(self.config["adam_min_steps"])
        check_interval = int(self.config["adam_check_interval"])
        stable_checks = 0
        previous_parameters: torch.Tensor | None = None
        previous_objective: float | None = None
        for step in range(1, maximum_steps + 1):
            fraction = (step - 1) / max(maximum_steps - 1, 1)
            learning_rate = float(self.config["minimum_learning_rate"]) + 0.5 * (
                float(self.config["learning_rate"])
                - float(self.config["minimum_learning_rate"])
            ) * (1.0 + math.cos(math.pi * fraction))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            objective, _ = self._uncached_objective(
                normalized, histories, observations
            )
            objective.sum().backward()
            optimizer.step()
            with torch.no_grad():
                normalized.clamp_(-1.0, 1.0)
            if step % check_interval != 0 and step != maximum_steps:
                continue
            with torch.no_grad():
                checked, _ = self._uncached_objective(
                    normalized, histories, observations
                )
                selected = int(torch.argmin(checked).cpu())
                current_parameters = normalized[selected].detach().clone()
                current_objective = float(checked[selected].cpu())
            parameter_change = float("inf")
            relative_change = float("inf")
            if previous_parameters is not None:
                parameter_change = float(
                    torch.max(torch.abs(current_parameters - previous_parameters)).cpu()
                )
            if previous_objective is not None:
                relative_change = abs(current_objective - previous_objective) / max(
                    abs(previous_objective), 1.0e-12
                )
            simultaneous = (
                step >= minimum_steps
                and parameter_change
                < float(self.config["adam_parameter_tolerance"])
                and relative_change
                < float(self.config["adam_relative_objective_tolerance"])
            )
            stable_checks = stable_checks + 1 if simultaneous else 0
            previous_parameters = current_parameters
            previous_objective = current_objective
            if stable_checks >= int(self.config["adam_patience"]):
                break
        with torch.no_grad():
            objective, physical = self._uncached_objective(
                normalized, histories, observations
            )
            selected = int(torch.argmin(objective).cpu())
        return physical[selected].detach().cpu().numpy().astype(np.float64)

    def invert(
        self,
        history_features: torch.Tensor,
        observations: torch.Tensor,
    ) -> tuple[np.ndarray, float, list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
        synchronize(self.device)
        grid_started = time.perf_counter()
        with torch.no_grad():
            grid_objective = self._grid_objectives(
                history_features, observations, int(self.config["grid_batch_size"])
            )
            top_values, top_indices = torch.topk(
                grid_objective,
                k=int(self.config["top_k"]),
                largest=False,
                sorted=True,
            )
            starts = self.grid_normalized.index_select(0, top_indices).clone()
        synchronize(self.device)
        grid_seconds = time.perf_counter() - grid_started
        top_rows = [
            {
                "rank": rank + 1,
                "grid_index": int(top_indices[rank].cpu()),
                "normalized_parameter_0": float(starts[rank, 0].cpu()),
                "normalized_parameter_1": float(starts[rank, 1].cpu()),
                "observation_mse": float(top_values[rank].cpu()),
            }
            for rank in range(starts.shape[0])
        ]

        normalized = starts.detach().requires_grad_(True)
        optimizer = torch.optim.Adam(
            [normalized], lr=float(self.config["learning_rate"])
        )
        maximum_steps = int(self.config["adam_max_steps"])
        minimum_steps = int(self.config["adam_min_steps"])
        check_interval = int(self.config["adam_check_interval"])
        parameter_tolerance = float(self.config["adam_parameter_tolerance"])
        objective_tolerance = float(
            self.config["adam_relative_objective_tolerance"]
        )
        patience = int(self.config["adam_patience"])
        trace: list[dict[str, Any]] = []
        previous_parameters: torch.Tensor | None = None
        previous_objective: float | None = None
        stable_checks = 0
        stopped_early = False

        synchronize(self.device)
        adam_started = time.perf_counter()
        completed = 0
        for step in range(1, maximum_steps + 1):
            fraction = (step - 1) / max(maximum_steps - 1, 1)
            learning_rate = float(self.config["minimum_learning_rate"]) + 0.5 * (
                float(self.config["learning_rate"])
                - float(self.config["minimum_learning_rate"])
            ) * (1.0 + math.cos(math.pi * fraction))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            objective, _ = self._objective(normalized, history_features, observations)
            objective.sum().backward()
            optimizer.step()
            with torch.no_grad():
                normalized.clamp_(-1.0, 1.0)
            completed = step
            if step % check_interval != 0 and step != maximum_steps:
                continue
            with torch.no_grad():
                checked, physical = self._objective(
                    normalized, history_features, observations
                )
                selected = int(torch.argmin(checked).cpu())
                current_parameters = normalized[selected].detach().clone()
                current_objective = float(checked[selected].cpu())
            parameter_change = float("inf")
            relative_objective_change = float("inf")
            if previous_parameters is not None:
                parameter_change = float(
                    torch.max(torch.abs(current_parameters - previous_parameters)).cpu()
                )
            if previous_objective is not None:
                relative_objective_change = abs(
                    current_objective - previous_objective
                ) / max(abs(previous_objective), 1.0e-12)
            simultaneous = (
                step >= minimum_steps
                and parameter_change < parameter_tolerance
                and relative_objective_change < objective_tolerance
            )
            stable_checks = stable_checks + 1 if simultaneous else 0
            trace.append(
                {
                    "step": int(step),
                    "learning_rate": float(learning_rate),
                    "selected_restart": int(selected),
                    "selected_objective": float(current_objective),
                    "selected_parameter_0": float(physical[selected, 0].cpu()),
                    "selected_parameter_1": float(physical[selected, 1].cpu()),
                    "normalized_parameter_change_inf": parameter_change,
                    "relative_objective_change": relative_objective_change,
                    "simultaneous_stability_check": int(simultaneous),
                    "consecutive_stable_checks": int(stable_checks),
                }
            )
            previous_parameters = current_parameters
            previous_objective = current_objective
            if stable_checks >= patience:
                stopped_early = True
                break
        with torch.no_grad():
            final_objective, final_physical = self._objective(
                normalized, history_features, observations
            )
            selected = int(torch.argmin(final_objective).cpu())
            estimate = final_physical[selected].detach().cpu().numpy().astype(np.float64)
            best_objective = float(final_objective[selected].cpu())
        synchronize(self.device)
        adam_seconds = time.perf_counter() - adam_started
        timing = {
            "grid_screening_seconds": float(grid_seconds),
            "projected_adam_seconds": float(adam_seconds),
            "warm_online_seconds": float(grid_seconds + adam_seconds),
            "adam_steps_completed": int(completed),
            "adam_stopped_early": int(stopped_early),
            "adam_stop_reason": (
                "simultaneous_parameter_and_objective_stability"
                if stopped_early
                else "maximum_steps"
            ),
        }
        return estimate, best_objective, trace, timing, top_rows


def _result_path(
    panel_dir: Path, case_index: int, sigma: float
) -> Path:
    noise_tag = str(float(sigma)).replace(".", "p")
    return panel_dir / "jobs" / "frozen" / f"case_{case_index:03d}" / f"noise_{noise_tag}" / "result.json"


def run_frozen_worker(
    worker_index: int,
    device_name: str,
    panel_indices: Sequence[int],
    system_key: str,
    system_config: Mapping[str, Any],
    common_config: Mapping[str, Any],
    benchmark_root_text: str,
    source_root_text: str | None,
    checkpoint_text: str,
    archive_dir_text: str,
    full_dir_text: str,
    frozen_config: Mapping[str, Any],
    histories_per_request: int,
    resume: bool,
    validate_cache: bool,
) -> list[dict[str, Any]]:
    adapter = build_adapter(
        system_key,
        system_config,
        Path(benchmark_root_text),
        Path(source_root_text) if source_root_text else None,
    )
    archive_dir = Path(archive_dir_text)
    full_dir = Path(full_dir_text)
    manifest = read_json(archive_dir / "manifest.json")
    first_panel = select_panel_histories(
        load_panel(archive_dir, int(panel_indices[0])), histories_per_request
    )
    runtime = FrozenPanelRuntime(
        adapter,
        Path(checkpoint_text),
        device_name,
        manifest,
        frozen_config,
        first_panel,
        validate_cache,
    )
    deployment_path = full_dir / "deployment_initialization" / f"worker_{worker_index:02d}.json"
    write_json(deployment_path, runtime.deployment)
    print(
        f"[frozen worker {worker_index}] model resident on {device_name}; "
        f"panels={list(panel_indices)}",
        flush=True,
    )

    thresholds = common_config["reporting"]["parameter_error_thresholds"]
    boundary_tolerance = float(
        common_config["reporting"]["boundary_tolerance_normalized"]
    )
    rows: list[dict[str, Any]] = []
    request_index = 0
    inverse_config_sha256 = sha256_mapping(frozen_config)
    for panel_index in panel_indices:
        print(
            f"[frozen worker {worker_index}] starting panel {int(panel_index):03d}",
            flush=True,
        )
        panel = select_panel_histories(
            load_panel(archive_dir, int(panel_index)), histories_per_request
        )
        panel_dir = full_dir / f"panel_{int(panel_index):03d}"
        panel_dir.mkdir(parents=True, exist_ok=True)
        history_features, panel_seconds, panel_cache = runtime.prepare_panel(panel)
        panel_cache.update(
            {
                "panel_index": int(panel_index),
                "worker_index": int(worker_index),
                "device": device_name,
                "checkpoint_sha256": runtime.checkpoint_identity["sha256"],
                "panel_archive_sha256": sha256_file(
                    archive_dir / f"panel_{int(panel_index):03d}.npz"
                ),
            }
        )
        write_json(panel_dir / "panel_cache.json", panel_cache)
        panel_rows: list[dict[str, Any]] = []
        panel_request_index = 0
        histories_tensor = torch.as_tensor(
            panel["histories"], dtype=torch.float32, device=runtime.device
        )
        for case_index in range(int(manifest["cases_per_panel"])):
            for sigma in manifest["noise_standard_deviations"]:
                result_path = _result_path(panel_dir, case_index, float(sigma))
                if resume and result_path.is_file():
                    result = read_json(result_path)
                    expected = {
                        "protocol_version": common_config["protocol_version"],
                        "system_key": system_key,
                        "method_key": "frozen",
                        "panel_index": int(panel_index),
                        "case_index": int(case_index),
                        "noise_standard_deviation": float(sigma),
                        "checkpoint_sha256": runtime.checkpoint_identity["sha256"],
                        "panel_archive_sha256": panel_cache["panel_archive_sha256"],
                        "inverse_config_sha256": inverse_config_sha256,
                        "history_count": int(histories_per_request),
                    }
                    if any(result.get(key) != value for key, value in expected.items()):
                        raise RuntimeError(f"incompatible completed Frozen result: {result_path}")
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
                cache_parameter_difference = None
                validation_case_count = int(
                    frozen_config.get("cache_parameter_validation_cases", 0)
                )
                if (
                    validate_cache
                    and panel_index == panel_indices[0]
                    and case_index < validation_case_count
                    and float(sigma) == float(manifest["noise_standard_deviations"][0])
                ):
                    starts = np.asarray(
                        [
                            [row["normalized_parameter_0"], row["normalized_parameter_1"]]
                            for row in top_rows
                        ],
                        dtype=np.float32,
                    )
                    uncached_estimate = runtime.replay_uncached_parameter_estimate(
                        starts, histories_tensor, observations
                    )
                    cache_parameter_difference = float(
                        np.max(np.abs(estimate - uncached_estimate))
                    )
                    if cache_parameter_difference > 5.0e-5:
                        raise RuntimeError(
                            "cached and uncached projected-Adam estimates differ: "
                            f"{cache_parameter_difference:.3e}"
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
                    "protocol_version": common_config["protocol_version"],
                    "system_key": system_key,
                    "method_key": "frozen",
                    "panel_index": int(panel_index),
                    "case_index": int(case_index),
                    "request_index_within_worker": int(request_index),
                    "request_index_within_panel": int(panel_request_index),
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
                    "panel_initialization_seconds": float(panel_seconds),
                    "checkpoint_sha256": runtime.checkpoint_identity["sha256"],
                    "panel_archive_sha256": panel_cache["panel_archive_sha256"],
                    "inverse_config_sha256": inverse_config_sha256,
                    "compute_device": device_name,
                    "accelerator_name": _device_name(runtime.device),
                    "grid_candidate_count": int(runtime.grid_normalized.shape[0]),
                    "top_k": int(frozen_config["top_k"]),
                    "physics_weight": 0.0,
                    "lbfgsb_used": 0,
                    "cached_uncached_parameter_max_abs_difference": cache_parameter_difference,
                    "gpu_memory_allocated_after_query_bytes": (
                        int(torch.cuda.memory_allocated(runtime.device))
                        if runtime.device.type == "cuda"
                        else 0
                    ),
                    **timing,
                    **metrics,
                }
                write_csv(result_path.parent / "adam_trace.csv", trace)
                write_csv(result_path.parent / "top_grid_starts.csv", top_rows)
                write_json(result_path, result)
                panel_rows.append(result)
                rows.append(result)
                request_index += 1
                panel_request_index += 1
        write_csv(panel_dir / "frozen_per_case_results.csv", panel_rows)
        del history_features, histories_tensor, panel
        gc.collect()
        if runtime.device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            f"[frozen worker {worker_index}] completed panel {int(panel_index):03d} "
            f"({len(panel_rows)} requests)",
            flush=True,
        )
    return rows
