"""Frozen-operator inversion with projected damped LM and exact JVPs."""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping

import numpy as np
import torch

from common.frozen_panel_runner import FrozenPanelRuntime
from systems.base import synchronize


ResidualFunction = Callable[[torch.Tensor], torch.Tensor]


def projected_damped_lm(
    residual_function: ResidualFunction,
    starts: torch.Tensor,
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, float, list[dict[str, Any]], dict[str, Any]]:
    """Minimize a batched residual in normalized coordinates on ``[-1, 1]^2``."""
    if starts.ndim != 2 or starts.shape[1] != 2:
        raise ValueError("LM starts must have shape (restart_count, 2)")
    restart_count = int(starts.shape[0])
    if restart_count < 1:
        raise ValueError("at least one LM restart is required")

    normalized = starts.detach().clone()
    device = normalized.device
    damping = torch.full(
        (restart_count,),
        float(config["lm_initial_damping"]),
        dtype=torch.float64,
        device=device,
    )
    damping_increase = float(config["lm_damping_increase"])
    damping_decrease = float(config["lm_damping_decrease"])
    minimum_damping = float(config["lm_minimum_damping"])
    maximum_damping = float(config["lm_maximum_damping"])
    minimum_iterations = int(config["lm_min_iterations"])
    maximum_iterations = int(config["lm_max_iterations"])
    parameter_tolerance = float(config["lm_parameter_tolerance"])
    objective_tolerance = float(config["lm_relative_objective_tolerance"])
    patience = int(config["lm_early_stop_patience"])
    identity = torch.eye(2, dtype=torch.float64, device=device).expand(
        restart_count, -1, -1
    )

    previous_best: torch.Tensor | None = None
    previous_objective: float | None = None
    stable_count = 0
    stopped_early = False
    accepted_steps = 0
    trace: list[dict[str, Any]] = []
    completed = 0

    for iteration in range(1, maximum_iterations + 1):
        current = normalized.detach().requires_grad_(True)
        current_residual = residual_function(current)
        if current_residual.ndim != 2 or current_residual.shape[0] != restart_count:
            raise ValueError("LM residual must have shape (restart_count, residual_count)")
        if not torch.all(torch.isfinite(current_residual)):
            raise FloatingPointError("non-finite Frozen-LM residual")

        columns = []
        for parameter_index in range(2):
            tangent = torch.zeros_like(current)
            tangent[:, parameter_index] = 1.0
            _, column = torch.autograd.functional.jvp(
                residual_function,
                (current,),
                (tangent,),
                create_graph=False,
                strict=False,
            )
            columns.append(column.detach())

        residual64 = current_residual.detach().to(torch.float64)
        jacobian64 = torch.stack(columns, dim=2).to(torch.float64)
        residual_count = int(residual64.shape[1])
        objectives = torch.mean(residual64.square(), dim=1)
        normal = torch.einsum("bmi,bmj->bij", jacobian64, jacobian64) / residual_count
        gradient = torch.einsum("bmi,bm->bi", jacobian64, residual64) / residual_count
        scaling = torch.diagonal(normal, dim1=1, dim2=2).clamp_min(1.0e-12)
        system = (
            normal
            + damping[:, None, None] * torch.diag_embed(scaling)
            + 1.0e-12 * identity
        )
        try:
            delta = torch.linalg.solve(system, -gradient.unsqueeze(2)).squeeze(2)
        except RuntimeError:
            delta = torch.linalg.lstsq(system, -gradient.unsqueeze(2)).solution.squeeze(2)
        proposed = torch.clamp(current.detach() + delta.to(torch.float32), -1.0, 1.0)

        with torch.no_grad():
            proposed_residual = residual_function(proposed)
            proposed_objectives = torch.mean(
                proposed_residual.to(torch.float64).square(), dim=1
            )
            accepted = proposed_objectives < objectives
            accepted_steps += int(accepted.sum().cpu())
            normalized = torch.where(accepted[:, None], proposed, current.detach())
            updated_objectives = torch.where(accepted, proposed_objectives, objectives)
            damping = torch.where(
                accepted,
                damping / damping_decrease,
                damping * damping_increase,
            ).clamp(minimum_damping, maximum_damping)
            selected = int(torch.argmin(updated_objectives).cpu())
            best = normalized[selected].detach().clone()
            best_objective = float(updated_objectives[selected].cpu())

        parameter_change = float("inf")
        relative_objective_change = float("inf")
        if previous_best is not None:
            parameter_change = float(torch.max(torch.abs(best - previous_best)).cpu())
        if previous_objective is not None:
            relative_objective_change = abs(
                best_objective - previous_objective
            ) / max(abs(previous_objective), 1.0e-12)
        simultaneous = (
            iteration >= minimum_iterations
            and parameter_change < parameter_tolerance
            and relative_objective_change < objective_tolerance
        )
        stable_count = stable_count + 1 if simultaneous else 0
        trace.append(
            {
                "iteration": int(iteration),
                "selected_restart": int(selected),
                "selected_objective": float(best_objective),
                "selected_normalized_parameter_0": float(best[0].cpu()),
                "selected_normalized_parameter_1": float(best[1].cpu()),
                "normalized_parameter_change_inf": parameter_change,
                "relative_objective_change": relative_objective_change,
                "accepted_restart_steps": int(accepted.sum().cpu()),
                "mean_damping": float(damping.mean().cpu()),
                "simultaneous_stability_check": int(simultaneous),
                "consecutive_stable_checks": int(stable_count),
            }
        )
        previous_best = best
        previous_objective = best_objective
        completed = iteration
        if stable_count >= patience:
            stopped_early = True
            break

    with torch.no_grad():
        final_residual = residual_function(normalized)
        final_objectives = torch.mean(final_residual.square(), dim=1)
        selected = int(torch.argmin(final_objectives).cpu())
        best_normalized = normalized[selected].detach().clone()
        best_objective = float(final_objectives[selected].cpu())
    metadata = {
        "lm_iterations_completed": int(completed),
        "lm_stopped_early": int(stopped_early),
        "lm_stop_reason": (
            "simultaneous_parameter_and_objective_stability"
            if stopped_early
            else "maximum_iterations"
        ),
        "lm_accepted_restart_steps": int(accepted_steps),
        "lm_final_damping_mean": float(damping.mean().cpu()),
        "jacobian_jvp_calls": int(2 * completed),
        "lm_restart_count": int(restart_count),
        "forward_parameter_evaluations_after_grid": int(
            restart_count * (4 * completed + 1)
        ),
    }
    return best_normalized, best_objective, trace, metadata


class FrozenLMRuntime(FrozenPanelRuntime):
    """Keep Frozen feature reuse and replace projected Adam by damped LM."""

    def __init__(
        self,
        adapter: Any,
        checkpoint_path: Any,
        device_name: str,
        manifest: Mapping[str, Any],
        config: Mapping[str, Any],
        first_panel: Mapping[str, np.ndarray],
        validate_cache: bool,
    ) -> None:
        super().__init__(
            adapter,
            checkpoint_path,
            device_name,
            manifest,
            config,
            first_panel,
            validate_cache,
        )
        histories = torch.as_tensor(
            first_panel["histories"], dtype=torch.float32, device=self.device
        )
        history_features = self.adapter.encode_history(self.model, histories)
        observations = self.observations(
            first_panel, 0, float(manifest["noise_standard_deviations"][0])
        )
        starts = torch.tensor(
            [[-0.25, 0.25], [0.25, -0.25]],
            dtype=torch.float32,
            device=self.device,
        )

        def warmup_residual(normalized: torch.Tensor) -> torch.Tensor:
            parameter_features, _ = self.adapter.encode_parameter(
                self.model, normalized, detach=False
            )
            prediction = self.adapter.predict_from_features(
                self.model,
                history_features,
                parameter_features,
                self.trunk_features,
            )
            return (prediction - observations.unsqueeze(0)).reshape(
                normalized.shape[0], -1
            )

        warmup_config = dict(config)
        warmup_config.update(
            {
                "lm_min_iterations": 1,
                "lm_max_iterations": 1,
                "lm_early_stop_patience": 1,
            }
        )
        synchronize(self.device)
        started = time.perf_counter()
        projected_damped_lm(warmup_residual, starts, warmup_config)
        synchronize(self.device)
        seconds = float(time.perf_counter() - started)
        self.deployment["cuda_exact_jvp_lm_warmup_seconds"] = seconds
        self.deployment["exact_jvp_lm_warmed_once"] = True
        self.deployment["total_deployment_initialization_seconds"] = float(
            self.deployment["total_deployment_initialization_seconds"] + seconds
        )

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
        lm_starts = min(int(self.config["lm_starts"]), int(starts.shape[0]))
        starts = starts[:lm_starts]
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

        def residual_function(normalized: torch.Tensor) -> torch.Tensor:
            parameter_features, _ = self.adapter.encode_parameter(
                self.model, normalized, detach=False
            )
            prediction = self.adapter.predict_from_features(
                self.model,
                history_features,
                parameter_features,
                self.trunk_features,
            )
            residual = prediction - observations.unsqueeze(0)
            return residual.reshape(normalized.shape[0], -1)

        synchronize(self.device)
        lm_started = time.perf_counter()
        best_normalized, _, trace, metadata = projected_damped_lm(
            residual_function, starts, self.config
        )
        with torch.no_grad():
            final_objective, final_physical = self._objective(
                best_normalized.unsqueeze(0), history_features, observations
            )
            estimate = final_physical[0].detach().cpu().numpy().astype(np.float64)
            best_objective = float(final_objective[0].cpu())
        synchronize(self.device)
        lm_seconds = time.perf_counter() - lm_started

        if trace:
            normalized_trace = torch.as_tensor(
                [
                    [row["selected_normalized_parameter_0"], row["selected_normalized_parameter_1"]]
                    for row in trace
                ],
                dtype=torch.float32,
                device=self.device,
            )
            with torch.no_grad():
                physical_trace = self.adapter.normalized_to_physical_tensor(
                    normalized_trace
                ).cpu().numpy()
            for row, physical in zip(trace, physical_trace):
                row["selected_parameter_0"] = float(physical[0])
                row["selected_parameter_1"] = float(physical[1])

        timing = {
            "grid_screening_seconds": float(grid_seconds),
            "projected_lm_seconds": float(lm_seconds),
            "warm_online_seconds": float(grid_seconds + lm_seconds),
            "candidate_objective_evaluations": int(
                self.grid_normalized.shape[0]
                + metadata["forward_parameter_evaluations_after_grid"]
            ),
            **metadata,
        }
        return estimate, best_objective, trace, timing, top_rows
