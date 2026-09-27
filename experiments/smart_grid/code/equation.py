"""Eight-state delayed swing equation and a causal fixed-step RK4 solver."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np


NODE_COUNT = 4
STATE_DIM = 8
PARAMETER_DIM = 3
STATE_NAMES = (
    "theta_1", "theta_2", "theta_3", "theta_4",
    "omega_1", "omega_2", "omega_3", "omega_4",
)
PARAMETER_NAMES = ("coupling_K", "response_gamma", "damping_alpha")


@dataclass(frozen=True)
class SmartGridConfig:
    """Immutable physical definition of the four-node delayed smart grid."""

    maximum_history: float
    fixed_delay: float
    power_injections: tuple[float, float, float, float]
    edges: tuple[tuple[int, int], ...]
    parameter_bounds: tuple[tuple[float, float], ...]
    nominal_parameters: tuple[float, float, float]

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SmartGridConfig":
        powers = tuple(float(item) for item in values["power_injections"])
        edges = tuple(tuple(int(index) for index in edge) for edge in values["star_edges_zero_based"])
        bounds = tuple(tuple(float(item) for item in row) for row in values["parameter_bounds"])
        nominal = tuple(float(item) for item in values["nominal_parameters"])
        result = cls(
            maximum_history=float(values["maximum_history"]),
            fixed_delay=float(values["fixed_delay"]),
            power_injections=powers,  # type: ignore[arg-type]
            edges=edges,
            parameter_bounds=bounds,
            nominal_parameters=nominal,  # type: ignore[arg-type]
        )
        result.validate()
        return result

    def validate(self) -> None:
        if len(self.power_injections) != NODE_COUNT:
            raise ValueError("power_injections must contain four values")
        if not math.isclose(sum(self.power_injections), 0.0, abs_tol=1e-12):
            raise ValueError("power injections must balance to zero")
        if self.maximum_history <= 0.0 or self.fixed_delay <= 0.0:
            raise ValueError("history length and delay must be positive")
        if self.fixed_delay > self.maximum_history + 1e-12:
            raise ValueError("maximum_history must cover fixed_delay")
        if len(self.parameter_bounds) != PARAMETER_DIM or len(self.nominal_parameters) != PARAMETER_DIM:
            raise ValueError("three parameter bounds and nominal values are required")
        for bounds, nominal in zip(self.parameter_bounds, self.nominal_parameters):
            if len(bounds) != 2 or not bounds[0] < bounds[1]:
                raise ValueError("every parameter bound must be increasing")
            if not bounds[0] <= nominal <= bounds[1]:
                raise ValueError("nominal parameter lies outside its bounds")
        normalized_edges = {tuple(sorted(edge)) for edge in self.edges}
        if normalized_edges != {(0, 1), (0, 2), (0, 3)}:
            raise ValueError("the configured topology must be the four-node star")

    @property
    def parameter_lower(self) -> np.ndarray:
        return np.asarray([row[0] for row in self.parameter_bounds], dtype=np.float64)

    @property
    def parameter_upper(self) -> np.ndarray:
        return np.asarray([row[1] for row in self.parameter_bounds], dtype=np.float64)

    @property
    def parameter_span(self) -> np.ndarray:
        return self.parameter_upper - self.parameter_lower


def nominal_equilibrium_angles(coupling: float) -> np.ndarray:
    """Return one gauge-fixed equilibrium for the balanced star network."""
    if coupling <= 1.0:
        raise ValueError("coupling must exceed the unit consumer demand")
    consumer_angle = -math.asin(1.0 / coupling)
    return np.asarray([0.0, consumer_angle, consumer_angle, consumer_angle], dtype=np.float64)


def _broadcast_parameters(parameters: Any, state_ndim: int) -> tuple[Any, Any, Any]:
    coupling = parameters[..., 0]
    response = parameters[..., 1]
    damping = parameters[..., 2]
    while coupling.ndim < state_ndim - 1:
        coupling = coupling[..., None]
        response = response[..., None]
        damping = damping[..., None]
    return coupling, response, damping


def rhs_numpy(
    current: np.ndarray,
    delayed: np.ndarray,
    parameters: np.ndarray,
    config: SmartGridConfig,
) -> np.ndarray:
    """Evaluate the physical DDE right-hand side with NumPy broadcasting."""
    current_values = np.asarray(current, dtype=np.float64)
    delayed_values = np.asarray(delayed, dtype=np.float64)
    physical_parameters = np.asarray(parameters, dtype=np.float64)
    if current_values.shape[-1] != STATE_DIM or delayed_values.shape != current_values.shape:
        raise ValueError("current and delayed states must have the same final dimension eight")
    if physical_parameters.shape[-1] != PARAMETER_DIM:
        raise ValueError("parameters must end in dimension three")
    coupling, response, damping = _broadcast_parameters(physical_parameters, current_values.ndim)
    theta = current_values[..., :NODE_COUNT]
    omega = current_values[..., NODE_COUNT:]
    delayed_omega = delayed_values[..., NODE_COUNT:]
    power_flow = np.zeros_like(theta)
    for left, right in config.edges:
        flow = coupling * np.sin(theta[..., right] - theta[..., left])
        power_flow[..., left] += flow
        power_flow[..., right] -= flow
    powers = np.asarray(config.power_injections, dtype=np.float64)
    acceleration = powers - damping[..., None] * omega + power_flow - response[..., None] * delayed_omega
    return np.concatenate((omega, acceleration), axis=-1)


def rhs_torch(current: Any, delayed: Any, parameters: Any, config: SmartGridConfig) -> Any:
    """Evaluate the same DDE right-hand side while preserving Torch autograd."""
    import torch

    if current.shape[-1] != STATE_DIM or delayed.shape != current.shape:
        raise ValueError("current and delayed states must have the same final dimension eight")
    if parameters.shape[-1] != PARAMETER_DIM:
        raise ValueError("parameters must end in dimension three")
    coupling, response, damping = _broadcast_parameters(parameters, current.ndim)
    theta = current[..., :NODE_COUNT]
    omega = current[..., NODE_COUNT:]
    delayed_omega = delayed[..., NODE_COUNT:]
    power_flow = torch.zeros_like(theta)
    for left, right in config.edges:
        flow = coupling * torch.sin(theta[..., right] - theta[..., left])
        updated_left = power_flow[..., left] + flow
        updated_right = power_flow[..., right] - flow
        power_flow = power_flow.clone()
        power_flow[..., left] = updated_left
        power_flow[..., right] = updated_right
    powers = torch.as_tensor(config.power_injections, dtype=current.dtype, device=current.device)
    acceleration = powers - damping.unsqueeze(-1) * omega + power_flow - response.unsqueeze(-1) * delayed_omega
    return torch.cat((omega, acceleration), dim=-1)


def interpolate_history(
    histories: np.ndarray, history_grid: np.ndarray, query_times: np.ndarray
) -> np.ndarray:
    """Linearly interpolate a batch of eight-component histories."""
    queries = np.asarray(query_times, dtype=np.float64)
    if queries.shape[0] != histories.shape[0]:
        raise ValueError("query batch must equal history batch")
    clipped = np.clip(queries, float(history_grid[0]), float(history_grid[-1]))
    upper = np.clip(np.searchsorted(history_grid, clipped, side="right"), 1, history_grid.size - 1)
    lower = upper - 1
    denominator = np.maximum(history_grid[upper] - history_grid[lower], 1e-15)
    weight = (clipped - history_grid[lower]) / denominator
    rows = np.arange(histories.shape[0]).reshape((-1,) + (1,) * (queries.ndim - 1))
    left = histories[rows, :, lower]
    right = histories[rows, :, upper]
    return left + weight[..., None] * (right - left)


def interpolate_solution(
    solution: np.ndarray,
    internal_step: float,
    query_times: np.ndarray,
    available_index: int,
) -> np.ndarray:
    """Interpolate only already-computed positive-time solution values."""
    positions = np.maximum(np.asarray(query_times, dtype=np.float64), 0.0) / internal_step
    lower = np.minimum(np.floor(positions + 1e-12).astype(np.int64), available_index)
    upper = np.minimum(lower + 1, available_index)
    weight = positions - lower
    rows = np.arange(solution.shape[0]).reshape((-1,) + (1,) * (positions.ndim - 1))
    return solution[rows, lower, :] + weight[..., None] * (
        solution[rows, upper, :] - solution[rows, lower, :]
    )


def solve_batch(
    histories: np.ndarray,
    parameters: np.ndarray,
    history_grid: np.ndarray,
    output_times: np.ndarray,
    config: SmartGridConfig,
    internal_step: float,
) -> np.ndarray:
    """Solve a batch of constant-delay systems using causal method-of-steps RK4."""
    histories = np.asarray(histories, dtype=np.float64)
    parameters = np.asarray(parameters, dtype=np.float64)
    history_grid = np.asarray(history_grid, dtype=np.float64)
    output_times = np.asarray(output_times, dtype=np.float64)
    batch = histories.shape[0]
    if histories.ndim != 3 or histories.shape[1] != STATE_DIM:
        raise ValueError("histories must have shape [B,8,M]")
    if parameters.shape != (batch, PARAMETER_DIM):
        raise ValueError("parameters must have shape [B,3]")
    if history_grid.shape != (histories.shape[2],):
        raise ValueError("history grid does not match history sensors")
    if not math.isclose(float(history_grid[0]), -config.maximum_history, abs_tol=1e-10):
        raise ValueError("history grid must begin at -maximum_history")
    if not math.isclose(float(history_grid[-1]), 0.0, abs_tol=1e-10):
        raise ValueError("history grid must end at zero")
    if np.any(parameters < config.parameter_lower) or np.any(parameters > config.parameter_upper):
        raise ValueError("parameters fall outside the configured box")
    if output_times.ndim != 1 or output_times.size < 2 or not math.isclose(float(output_times[0]), 0.0):
        raise ValueError("output_times must be one-dimensional and begin at zero")
    if internal_step <= 0.0 or config.fixed_delay <= internal_step:
        raise ValueError("internal_step must be positive and smaller than the delay")
    horizon = float(output_times[-1])
    step_count = int(round(horizon / internal_step))
    if not math.isclose(step_count * internal_step, horizon, abs_tol=1e-10):
        raise ValueError("internal_step must divide the horizon")

    solution = np.empty((batch, step_count + 1, STATE_DIM), dtype=np.float64)
    solution[:, 0, :] = histories[:, :, -1]

    def delayed_at(stage_time: float, available_index: int) -> np.ndarray:
        delayed_time = stage_time - config.fixed_delay
        queries = np.full(batch, delayed_time, dtype=np.float64)
        if delayed_time <= 0.0:
            return interpolate_history(histories, history_grid, queries)
        return interpolate_solution(solution, internal_step, queries, available_index)

    for index in range(step_count):
        time = index * internal_step
        current = solution[:, index, :]
        k1 = rhs_numpy(current, delayed_at(time, index), parameters, config)
        half_time = time + 0.5 * internal_step
        k2 = rhs_numpy(
            current + 0.5 * internal_step * k1,
            delayed_at(half_time, index), parameters, config,
        )
        k3 = rhs_numpy(
            current + 0.5 * internal_step * k2,
            delayed_at(half_time, index), parameters, config,
        )
        full_time = time + internal_step
        k4 = rhs_numpy(
            current + internal_step * k3,
            delayed_at(full_time, index), parameters, config,
        )
        solution[:, index + 1, :] = current + internal_step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        if not np.isfinite(solution[:, index + 1, :]).all():
            raise FloatingPointError(f"non-finite DDE solution at step {index + 1}")

    positions = output_times / internal_step
    lower = np.floor(positions + 1e-12).astype(np.int64)
    upper = np.minimum(lower + 1, step_count)
    weight = positions - lower
    sampled = solution[:, lower, :] + weight[None, :, None] * (
        solution[:, upper, :] - solution[:, lower, :]
    )
    return sampled.astype(np.float32)
