"""Reference and differentiable equations for a delayed-cooling CSTR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch


STATE_DIM = 2
CONDITION_DIM = 4
INVERSE_DIM = 2
STATE_NAMES = ("c", "T")


@dataclass(frozen=True)
class CSTRConfig:
    delay: float
    beta: float
    gamma: float
    condition_bounds: tuple[tuple[float, float], ...]
    condition_names: tuple[str, ...]
    inverse_condition_indices: tuple[int, ...]

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "CSTRConfig":
        result = cls(
            delay=float(values["delay"]),
            beta=float(values["beta"]),
            gamma=float(values["gamma"]),
            condition_bounds=tuple(
                tuple(float(item) for item in pair) for pair in values["condition_bounds"]
            ),
            condition_names=tuple(str(item) for item in values["condition_names"]),
            inverse_condition_indices=tuple(
                int(item) for item in values["inverse_condition_indices"]
            ),
        )
        if len(result.condition_bounds) != CONDITION_DIM:
            raise ValueError("CSTR requires four condition bounds")
        if len(result.condition_names) != CONDITION_DIM:
            raise ValueError("CSTR requires four condition names")
        if result.inverse_condition_indices != (0, 1):
            raise ValueError("the inverse parameters must be k and kappa")
        bounds = np.asarray(result.condition_bounds)
        if np.any(bounds[:, 1] <= bounds[:, 0]):
            raise ValueError("condition upper bounds must exceed lower bounds")
        return result

    @property
    def bounds_array(self) -> np.ndarray:
        return np.asarray(self.condition_bounds, dtype=np.float64)

    @property
    def inverse_spans(self) -> np.ndarray:
        bounds = self.bounds_array[list(self.inverse_condition_indices)]
        return bounds[:, 1] - bounds[:, 0]


def _rhs_numpy(
    current: np.ndarray,
    delayed_temperature: np.ndarray,
    conditions: np.ndarray,
    equation: CSTRConfig,
) -> np.ndarray:
    c = current[:, 0]
    temperature = current[:, 1]
    k, kappa, dilution, coolant = conditions.T
    exponent = equation.gamma * temperature
    if np.any(np.abs(exponent) > 5.0):
        raise FloatingPointError("trajectory left the configured CSTR domain")
    reaction = k * c * np.exp(exponent)
    result = np.empty_like(current)
    result[:, 0] = dilution * (1.0 - c) - reaction
    result[:, 1] = (
        -dilution * temperature
        + equation.beta * reaction
        - kappa * (delayed_temperature - coolant)
    )
    return result


def solve_batch(
    histories: np.ndarray,
    conditions: np.ndarray,
    horizon: float,
    output_points: int,
    step: float,
    equation: CSTRConfig,
    return_internal: bool = False,
) -> np.ndarray:
    """Solve a batch by a fixed-step second-order predictor-corrector method."""
    histories = np.asarray(histories, dtype=np.float64)
    conditions = np.asarray(conditions, dtype=np.float64)
    if histories.ndim != 3 or histories.shape[1] != STATE_DIM:
        raise ValueError("histories must have shape [N,2,M]")
    count, _, sensors = histories.shape
    if conditions.shape != (count, CONDITION_DIM):
        raise ValueError("conditions must have shape [N,4]")
    bounds = equation.bounds_array
    if np.any(conditions < bounds[:, 0] - 1e-10) or np.any(
        conditions > bounds[:, 1] + 1e-10
    ):
        raise ValueError("conditions fall outside configured bounds")
    delay_steps = int(round(equation.delay / step))
    integration_steps = int(round(horizon / step))
    if abs(delay_steps * step - equation.delay) > 1e-10:
        raise ValueError("delay must be an integer multiple of the internal step")
    if sensors != delay_steps + 1:
        raise ValueError("history sensors must equal delay/internal_step + 1")
    if abs(integration_steps * step - horizon) > 1e-10:
        raise ValueError("horizon must be an integer multiple of the internal step")

    full = np.empty((count, delay_steps + integration_steps + 1, STATE_DIM), dtype=np.float64)
    full[:, : delay_steps + 1, :] = np.transpose(histories, (0, 2, 1))
    for local in range(integration_steps):
        index = delay_steps + local
        current = full[:, index, :]
        rhs_1 = _rhs_numpy(
            current, full[:, index - delay_steps, 1], conditions, equation
        )
        predictor = current + step * rhs_1
        rhs_2 = _rhs_numpy(
            predictor, full[:, index - delay_steps + 1, 1], conditions, equation
        )
        full[:, index + 1, :] = current + 0.5 * step * (rhs_1 + rhs_2)
        if not np.isfinite(full[:, index + 1, :]).all():
            raise FloatingPointError("non-finite CSTR reference solution")

    future = full[:, delay_steps:, :]
    if return_internal:
        return future.astype(np.float32)
    positions = np.linspace(0, integration_steps, int(output_points))
    indices = np.rint(positions).astype(np.int64)
    if not np.allclose(positions, indices, atol=1e-10):
        raise ValueError("output grid must align with the internal grid")
    return future[:, indices, :].astype(np.float32)


def constant_histories(count: int, sensors: int) -> np.ndarray:
    result = np.empty((count, STATE_DIM, sensors), dtype=np.float32)
    result[:, 0, :] = 0.8
    result[:, 1, :] = 0.15
    return result


def rhs_torch(
    current: torch.Tensor,
    delayed: torch.Tensor,
    conditions: torch.Tensor,
    equation: CSTRConfig,
) -> torch.Tensor:
    c = current[..., 0]
    temperature = current[..., 1]
    delayed_temperature = delayed[..., 1]
    k = conditions[:, 0].unsqueeze(-1)
    kappa = conditions[:, 1].unsqueeze(-1)
    dilution = conditions[:, 2].unsqueeze(-1)
    coolant = conditions[:, 3].unsqueeze(-1)
    reaction = k * c * torch.exp(equation.gamma * temperature)
    dc = dilution * (1.0 - c) - reaction
    dt = (
        -dilution * temperature
        + equation.beta * reaction
        - kappa * (delayed_temperature - coolant)
    )
    return torch.stack((dc, dt), dim=-1)
