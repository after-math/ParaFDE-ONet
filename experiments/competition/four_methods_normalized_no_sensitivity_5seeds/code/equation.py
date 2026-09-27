"""Time-varying-delay two-species competition equation and reference solver.

The learned parameters are ``(r1,r2)``.  All carrying capacities, competition
coefficients and the sinusoidal delay law are fixed by one immutable configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np


STATE_DIM = 2
STATE_NAMES = ("Species 1", "Species 2")
PARAMETER_NAMES = ("r1", "r2")


@dataclass(frozen=True)
class CompetitionConfig:
    """Store the complete mathematical definition of the variable-delay DDE.

    ``maximum_history`` defines the supplied interval ``[-d_bar,0]``;
    ``parameter_bounds`` contains the two growth-rate boxes; the remaining fields
    are ``(K1,K2)``, ``(a12,a21)``, and ``(d0,epsilon,omega)``.  The principal
    constructor is ``from_mapping``.  For example, the formal config produces
    ``delay(0)=0.8`` and ``delay range=[0.6,1.0]``.  Data generation and physics
    residuals receive the same instance, so the class has no mutable side effect.
    """

    maximum_history: float
    parameter_bounds: tuple[tuple[float, float], tuple[float, float]]
    carrying_capacities: tuple[float, float]
    competition_coefficients: tuple[float, float]
    delay_mean: float
    delay_amplitude: float
    delay_angular_frequency: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "CompetitionConfig":
        """Construct and validate the equation from the resolved JSON mapping.

        ``values`` must contain two increasing parameter intervals and all fixed
        coefficients.  The returned immutable config is used by ``solve_batch`` and
        ``training.operator_physics_residual``.  For the formal mapping,
        ``result.parameter_bounds[0] == (0.8,1.4)``.  Invalid positivity, delay range,
        monotone-retarded-time, or weak-competition conditions raise ``ValueError``;
        the input mapping is never changed.
        """
        bounds = tuple(tuple(float(item) for item in pair) for pair in values["parameter_bounds"])
        capacities = tuple(float(item) for item in values["carrying_capacities"])
        competition = tuple(float(item) for item in values["competition_coefficients"])
        if len(bounds) != 2 or any(len(pair) != 2 or pair[0] >= pair[1] for pair in bounds):
            raise ValueError("parameter_bounds must contain two increasing intervals")
        if len(capacities) != 2 or len(competition) != 2:
            raise ValueError("capacities and competition coefficients must each contain two values")
        result = cls(
            maximum_history=float(values["maximum_history"]),
            parameter_bounds=bounds,  # type: ignore[arg-type]
            carrying_capacities=capacities,  # type: ignore[arg-type]
            competition_coefficients=competition,  # type: ignore[arg-type]
            delay_mean=float(values["delay_mean"]),
            delay_amplitude=float(values["delay_amplitude"]),
            delay_angular_frequency=float(values["delay_angular_frequency"]),
        )
        positives = (
            result.maximum_history, *result.carrying_capacities,
            *result.competition_coefficients, result.delay_mean,
            result.delay_angular_frequency,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positives):
            raise ValueError("all fixed scales and coefficients must be finite and positive")
        if not 0.0 <= result.delay_amplitude < result.delay_mean:
            raise ValueError("delay amplitude must lie in [0,delay_mean)")
        if result.delay_mean + result.delay_amplitude > result.maximum_history + 1e-12:
            raise ValueError("maximum_history does not cover the largest delay")
        if result.delay_amplitude * result.delay_angular_frequency >= 1.0:
            raise ValueError("epsilon*omega must be below one so t-d(t) advances monotonically")
        r1_min, r2_min = bounds[0][0], bounds[1][0]
        k1, k2 = capacities
        a12, a21 = competition
        if a12 * k2 >= r1_min or a21 * k1 >= r2_min:
            raise ValueError("the parameter box must remain in the weak-competition regime")
        return result


def delay_numpy(times: np.ndarray | float, config: CompetitionConfig) -> np.ndarray:
    """Evaluate ``d(t)=d0+epsilon*sin(omega*t)`` with NumPy.

    ``times`` may be a scalar or any array; the return has the same array shape and
    float64 dtype.  For example ``delay_numpy(np.array([0]),cfg)[0]`` is ``0.8``.
    The function has no side effect and is called at every RK4 stage by
    ``solve_batch``.
    """
    values = np.asarray(times, dtype=np.float64)
    return config.delay_mean + config.delay_amplitude * np.sin(
        config.delay_angular_frequency * values
    )


def delay_torch(times: Any, config: CompetitionConfig) -> Any:
    """Evaluate the same delay law while retaining PyTorch autograd.

    ``times`` is normally ``[B,Q]`` and the returned tensor has identical shape,
    dtype and device.  For example zero input returns a tensor filled with ``0.8``.
    The only side effect is its autograd graph; ``operator_physics_residual`` calls
    it when constructing delayed query times.
    """
    import torch

    return config.delay_mean + config.delay_amplitude * torch.sin(
        config.delay_angular_frequency * times
    )


def rhs_numpy(
    current: np.ndarray,
    delayed: np.ndarray,
    parameters: np.ndarray,
    config: CompetitionConfig,
) -> np.ndarray:
    """Evaluate the batched two-species competition right-hand side.

    ``current`` and ``delayed`` end in state dimension two; ``parameters`` ends in
    ``(r1,r2)``.  The returned derivative broadcasts to ``current.shape``.  For
    example inputs ``[64,2]`` return ``[64,2]``.  It implements exactly the two
    displayed DDE equations and is called by all four RK4 stages without modifying
    its inputs.
    """
    current = np.asarray(current)
    delayed = np.asarray(delayed)
    parameters = np.asarray(parameters)
    x1, x2 = np.moveaxis(current, -1, 0)
    delayed_x1, delayed_x2 = np.moveaxis(delayed, -1, 0)
    r1, r2 = np.moveaxis(parameters, -1, 0)
    k1, k2 = config.carrying_capacities
    a12, a21 = config.competition_coefficients
    derivative_1 = x1 * (r1 * (1.0 - x1 / k1) - a12 * delayed_x2)
    derivative_2 = x2 * (r2 * (1.0 - x2 / k2) - a21 * delayed_x1)
    return np.stack((derivative_1, derivative_2), axis=-1)


def rhs_torch(current: Any, delayed: Any, parameters: Any, config: CompetitionConfig) -> Any:
    """Evaluate the DDE right-hand side with differentiable PyTorch operations.

    Current and delayed predictions have shape ``[B,Q,2]``; parameters are ``[B,2]``
    and are expanded over Q.  The return is ``[B,Q,2]`` and preserves gradients to
    model weights, times and inputs.  For example ``B=4,Q=8`` returns ``[4,8,2]``.
    ``training.operator_physics_residual`` is the sole caller.
    """
    import torch

    while parameters.ndim < current.ndim:
        parameters = parameters.unsqueeze(-2)
    x1, x2 = current.unbind(dim=-1)
    delayed_x1, delayed_x2 = delayed.unbind(dim=-1)
    r1, r2 = parameters.unbind(dim=-1)
    k1, k2 = config.carrying_capacities
    a12, a21 = config.competition_coefficients
    derivative_1 = x1 * (r1 * (1.0 - x1 / k1) - a12 * delayed_x2)
    derivative_2 = x2 * (r2 * (1.0 - x2 / k2) - a21 * delayed_x1)
    return torch.stack((derivative_1, derivative_2), dim=-1)


def interpolate_history(
    histories: np.ndarray, history_grid: np.ndarray, query_times: np.ndarray
) -> np.ndarray:
    """Linearly interpolate each sample's two history functions.

    Histories have shape ``[B,2,M]``, the grid is ``[M]`` and queries are
    ``[B,...]``; output is ``[B,...,2]``.  Query zero returns the last history value.
    For example ``B=8`` and four queries produce ``[8,4,2]``.  Queries are clipped
    to the supplied interval.  ``solve_batch`` calls this for nonpositive delayed
    times and no input is modified.
    """
    queries = np.asarray(query_times, dtype=np.float64)
    if queries.shape[0] != histories.shape[0]:
        raise ValueError("query batch must equal history batch")
    clipped = np.clip(queries, float(history_grid[0]), float(history_grid[-1]))
    upper = np.clip(np.searchsorted(history_grid, clipped, side="right"), 1, history_grid.size - 1)
    lower = upper - 1
    weight = (clipped - history_grid[lower]) / np.maximum(
        history_grid[upper] - history_grid[lower], 1e-15
    )
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
    """Causally interpolate only already-computed nonnegative solution values.

    ``solution`` is ``[B,N,2]``, queries are ``[B,...]`` and output is
    ``[B,...,2]``.  ``available_index`` prevents reading future uninitialized memory;
    for example query zero returns ``solution[:,0]``.  The helper is called by
    ``solve_batch`` and has no side effect.
    """
    positions = np.maximum(np.asarray(query_times), 0.0) / internal_step
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
    config: CompetitionConfig,
    internal_step: float,
) -> np.ndarray:
    """Solve a batch of variable-delay DDEs by explicit causal RK4.

    Inputs are histories ``[B,2,M]``, growth rates ``[B,2]``, common history/output
    grids, the equation and internal step.  The return is float32
    ``[B,len(output_times),2]``; e.g. 64 formal cases return ``[64,401,2]``.  Since
    the minimum delay is 0.6 and step is 0.01, every RK4 delayed query lies in
    history or already-computed solution.  The function allocates only its batch
    workspace and is called by ``data.solve_parallel`` workers.
    """
    histories = np.asarray(histories, dtype=np.float64)
    parameters = np.asarray(parameters, dtype=np.float64)
    history_grid = np.asarray(history_grid, dtype=np.float64)
    output_times = np.asarray(output_times, dtype=np.float64)
    if histories.ndim != 3 or histories.shape[1] != STATE_DIM:
        raise ValueError("histories must have shape [B,2,M]")
    if parameters.shape != (histories.shape[0], 2):
        raise ValueError("parameters must have shape [B,2]")
    if history_grid.shape != (histories.shape[2],):
        raise ValueError("history grid does not match histories")
    if not math.isclose(float(history_grid[0]), -config.maximum_history, abs_tol=1e-10):
        raise ValueError("history grid must start at -maximum_history")
    if not math.isclose(float(history_grid[-1]), 0.0, abs_tol=1e-10):
        raise ValueError("history grid must end at zero")
    bounds = np.asarray(config.parameter_bounds)
    if np.any(parameters < bounds[:, 0]) or np.any(parameters > bounds[:, 1]):
        raise ValueError("parameters fall outside configured bounds")
    if output_times.ndim != 1 or output_times.size < 2 or not math.isclose(float(output_times[0]), 0.0):
        raise ValueError("output_times must be one-dimensional and start at zero")
    horizon = float(output_times[-1])
    step_count = int(round(horizon / internal_step))
    if internal_step <= 0.0 or not math.isclose(step_count * internal_step, horizon, abs_tol=1e-10):
        raise ValueError("internal_step must divide the horizon")
    minimum_delay = config.delay_mean - config.delay_amplitude
    if minimum_delay <= internal_step:
        raise ValueError("minimum delay must exceed internal_step for explicit causal RK4")

    batch = histories.shape[0]
    solution = np.empty((batch, step_count + 1, STATE_DIM), dtype=np.float64)
    solution[:, 0, :] = histories[:, :, -1]

    def delayed_at(stage_time: float, available_index: int) -> np.ndarray:
        """Return both states at the stage-specific delayed time for all samples."""
        query = np.full((batch,), stage_time - float(delay_numpy(stage_time, config)))
        historical = interpolate_history(histories, history_grid, query)
        computed = interpolate_solution(solution, internal_step, query, available_index)
        return np.where((query <= 0.0)[:, None], historical, computed)

    for index in range(step_count):
        time_value = index * internal_step
        state = solution[:, index, :]
        k1 = rhs_numpy(state, delayed_at(time_value, index), parameters, config)
        half_time = time_value + 0.5 * internal_step
        k2 = rhs_numpy(
            state + 0.5 * internal_step * k1,
            delayed_at(half_time, index), parameters, config,
        )
        k3 = rhs_numpy(
            state + 0.5 * internal_step * k2,
            delayed_at(half_time, index), parameters, config,
        )
        end_time = time_value + internal_step
        k4 = rhs_numpy(
            state + internal_step * k3,
            delayed_at(end_time, index), parameters, config,
        )
        solution[:, index + 1, :] = state + internal_step * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        ) / 6.0

    positions = output_times / internal_step
    lower = np.floor(positions + 1e-12).astype(np.int64)
    upper = np.minimum(lower + 1, step_count)
    weight = (positions - lower).reshape(1, -1, 1)
    sampled = solution[:, lower, :] + weight * (solution[:, upper, :] - solution[:, lower, :])
    if not np.isfinite(sampled).all():
        raise FloatingPointError("reference solver generated NaN or Inf")
    if float(sampled.min()) < -1e-6:
        raise FloatingPointError("reference solver generated a negative population")
    return sampled.astype(np.float32)
