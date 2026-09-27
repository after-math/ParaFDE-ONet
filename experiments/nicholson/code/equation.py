"""Four-patch Nicholson blowflies equation and causal reference solver."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

STATE_DIM = 4
STATE_NAMES = ("Patch 1", "Patch 2", "Patch 3", "Patch 4")
PARAMETER_NAMES = ("beta", "tau0")


@dataclass(frozen=True)
class NicholsonConfig:
    """Store the complete mathematical definition of the four-patch DDE.

    Learned parameters are maximum recruitment beta and mean delay tau0. Fixed
    values are death, nearest-neighbour migration, density suppression and the
    sinusoidal delay modulation. from_mapping is the normal constructor. For the
    formal mapping, beta lies in [3,7], tau0 in [0.8,1.4], and the largest delay
    equals the supplied 1.6 history length. The object is immutable.
    """

    maximum_history: float
    parameter_bounds: tuple[tuple[float, float], tuple[float, float]]
    death_rate: float
    migration_rate: float
    density_suppression: float
    delay_amplitude: float
    delay_angular_frequency: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "NicholsonConfig":
        """Construct and validate an equation from a resolved JSON mapping.

        values contains two parameter boxes and all fixed constants. The returned
        immutable config is used by solve_batch and the physics residual. For the
        formal example, its maximum delay is 1.6. Invalid positivity, history
        coverage, or retarded-time monotonicity raises ValueError. No input changes.
        """
        bounds = tuple(tuple(float(item) for item in pair) for pair in values["parameter_bounds"])
        if len(bounds) != 2 or any(len(pair) != 2 or pair[0] >= pair[1] for pair in bounds):
            raise ValueError("parameter_bounds must contain increasing beta and tau0 intervals")
        result = cls(
            maximum_history=float(values["maximum_history"]),
            parameter_bounds=bounds,  # type: ignore[arg-type]
            death_rate=float(values["death_rate"]),
            migration_rate=float(values["migration_rate"]),
            density_suppression=float(values["density_suppression"]),
            delay_amplitude=float(values["delay_amplitude"]),
            delay_angular_frequency=float(values["delay_angular_frequency"]),
        )
        positive = (
            result.maximum_history, result.death_rate, result.migration_rate,
            result.density_suppression, result.delay_angular_frequency,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("fixed scales except delay amplitude must be finite and positive")
        if result.delay_amplitude < 0.0:
            raise ValueError("delay amplitude cannot be negative")
        beta_bounds, tau_bounds = result.parameter_bounds
        if beta_bounds[0] <= result.death_rate:
            raise ValueError("beta must exceed death rate so a positive equilibrium exists")
        if tau_bounds[0] <= result.delay_amplitude:
            raise ValueError("the minimum variable delay must stay positive")
        if tau_bounds[1] + result.delay_amplitude > result.maximum_history + 1e-12:
            raise ValueError("maximum_history does not cover every possible delay")
        if result.delay_amplitude * result.delay_angular_frequency >= 1.0:
            raise ValueError("epsilon times omega must be below one")
        return result


def delay_numpy(
    times: np.ndarray | float, parameters: np.ndarray, config: NicholsonConfig
) -> np.ndarray:
    """Evaluate tau(t)=tau0+epsilon*sin(omega*t) for a NumPy batch.

    times is scalar or batch-shaped and parameters ends in (beta,tau0). The
    float64 return broadcasts to the time shape. For example tau0=0.8 gives range
    [0.6,1.0]. The function has no side effect and every RK4 stage calls it.
    """
    time_values = np.asarray(times, dtype=np.float64)
    parameter_values = np.asarray(parameters, dtype=np.float64)
    tau0 = parameter_values[..., 1]
    while tau0.ndim < time_values.ndim:
        tau0 = np.expand_dims(tau0, axis=-1)
    return tau0 + config.delay_amplitude * np.sin(
        config.delay_angular_frequency * time_values
    )


def delay_torch(times: Any, parameters: Any, config: NicholsonConfig) -> Any:
    """Evaluate the same delay while retaining Torch autograd.

    times is normally [B,Q] and parameters is [B,2]. The return is [B,Q] on the
    same device with gradients to time and tau0; zero time returns tau0. The
    operator physics residual is the caller and no external state changes.
    """
    import torch

    tau0 = parameters[..., 1]
    while tau0.ndim < times.ndim:
        tau0 = tau0.unsqueeze(-1)
    return tau0 + config.delay_amplitude * torch.sin(
        config.delay_angular_frequency * times
    )


def rhs_numpy(
    current: np.ndarray,
    delayed: np.ndarray,
    parameters: np.ndarray,
    config: NicholsonConfig,
) -> np.ndarray:
    """Evaluate mortality, ring migration and Nicholson recruitment with NumPy.

    current and delayed end in four states and parameters ends in (beta,tau0).
    The return broadcasts to current.shape; for example [8,4] returns [8,4].
    The exact documented equation is used by all four RK stages. Inputs remain
    unchanged and there is no external side effect.
    """
    current_values = np.asarray(current)
    delayed_values = np.asarray(delayed)
    beta = np.asarray(parameters)[..., 0]
    while beta.ndim < current_values.ndim:
        beta = np.expand_dims(beta, axis=-1)
    incoming = np.roll(current_values, 1, axis=-1) + np.roll(current_values, -1, axis=-1)
    recruitment = beta * delayed_values * np.exp(
        -config.density_suppression * delayed_values
    )
    return (
        -(config.death_rate + 2.0 * config.migration_rate) * current_values
        + config.migration_rate * incoming
        + recruitment
    )


def rhs_torch(current: Any, delayed: Any, parameters: Any, config: NicholsonConfig) -> Any:
    """Evaluate the four-patch right-hand side with differentiable Torch ops.

    Current and delayed states are [B,Q,4] and parameters is [B,2]. The return is
    [B,Q,4]; for example [49,48,4] remains [49,48,4]. Gradients flow to all
    model-dependent inputs. The operator physics residual is the sole caller.
    """
    import torch

    beta = parameters[..., 0]
    while beta.ndim < current.ndim - 1:
        beta = beta.unsqueeze(-1)
    beta = beta.unsqueeze(-1)
    incoming = torch.roll(current, 1, dims=-1) + torch.roll(current, -1, dims=-1)
    recruitment = beta * delayed * torch.exp(-config.density_suppression * delayed)
    return (
        -(config.death_rate + 2.0 * config.migration_rate) * current
        + config.migration_rate * incoming
        + recruitment
    )


def interpolate_history(
    histories: np.ndarray, history_grid: np.ndarray, query_times: np.ndarray
) -> np.ndarray:
    """Linearly interpolate every sample's four history functions.

    Histories is [B,4,M], grid is [M], queries is [B,...], and return is
    [B,...,4]. Query zero returns the final history value. For example B=8 with
    four queries returns [8,4,4]. Queries are clipped to the supplied interval.
    solve_batch calls this helper and no input changes.
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
    """Causally interpolate only already-computed solution states.

    solution is [B,N,4], queries is [B,...], and return is [B,...,4].
    available_index prevents future-memory reads; query zero returns the initial
    state. The RK4 solver calls it for positive delay queries. No input changes.
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
    config: NicholsonConfig,
    internal_step: float,
) -> np.ndarray:
    """Solve a batch of parameter-dependent variable-delay equations by RK4.

    Inputs are histories [B,4,M], parameters [B,2], common grids, config and step.
    The float32 return is [B,Q,4]; a formal chunk returns [8,401,4]. Because the
    minimum delay exceeds 0.01, delayed RK stages query history or computed states
    only. The function allocates batch-local memory and data workers call it.
    """
    histories = np.asarray(histories, dtype=np.float64)
    parameters = np.asarray(parameters, dtype=np.float64)
    history_grid = np.asarray(history_grid, dtype=np.float64)
    output_times = np.asarray(output_times, dtype=np.float64)
    batch = histories.shape[0]
    if histories.ndim != 3 or histories.shape[1] != STATE_DIM:
        raise ValueError("histories must have shape [B,4,M]")
    if parameters.shape != (batch, 2):
        raise ValueError("parameters must have shape [B,2]")
    if history_grid.shape != (histories.shape[2],):
        raise ValueError("history grid does not match histories")
    if not math.isclose(float(history_grid[0]), -config.maximum_history, abs_tol=1e-10):
        raise ValueError("history grid must start at -maximum_history")
    if not math.isclose(float(history_grid[-1]), 0.0, abs_tol=1e-10):
        raise ValueError("history grid must end at zero")
    bounds = np.asarray(config.parameter_bounds)
    if np.any(parameters < bounds[:, 0]) or np.any(parameters > bounds[:, 1]):
        raise ValueError("parameters fall outside beta/tau0 bounds")
    if output_times.ndim != 1 or output_times.size < 2 or not math.isclose(
        float(output_times[0]), 0.0
    ):
        raise ValueError("output_times must start at zero")
    horizon = float(output_times[-1])
    step_count = int(round(horizon / internal_step))
    if internal_step <= 0.0 or not math.isclose(
        step_count * internal_step, horizon, abs_tol=1e-10
    ):
        raise ValueError("internal_step must divide the horizon")
    if config.parameter_bounds[1][0] - config.delay_amplitude <= internal_step:
        raise ValueError("minimum delay must exceed internal_step")

    solution = np.empty((batch, step_count + 1, STATE_DIM), dtype=np.float64)
    solution[:, 0, :] = histories[:, :, -1]

    def delayed_at(stage_time: float, available_index: int) -> np.ndarray:
        """Return four states at sample-specific delayed RK stage times."""
        time_batch = np.full((batch,), stage_time, dtype=np.float64)
        query = time_batch - delay_numpy(time_batch, parameters, config)
        historical = interpolate_history(histories, history_grid, query)
        computed = interpolate_solution(solution, internal_step, query, available_index)
        return np.where((query <= 0.0)[:, None], historical, computed)

    for index in range(step_count):
        time_value = index * internal_step
        state = solution[:, index, :]
        k1 = rhs_numpy(state, delayed_at(time_value, index), parameters, config)
        half_time = time_value + 0.5 * internal_step
        k2 = rhs_numpy(state + 0.5 * internal_step * k1, delayed_at(half_time, index), parameters, config)
        k3 = rhs_numpy(state + 0.5 * internal_step * k2, delayed_at(half_time, index), parameters, config)
        end_time = time_value + internal_step
        k4 = rhs_numpy(state + internal_step * k3, delayed_at(end_time, index), parameters, config)
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
    if float(sampled.min()) <= 0.0:
        raise FloatingPointError("reference solver generated a nonpositive population")
    return sampled.astype(np.float32)

