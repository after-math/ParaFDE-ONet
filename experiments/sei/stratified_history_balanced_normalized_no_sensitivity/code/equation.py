"""Three-state delayed SEI equation and causal RK4 reference solver.

The dimensionless delayed epidemic subsystem follows the modelling form used in
"Time-delayed modelling of the COVID-19 dynamics with a convex incidence rate"
(Informatics in Medicine Unlocked, 2022). Recruitment is normalized to ``mu`` so
the disease-free susceptible level is one. The learned parameters are transmission
``b`` and convex-incidence coefficient ``a``; delay and removal rates stay fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

STATE_DIM = 3
STATE_NAMES = ("Susceptible", "Exposed", "Infectious")
PARAMETER_NAMES = ("transmission_b", "convexity_a")
PARAMETER_DIM = 2


@dataclass(frozen=True)
class DelayedSEIConfig:
    """Store one immutable delayed SEI mathematical definition.

    ``transmission_bounds`` and ``convexity_bounds`` define learned ``b`` and ``a``.
    ``delay``, natural loss ``mu``, exposed-to-infectious progression ``sigma`` and
    recovery ``gamma`` are fixed. For example the formal config has bounds
    ``(0.3,0.7)`` and ``(0.5,2.0)``, delay 1, mu 0.02, sigma 0.1 and gamma 0.2.
    ``from_mapping`` constructs this class for data generation and physics loss.
    """

    maximum_history: float
    transmission_bounds: tuple[float, float]
    convexity_bounds: tuple[float, float]
    delay: float
    natural_loss_rate: float
    progression_rate: float
    recovery_rate: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "DelayedSEIConfig":
        """Validate a JSON mapping and return an immutable equation configuration.

        ``values`` contains two learned intervals plus four fixed positive constants.
        Formal input returns ``delay=1`` and the two intervals above. Invalid bounds,
        nonpositive values or insufficient history coverage raise ``ValueError``.
        The input mapping is unchanged. Data generation, solving and training call
        this constructor, so all stages use exactly one mathematical definition.
        """
        transmission = tuple(float(item) for item in values["transmission_bounds"])
        convexity = tuple(float(item) for item in values["convexity_bounds"])
        if len(transmission) != 2 or transmission[0] >= transmission[1]:
            raise ValueError("transmission_bounds must be one increasing pair")
        if len(convexity) != 2 or convexity[0] >= convexity[1]:
            raise ValueError("convexity_bounds must be one increasing pair")
        result = cls(
            maximum_history=float(values["maximum_history"]),
            transmission_bounds=(transmission[0], transmission[1]),
            convexity_bounds=(convexity[0], convexity[1]),
            delay=float(values["delay"]),
            natural_loss_rate=float(values["natural_loss_rate"]),
            progression_rate=float(values["progression_rate"]),
            recovery_rate=float(values["recovery_rate"]),
        )
        positive = (
            result.maximum_history,
            result.delay,
            result.natural_loss_rate,
            result.progression_rate,
            result.recovery_rate,
            *result.transmission_bounds,
            *result.convexity_bounds,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("all equation values must be finite and positive")
        if result.delay > result.maximum_history + 1e-12:
            raise ValueError("maximum_history must cover the fixed delay")
        return result

    @property
    def parameter_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return learned ``(b,a)`` intervals in Branch-input order.

        No input is required. Formal output is ``((0.3,0.7),(0.5,2.0))`` and no
        state changes. Parameter sampling, finite differences and validation use
        this property to avoid independently ordered parameter definitions.
        """
        return self.transmission_bounds, self.convexity_bounds


def rhs_numpy(
    current: np.ndarray,
    delayed: np.ndarray,
    parameters: np.ndarray,
    config: DelayedSEIConfig,
) -> np.ndarray:
    """Evaluate the delayed SEI right-hand side with NumPy broadcasting.

    ``current`` and ``delayed`` end in ``[S,E,I]`` and ``parameters`` ends in
    ``[b,a]``. For inputs ``[8,3]`` and ``[8,2]`` the output is ``[8,3]``. The
    convex delayed incidence is ``b*S_tau*I_tau*(1+a*I_tau)``. Inputs remain
    unchanged; ``solve_batch`` calls this function at every RK4 stage.
    """
    current_values = np.asarray(current, dtype=np.float64)
    delayed_values = np.asarray(delayed, dtype=np.float64)
    physical_parameters = np.asarray(parameters, dtype=np.float64)
    transmission = physical_parameters[..., 0]
    convexity = physical_parameters[..., 1]
    while transmission.ndim < current_values.ndim - 1:
        transmission = np.expand_dims(transmission, axis=-1)
        convexity = np.expand_dims(convexity, axis=-1)
    susceptible = current_values[..., 0]
    exposed = current_values[..., 1]
    infectious = current_values[..., 2]
    delayed_susceptible = delayed_values[..., 0]
    delayed_exposed = delayed_values[..., 1]
    delayed_infectious = delayed_values[..., 2]
    incidence = (
        transmission * delayed_susceptible * delayed_infectious
        * (1.0 + convexity * delayed_infectious)
    )
    return np.stack(
        (
            config.natural_loss_rate * (1.0 - susceptible) - incidence,
            incidence - config.natural_loss_rate * exposed
            - config.progression_rate * delayed_exposed,
            config.progression_rate * delayed_exposed
            - (config.natural_loss_rate + config.recovery_rate) * infectious,
        ),
        axis=-1,
    )


def rhs_torch(
    current: Any, delayed: Any, parameters: Any, config: DelayedSEIConfig
) -> Any:
    """Evaluate the same delayed SEI right-hand side with Torch autograd intact.

    Current and delayed states are normally ``[B,Q,3]`` and parameters ``[B,2]``.
    The return is ``[B,Q,3]`` with gradients to states and both learned parameters.
    No external state changes. ``training.operator_physics_residual`` calls it.
    """
    import torch

    transmission = parameters[..., 0]
    convexity = parameters[..., 1]
    while transmission.ndim < current.ndim - 1:
        transmission = transmission.unsqueeze(-1)
        convexity = convexity.unsqueeze(-1)
    susceptible = current[..., 0]
    exposed = current[..., 1]
    infectious = current[..., 2]
    delayed_susceptible = delayed[..., 0]
    delayed_exposed = delayed[..., 1]
    delayed_infectious = delayed[..., 2]
    incidence = (
        transmission * delayed_susceptible * delayed_infectious
        * (1.0 + convexity * delayed_infectious)
    )
    return torch.stack(
        (
            config.natural_loss_rate * (1.0 - susceptible) - incidence,
            incidence - config.natural_loss_rate * exposed
            - config.progression_rate * delayed_exposed,
            config.progression_rate * delayed_exposed
            - (config.natural_loss_rate + config.recovery_rate) * infectious,
        ),
        dim=-1,
    )


def interpolate_history(
    histories: np.ndarray, history_grid: np.ndarray, query_times: np.ndarray
) -> np.ndarray:
    """Linearly interpolate every sample's three history functions.

    Histories are ``[B,3,M]``, grid is ``[M]`` and queries start with B. The return
    is ``[B,...,3]``; query zero returns the final sensor. With B=8 and four queries
    the shape is ``[8,4,3]``. Queries are clipped and inputs remain unchanged.
    ``solve_batch`` uses this for negative delayed times.
    """
    queries = np.asarray(query_times, dtype=np.float64)
    if queries.shape[0] != histories.shape[0]:
        raise ValueError("query batch must equal history batch")
    clipped = np.clip(queries, float(history_grid[0]), float(history_grid[-1]))
    upper = np.clip(
        np.searchsorted(history_grid, clipped, side="right"), 1, history_grid.size - 1
    )
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
    """Interpolate only already-computed positive-time states causally.

    ``solution`` is ``[B,N,3]`` and queries start with B. The return is
    ``[B,...,3]``; query zero returns the initial state and future rows are never
    read. Inputs remain unchanged. ``solve_batch`` calls it inside its delayed
    state closure; for B=8 and one query the output is ``[8,3]``.
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
    config: DelayedSEIConfig,
    internal_step: float,
) -> np.ndarray:
    """Solve a batch of constant-delay SEI systems with causal RK4.

    Histories are ``[B,3,M]``, physical ``[b,a]`` values are ``[B,2]`` and grids are
    common. The float32 return is ``[B,Q,3]``; eight formal-grid cases yield
    ``[8,401,3]``. Since delay 1 exceeds step 0.01, every delayed RK4 stage uses
    only history or already computed states. Worker-local arrays are the only side
    effect. ``data.solve_parallel`` calls this function.
    """
    histories = np.asarray(histories, dtype=np.float64)
    parameters = np.asarray(parameters, dtype=np.float64)
    history_grid = np.asarray(history_grid, dtype=np.float64)
    output_times = np.asarray(output_times, dtype=np.float64)
    batch = histories.shape[0]
    if histories.ndim != 3 or histories.shape[1] != STATE_DIM:
        raise ValueError("histories must have shape [B,3,M]")
    if parameters.shape != (batch, PARAMETER_DIM):
        raise ValueError("parameters must have shape [B,2]")
    if history_grid.shape != (histories.shape[2],):
        raise ValueError("history grid does not match histories")
    if not math.isclose(
        float(history_grid[0]), -config.maximum_history, abs_tol=1e-10
    ) or not math.isclose(float(history_grid[-1]), 0.0, abs_tol=1e-10):
        raise ValueError("history grid must span [-maximum_history,0]")
    for column, bounds in enumerate(config.parameter_bounds):
        if np.any(parameters[:, column] < bounds[0]) or np.any(
            parameters[:, column] > bounds[1]
        ):
            raise ValueError(f"parameter column {column} falls outside its bounds")
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
    if config.delay <= internal_step:
        raise ValueError("fixed delay must exceed internal_step")

    solution = np.empty((batch, step_count + 1, STATE_DIM), dtype=np.float64)
    solution[:, 0, :] = histories[:, :, -1]

    def delayed_at(stage_time: float, available_index: int) -> np.ndarray:
        """Return all states at one shared delayed RK4 stage time.

        ``stage_time`` is nonnegative and ``available_index`` is the newest solved
        row. The output is ``[B,3]`` with no future reads. ``solve_batch`` invokes
        it four times per RK4 step; it changes no external state.
        """
        query = np.full((batch,), stage_time - config.delay, dtype=np.float64)
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
    sampled = solution[:, lower, :] + weight * (
        solution[:, upper, :] - solution[:, lower, :]
    )
    if not np.isfinite(sampled).all():
        raise FloatingPointError("reference solver generated NaN or Inf")
    if float(sampled.min()) <= 0.0:
        raise FloatingPointError("reference solver generated a nonpositive state")
    if float(np.max(np.sum(sampled, axis=-1))) > 1.00001:
        raise FloatingPointError("reference solver left the normalized SEI simplex")
    return sampled.astype(np.float32)
