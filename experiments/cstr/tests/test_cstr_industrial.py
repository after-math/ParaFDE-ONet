from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cstr_forward.equation import CSTRConfig, constant_histories, solve_batch
from cstr_industrial.benchmark import inverse_direct_de, sample_parameter_events


ROOT = Path(__file__).resolve().parents[1]


def configurations() -> tuple[dict, dict, CSTRConfig]:
    experiment = json.loads((ROOT / "configs" / "experiment.json").read_text())
    industrial = json.loads((ROOT / "configs" / "industrial.json").read_text())
    return experiment, industrial, CSTRConfig.from_mapping(experiment["equation"])


def test_event_sampling_is_fixed_bounded_and_nontrivial() -> None:
    _, industrial, equation = configurations()
    old = np.asarray(industrial["old_parameters"], dtype=np.float64)
    first = sample_parameter_events(equation, old, 20, 1234, 0.15)
    second = sample_parameter_events(equation, old, 20, 1234, 0.15)
    bounds = equation.bounds_array[:2]
    jumps = np.linalg.norm((first - old) / (bounds[:, 1] - bounds[:, 0]), axis=1)
    assert first.shape == (20, 2)
    assert np.array_equal(first, second)
    assert np.all(first >= bounds[:, 0])
    assert np.all(first <= bounds[:, 1])
    assert np.all(jumps >= 0.15)


def test_small_direct_de_uses_two_parameter_inverse_only() -> None:
    _, industrial, equation = configurations()
    small = dict(industrial)
    small.update(
        {
            "observation_horizon": 1.0,
            "observation_points": 21,
            "de_population": 6,
            "de_generations": 2,
            "de_maximum_candidate_evaluations": 18,
        }
    )
    history = constant_histories(1, 101)
    truth = np.asarray([1.05, 1.15], dtype=np.float64)
    control = np.asarray([1.2, 0.0], dtype=np.float64)
    conditions = np.asarray([[*truth, *control]], dtype=np.float64)
    observations = solve_batch(history, conditions, 1.0, 21, 0.01, equation)[0]
    estimate, record = inverse_direct_de(
        history,
        control,
        observations,
        small,
        equation,
        np.ones(2),
        seed=987,
    )
    assert estimate.shape == (2,)
    assert np.all(estimate >= equation.bounds_array[:2, 0])
    assert np.all(estimate <= equation.bounds_array[:2, 1])
    assert record["solver_batch_calls"] == 3
    assert record["direct_trajectory_solves"] == 18
