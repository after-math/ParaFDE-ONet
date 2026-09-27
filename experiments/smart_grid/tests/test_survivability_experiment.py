from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from equation import SmartGridConfig
from scripts.run_survivability import (
    decode_cartesian_features,
    parameter_grid,
    risk_from_trajectories,
    spearman_correlation,
    survivability_surface,
    wilson_interval,
)


@pytest.fixture(scope="module")
def experiment_config() -> dict:
    return json.loads((PROJECT_DIR / "configs" / "experiment.json").read_text(encoding="utf-8"))


def test_risk_uses_frequency_and_star_edge_differences() -> None:
    values = np.zeros((2, 3, 8), dtype=np.float64)
    values[0, 1, 4] = -1.25
    values[0, 2, 2] = 0.6
    values[1, 1, 7] = 0.5
    values[1, 2, 0] = -0.2
    frequency, angle = risk_from_trajectories(values, ((0, 1), (0, 2), (0, 3)))
    np.testing.assert_allclose(frequency, [1.25, 0.5])
    np.testing.assert_allclose(angle, [0.6, 0.2])


def test_survivability_surface_has_parameterwise_probabilities() -> None:
    frequency = np.asarray([[0.5, 1.0], [1.5, 0.5]])
    angle = np.asarray([[0.2, 0.6], [0.4, 0.2]])
    surface = survivability_surface(
        frequency,
        angle,
        np.asarray([0.75, 2.0]),
        np.asarray([0.3, 0.5]),
    )
    assert surface.shape == (2, 2, 2)
    np.testing.assert_allclose(surface[0], [[0.5, 0.5], [0.5, 1.0]])
    np.testing.assert_allclose(surface[1], [[0.5, 0.5], [0.5, 0.5]])


def test_parameter_grid_is_full_factorial(experiment_config: dict) -> None:
    equation = SmartGridConfig.from_mapping(experiment_config["equation"])
    values = parameter_grid(equation, 4)
    assert values.shape == (64, 3)
    np.testing.assert_allclose(values.min(axis=0), equation.parameter_lower)
    np.testing.assert_allclose(values.max(axis=0), equation.parameter_upper)
    assert all(np.unique(values[:, column]).size == 4 for column in range(3))


def test_wilson_and_spearman_handle_probability_and_ties() -> None:
    lower, upper = wilson_interval(np.asarray([0.0, 0.5, 1.0]), 512)
    assert np.all(lower >= 0.0)
    assert np.all(upper <= 1.0)
    assert lower[0] == pytest.approx(0.0)
    assert upper[-1] == pytest.approx(1.0)
    assert spearman_correlation(np.asarray([1.0, 2.0, 2.0, 4.0]), np.asarray([2.0, 3.0, 3.0, 8.0])) == pytest.approx(1.0)


def test_cached_cartesian_decoder_matches_standard_forward(experiment_config: dict) -> None:
    torch = pytest.importorskip("torch")
    from model import build_operator

    operator_config = copy.deepcopy(experiment_config["operator"])
    operator_config.update(
        {
            "latent_dim": 8,
            "history_width": 12,
            "history_depth": 1,
            "parameter_width": 12,
            "parameter_depth": 1,
            "trunk_width": 12,
            "trunk_depth": 1,
            "fourier_modes": 2,
        }
    )
    data_config = copy.deepcopy(experiment_config["data"])
    data_config.update({"history_sensors": 15, "horizon": 3.5})
    model = build_operator(operator_config, data_config, [0.0] * 8, [1.0] * 8, 17).eval()
    histories = torch.randn(2, 8, 15)
    parameters = torch.tensor(
        [[6.0, 0.18, 0.07], [8.0, 0.25, 0.10], [10.0, 0.32, 0.13]]
    )
    times = torch.linspace(0.0, 3.5, 9)
    history_features = model.encode_histories(histories)
    parameter_features = model.encode_parameters(parameters)
    trunk_features = model.encode_times(times, 1)[0]
    cached = decode_cartesian_features(
        model, history_features, parameter_features, trunk_features
    )
    history_expanded = histories[:, None].expand(-1, 3, -1, -1).reshape(6, 8, 15)
    parameter_expanded = parameters[None].expand(2, -1, -1).reshape(6, 3)
    standard = model(history_expanded, parameter_expanded, times).reshape(2, 3, 9, 8)
    torch.testing.assert_close(cached, standard, atol=2e-6, rtol=2e-6)
