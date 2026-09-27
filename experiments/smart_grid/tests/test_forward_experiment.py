from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from data import dataset_signature, finite_difference_sensitivities, sample_histories
from equation import SmartGridConfig, nominal_equilibrium_angles, rhs_numpy, solve_batch
from scripts.pipeline import select_training_seeds


@pytest.fixture(scope="module")
def config() -> dict:
    return json.loads((PROJECT_DIR / "configs" / "experiment.json").read_text(encoding="utf-8"))


def test_configuration_is_single_model_and_has_expected_dimensions(config: dict) -> None:
    assert config["methods"] == ["parafdeonet"]
    assert len(config["equation"]["state_order"]) == 8
    assert len(config["equation"]["parameter_order"]) == 3
    assert config["operator"]["history_branch_count"] == 8
    assert config["operator"]["parameter_feature_mode"] == "shared_latent_dim"


def test_training_seed_subset_selection(config: dict) -> None:
    configured = config["randomness"]["training_seeds"]
    selected = select_training_seeds(
        configured, None, "20261002,20261003,20261004,20261005"
    )
    assert selected == [20261002, 20261003, 20261004, 20261005]
    with pytest.raises(ValueError, match="mutually exclusive"):
        select_training_seeds(configured, 20261001, "20261002,20261003")
    with pytest.raises(ValueError, match="absent"):
        select_training_seeds(configured, None, "20261002,99")


def test_nominal_state_is_an_equilibrium(config: dict) -> None:
    equation = SmartGridConfig.from_mapping(config["equation"])
    angles = nominal_equilibrium_angles(equation.nominal_parameters[0])
    state = np.concatenate((angles, np.zeros(4)))
    derivative = rhs_numpy(
        state[None, :], state[None, :], np.asarray(equation.nominal_parameters)[None, :], equation
    )
    np.testing.assert_allclose(derivative, 0.0, atol=1e-12)


def test_sampled_histories_are_kinematically_compatible(config: dict) -> None:
    equation = SmartGridConfig.from_mapping(config["equation"])
    times = np.linspace(-equation.maximum_history, 0.0, 1201)
    histories = sample_histories(
        4,
        times,
        config["data"]["history_generator"],
        equation,
        np.random.default_rng(91),
    ).astype(np.float64)
    numerical_derivative = np.gradient(histories[:, :4, :], times, axis=-1, edge_order=2)
    np.testing.assert_allclose(numerical_derivative, histories[:, 4:, :], atol=5e-5, rtol=5e-5)
    np.testing.assert_allclose(histories[:, 0, -1], 0.0, atol=1e-7)
    for left in range(1, 4):
        for right in range(left + 1, 4):
            assert np.all(np.linalg.norm(histories[:, left] - histories[:, right], axis=1) > 1e-4)


def test_solver_is_finite_and_crosses_the_delay(config: dict) -> None:
    equation = SmartGridConfig.from_mapping(config["equation"])
    history_times = np.linspace(-equation.maximum_history, 0.0, 121)
    histories = sample_histories(
        2,
        history_times,
        config["data"]["history_generator"],
        equation,
        np.random.default_rng(101),
    )
    parameters = np.asarray(
        [equation.nominal_parameters, (7.2, 0.29, 0.085)], dtype=np.float32
    )
    output_times = np.linspace(0.0, 3.2, 65)
    coarse = solve_batch(histories, parameters, history_times, output_times, equation, 0.01)
    fine = solve_batch(histories, parameters, history_times, output_times, equation, 0.005)
    assert coarse.shape == (2, 65, 8)
    assert np.isfinite(coarse).all()
    relative = np.linalg.norm((coarse - fine).reshape(2, -1), axis=1) / np.maximum(
        np.linalg.norm(fine.reshape(2, -1), axis=1), 1e-12
    )
    assert float(np.max(relative)) < 5e-4


def test_finite_difference_sensitivity_shape_and_scaling(config: dict) -> None:
    equation = SmartGridConfig.from_mapping(config["equation"])
    history_times = np.linspace(-equation.maximum_history, 0.0, 31)
    output_times = np.linspace(0.0, 0.1, 6)
    histories = sample_histories(
        2,
        history_times,
        config["data"]["history_generator"],
        equation,
        np.random.default_rng(201),
    )
    parameters = np.asarray(
        [equation.nominal_parameters, (6.01, 0.3199, 0.0701)], dtype=np.float32
    )
    base = solve_batch(histories, parameters, history_times, output_times, equation, 0.01)
    sensitivities = finite_difference_sensitivities(
        histories,
        parameters,
        base,
        history_times,
        output_times,
        equation,
        0.01,
        np.asarray(config["data"]["sensitivity_finite_difference_steps"]),
        1,
        2,
    )
    assert sensitivities.shape == (2, 6, 8, 3)
    assert np.isfinite(sensitivities).all()
    np.testing.assert_allclose(sensitivities[:, 0], 0.0, atol=2e-5)


def test_dataset_signature_includes_data_seed(config: dict) -> None:
    changed = copy.deepcopy(config)
    changed["randomness"]["data_seed"] += 1
    assert dataset_signature(config) != dataset_signature(changed)


def test_formal_parameter_count(config: dict) -> None:
    data = config["data"]
    operator = config["operator"]

    def residual_mlp_count(input_dim: int, output_dim: int, width: int, depth: int) -> int:
        return (
            input_dim * width + width
            + depth * 2 * (width * width + width)
            + width * output_dim + output_dim
        )

    latent = int(operator["latent_dim"])
    history_count = 8 * residual_mlp_count(
        int(data["history_sensors"]),
        8 * latent,
        int(operator["history_width"]),
        int(operator["history_depth"]),
    )
    parameter_count = residual_mlp_count(
        3, latent, int(operator["parameter_width"]), int(operator["parameter_depth"])
    )
    trunk_count = residual_mlp_count(
        1 + 2 * int(operator["fourier_modes"]),
        latent,
        int(operator["trunk_width"]),
        int(operator["trunk_depth"]),
    )
    total = history_count + parameter_count + trunk_count + 8
    assert total == 75_223_048
    assert total == config["expected_formal_parameter_counts"]["parafdeonet"]


def test_tiny_operator_forward_jvp_and_physics_residual(config: dict) -> None:
    torch = pytest.importorskip("torch")
    from model import build_operator
    from training import (
        learning_rate_at,
        normalized_model_parameter_sensitivities,
        operator_physics_residual,
    )
    from scripts.pipeline import resume_science_compatible

    assert learning_rate_at(59_999, config["operator"]) > 1e-6
    assert learning_rate_at(60_000, config["operator"]) == pytest.approx(1e-6)
    assert learning_rate_at(100_000, config["operator"]) == pytest.approx(1e-6)
    previous_config = copy.deepcopy(config)
    previous_config["operator"]["iterations"] = 60_000
    previous_config["operator"].pop("learning_rate_decay_iterations")
    assert resume_science_compatible(previous_config, config)
    incompatible = copy.deepcopy(config)
    incompatible["operator"]["learning_rate"] *= 2.0
    assert not resume_science_compatible(previous_config, incompatible)

    operator_config = copy.deepcopy(config["operator"])
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
    data_config = copy.deepcopy(config["data"])
    data_config.update({"history_sensors": 15, "horizon": 3.5})
    model = build_operator(operator_config, data_config, [0.0] * 8, [1.0] * 8, 7)
    histories = torch.zeros(2, 8, 15)
    parameters = torch.tensor([[8.0, 0.25, 0.1], [7.0, 0.22, 0.12]])
    times = torch.linspace(0.0, 3.5, 9)
    prediction = model(histories, parameters, times)
    sensitivity = normalized_model_parameter_sensitivities(
        model, histories, parameters, times[:4]
    )
    assert prediction.shape == (2, 9, 8)
    assert sensitivity.shape == (2, 4, 8, 3)
    physics_times = torch.tensor([[0.2, 3.2], [1.1, 3.4]], requires_grad=True)
    residual = operator_physics_residual(
        model,
        histories,
        parameters,
        physics_times,
        SmartGridConfig.from_mapping(config["equation"]),
        torch.ones(8),
    )
    assert residual.shape == (2, 2, 8)
    assert torch.isfinite(residual).all()
    (prediction.square().mean() + sensitivity.square().mean() + residual.square().mean()).backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
