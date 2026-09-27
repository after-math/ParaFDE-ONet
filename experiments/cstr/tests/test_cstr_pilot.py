from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from cstr_forward.equation import CSTRConfig, constant_histories
from cstr_forward.model import CSTRParaFDEONet
from cstr_pilot.pilot import (
    build_fixed_operator_features,
    cached_operator_prediction,
    choose_control,
    control_grid,
    direct_screen,
    feasible_mask,
    inverse_direct_lm,
    inverse_para_lm,
    save_csv,
    trajectory_features,
)


ROOT = Path(__file__).resolve().parents[1]


def load_config(name: str) -> dict:
    return json.loads((ROOT / "configs" / name).read_text())


def test_pilot_grid_and_feature_decision() -> None:
    experiment = load_config("experiment.json")
    pilot = load_config("pilot.json")
    equation = CSTRConfig.from_mapping(experiment["equation"])
    controls = control_grid(equation, 5)
    assert controls.shape == (25, 2)
    assert np.allclose(controls[0], [0.25, -0.5])
    assert np.allclose(controls[-1], [3.0, 0.35])

    trajectories = np.zeros((2, 10, 2), dtype=np.float32)
    trajectories[0, :, 0] = 0.4
    trajectories[0, :, 1] = 0.3
    trajectories[1, :, 0] = 0.6
    trajectories[1, :, 1] = 0.7
    features = trajectory_features(trajectories)
    mask = feasible_mask(features, pilot, margin=False)
    assert mask.tolist() == [True, False]

    simple_controls = np.asarray([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    simple_features = {
        "maximum_temperature": np.asarray([0.3, 0.3]),
        "conversion": np.asarray([0.6, 0.6]),
        "temperature_std": np.asarray([0.0, 0.0]),
    }
    index, productivity = choose_control(simple_controls, simple_features, pilot, margin=False)
    assert index == 1
    assert np.allclose(productivity, [0.6, 1.2])


def test_small_direct_screen() -> None:
    experiment = load_config("experiment.json")
    pilot = load_config("pilot.json")
    equation = CSTRConfig.from_mapping(experiment["equation"])
    small = dict(pilot)
    small.update(
        {
            "screen_horizon": 1.0,
            "screen_output_points": 21,
            "direct_batch_size": 2,
        }
    )
    controls = np.asarray([[1.0, 0.0], [1.2, 0.1], [1.4, 0.2]], dtype=np.float32)
    trajectories, elapsed = direct_screen(
        constant_histories(1, 101),
        np.asarray([1.2, 0.8], dtype=np.float64),
        controls,
        small,
        equation,
    )
    assert trajectories.shape == (3, 21, 2)
    assert np.isfinite(trajectories).all()
    assert elapsed >= 0.0


def test_csv_accepts_method_specific_fields(tmp_path: Path) -> None:
    target = tmp_path / "results.csv"
    save_csv(target, [{"method": "Para", "grid_loss": 1.0}, {"method": "LM", "nfev": 7}])
    header, first, second = target.read_text().splitlines()
    assert header == "method,grid_loss,nfev"
    assert first == "Para,1.0,"
    assert second == "LM,,7"


def test_cached_prediction_and_parameter_gradient_match_full_model() -> None:
    experiment = load_config("experiment.json")
    bounds = experiment["equation"]["condition_bounds"]
    model = CSTRParaFDEONet(
        history_sensors=101,
        horizon=30.0,
        latent_dim=16,
        history_width=24,
        history_depth=2,
        condition_width=24,
        condition_depth=2,
        trunk_width=24,
        trunk_depth=2,
        activation="gelu",
        fourier_modes=2,
        initialization_seed=17,
        condition_bounds=bounds,
        history_mean=[0.5, 0.2],
        history_std=[0.2, 0.1],
        output_mean=[0.4, 0.3],
        output_std=[0.2, 0.15],
    )
    model.eval()
    for weight in model.parameters():
        weight.requires_grad_(False)
    history = constant_histories(1, 101)
    times = torch.linspace(0.0, 5.0, 11)
    conditions = torch.tensor(
        [[0.9, 1.0, 1.2, 0.0], [1.1, 1.3, 2.0, 0.2]],
        dtype=torch.float32,
    )
    repeated = torch.as_tensor(np.repeat(history, 2, axis=0))

    full_conditions = conditions.clone().requires_grad_(True)
    full_prediction = model(repeated, full_conditions, times)
    full_loss = full_prediction.square().mean()
    full_gradient = torch.autograd.grad(full_loss, full_conditions)[0]

    history_features, trunk_features, _ = build_fixed_operator_features(
        model, history, times, torch.device("cpu")
    )
    cached_conditions = conditions.clone().requires_grad_(True)
    cached_prediction = cached_operator_prediction(
        model, history_features, trunk_features, cached_conditions
    )
    cached_loss = cached_prediction.square().mean()
    cached_gradient = torch.autograd.grad(cached_loss, cached_conditions)[0]

    assert torch.max(torch.abs(cached_prediction - full_prediction.detach())) < 2.0e-5
    assert torch.abs(cached_loss - full_loss.detach()) < 2.0e-7
    assert torch.max(torch.abs(cached_gradient - full_gradient)) < 2.0e-5


def test_projected_lm_uses_centered_batch_protocol() -> None:
    experiment = load_config("experiment.json")
    pilot = load_config("pilot.json")
    equation = CSTRConfig.from_mapping(experiment["equation"])
    small = dict(pilot)
    small.update(
        {
            "lm_starts": 2,
            "lm_minimum_iterations": 1,
            "lm_max_iterations": 2,
            "lm_early_stop_patience": 5,
            "observation_horizon": 1.0,
            "observation_points": 21,
        }
    )
    history = constant_histories(1, 101)
    control = np.asarray([1.2, 0.0], dtype=np.float64)
    truth = np.asarray([1.0, 1.1], dtype=np.float64)
    observations = direct_screen(
        history,
        truth,
        control.reshape(1, 2),
        {
            **small,
            "screen_horizon": 1.0,
            "screen_output_points": 21,
            "direct_batch_size": 1,
        },
        equation,
    )[0][0]
    estimate, timing = inverse_direct_lm(
        history, control, observations, small, equation
    )
    assert np.all(estimate >= equation.bounds_array[:2, 0])
    assert np.all(estimate <= equation.bounds_array[:2, 1])
    assert timing["solver_batch_calls"] == 7
    assert timing["direct_trajectory_solves"] == 26
    assert timing["lm_iterations_completed"] == 2


def test_frozen_operator_lm_uses_cached_exact_parameter_jacobian() -> None:
    experiment = load_config("experiment.json")
    industrial = load_config("industrial.json")
    equation = CSTRConfig.from_mapping(experiment["equation"])
    model = CSTRParaFDEONet(
        history_sensors=101,
        horizon=30.0,
        latent_dim=16,
        history_width=24,
        history_depth=2,
        condition_width=24,
        condition_depth=2,
        trunk_width=24,
        trunk_depth=2,
        activation="gelu",
        fourier_modes=2,
        initialization_seed=19,
        condition_bounds=experiment["equation"]["condition_bounds"],
        history_mean=[0.5, 0.2],
        history_std=[0.2, 0.1],
        output_mean=[0.4, 0.3],
        output_std=[0.2, 0.15],
    )
    model.eval()
    for weight in model.parameters():
        weight.requires_grad_(False)
    small = dict(industrial)
    small.update(
        {
            "parameter_grid_points": 5,
            "para_lm_starts": 2,
            "para_lm_minimum_iterations": 1,
            "para_lm_max_iterations": 2,
            "para_lm_early_stop_patience": 5,
            "observation_horizon": 1.0,
            "observation_points": 11,
            "operator_batch_size": 32,
        }
    )
    history = constant_histories(1, 101)
    control = np.asarray([1.2, 0.0], dtype=np.float64)
    truth = torch.tensor([[1.05, 1.15, *control]], dtype=torch.float32)
    times = torch.linspace(0.0, 1.0, 11)
    with torch.no_grad():
        observations = model(
            torch.as_tensor(history), truth, times
        )[0].cpu().numpy()
    estimate, timing = inverse_para_lm(
        model,
        history,
        control,
        observations,
        small,
        equation,
        torch.device("cpu"),
    )
    assert np.all(estimate >= equation.bounds_array[:2, 0])
    assert np.all(estimate <= equation.bounds_array[:2, 1])
    assert timing["lm_iterations_completed"] == 2
    assert timing["jacobian_jvp_calls"] == 4
    assert timing["condition_branch_calls"] == 10
    assert timing["forward_condition_evaluations"] == 25 + 2 * 9
    assert timing["cached_parameter_branch_only"] is True
