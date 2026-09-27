from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from cstr_forward.data import generate_dataset, normalization_statistics, resolve_sizes
from cstr_forward.equation import CSTRConfig, constant_histories, solve_batch
from cstr_forward.model import build_operator
from cstr_forward.training import (
    normalized_model_sensitivities,
    operator_physics_residual,
)


ROOT = Path(__file__).resolve().parents[1]


def config() -> dict:
    return json.loads((ROOT / "configs" / "experiment.json").read_text())


def test_reference_solver_and_step_convergence() -> None:
    values = config()
    equation = CSTRConfig.from_mapping(values["equation"])
    conditions = np.asarray([[1.0, 1.1, 1.2, 0.0]], dtype=np.float32)
    coarse_history = constant_histories(1, 101)
    coarse = solve_batch(coarse_history, conditions, 2.0, 41, 0.01, equation)
    fine_history = constant_histories(1, 201)
    fine = solve_batch(fine_history, conditions, 2.0, 41, 0.005, equation)
    relative = np.linalg.norm(coarse - fine) / np.linalg.norm(fine)
    assert coarse.shape == (1, 41, 2)
    assert relative < 1e-3


def test_smoke_dataset_shapes_and_no_leakage() -> None:
    values = config()
    splits, diagnostics = generate_dataset(values, workers=1, smoke=True)
    train = splits["train"]
    assert train.histories.shape == (16, 2, 101)
    assert train.conditions.shape == (16, 4)
    assert train.solutions.shape == (16, 21, 2)
    assert train.normalized_sensitivities is not None
    assert train.normalized_sensitivities.shape == (16, 21, 2, 2)
    assert splits["validation"].histories.shape == (4, 2, 101)
    assert splits["test"].histories.shape == (4, 2, 101)
    assert diagnostics["train_rows"] == 16
    train_endpoints = {row.tobytes() for row in train.histories[::2]}
    validation_endpoints = {row.tobytes() for row in splits["validation"].histories}
    assert train_endpoints.isdisjoint(validation_endpoints)


def test_model_sensitivity_and_physics_shapes() -> None:
    values = config()
    splits, _ = generate_dataset(values, workers=1, smoke=True)
    data_config, operator_config = resolve_sizes(values, smoke=True)
    normalization = normalization_statistics(splits["train"])
    device = torch.device("cpu")
    model = build_operator(
        data_config,
        operator_config,
        values["equation"],
        normalization,
        device,
        123,
    )
    histories = torch.as_tensor(splits["train"].histories[:2])
    conditions = torch.as_tensor(splits["train"].conditions[:2])
    times = torch.linspace(0.0, 1.0, 5)
    prediction = model(histories, conditions, times)
    sensitivity = normalized_model_sensitivities(
        model, histories, conditions, times[:3]
    )
    physics_times = torch.rand(2, 3, requires_grad=True)
    residual = operator_physics_residual(
        model,
        histories,
        conditions,
        physics_times,
        CSTRConfig.from_mapping(values["equation"]),
    )
    condition_features = model.encode_conditions(conditions)
    history_features = model.encode_histories(histories)
    assert prediction.shape == (2, 5, 2)
    assert condition_features.shape == (2, model.latent_dim)
    assert history_features.shape == (2, 2, model.latent_dim)
    assert model.condition_branch.output.out_features == model.latent_dim
    assert model.model_config()["condition_feature_mode"] == "shared_p"
    assert sensitivity.shape == (2, 3, 2, 2)
    assert residual.shape == (2, 3, 2)
    assert torch.isfinite(prediction).all()
    assert torch.isfinite(sensitivity).all()
    assert torch.isfinite(residual).all()
