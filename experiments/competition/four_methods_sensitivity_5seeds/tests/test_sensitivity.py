"""Regression tests for raw inputs, solver labels, JVPs and scheduling."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from data import DatasetSplit, generate_parameter_sensitivities
from equation import CompetitionConfig, solve_batch
from model import MODEL_TYPES, OPERATOR_NETWORK_FORMAT, build_operator
from scripts.pipeline import build_round_robin_queues, resolve_stage_config, validate_config
from training import model_parameter_sensitivities


class SensitivityExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(
            (ROOT / "configs" / "experiment.json").read_text(encoding="utf-8")
        )
        cls.equation = CompetitionConfig.from_mapping(cls.config["equation"])

    def tiny_operator_config(self) -> dict[str, object]:
        values = dict(self.config["operator"])
        values.update(
            {
                "latent_dim": 8,
                "history_width": 12,
                "history_depth": 2,
                "parameter_width": 12,
                "parameter_depth": 2,
                "trunk_width": 12,
                "trunk_depth": 2,
                "fourier_modes": 2,
            }
        )
        return values

    def test_config_and_twenty_jobs(self) -> None:
        validate_config(self.config)
        jobs = [
            (method, seed)
            for seed in self.config["randomness"]["training_seeds"]
            for method in MODEL_TYPES
        ]
        queues = build_round_robin_queues(jobs, 8)
        self.assertEqual(len(jobs), 20)
        self.assertEqual([len(queue) for queue in queues], [3, 3, 3, 3, 2, 2, 2, 2])

    def test_no_parameter_normalization(self) -> None:
        self.assertEqual(
            OPERATOR_NETWORK_FORMAT,
            "variable_delay_competition_operator_2d_sensitivity_v1",
        )
        model = build_operator(
            "separate_parameter_shared", 21, 1.0, self.tiny_operator_config(),
            torch.device("cpu"), 123,
        )
        self.assertFalse(hasattr(model, "normalize_parameters"))
        captured: list[torch.Tensor] = []
        handle = model.parameter_branch.register_forward_pre_hook(
            lambda _module, inputs: captured.append(inputs[0].detach().clone())
        )
        histories = torch.full((2, 2, 21), 0.6)
        physical = torch.tensor([[0.8, 1.4], [1.1, 0.9]])
        model(histories, physical, torch.linspace(0.0, 1.0, 3))
        handle.remove()
        self.assertTrue(torch.equal(captured[0], physical))

    def test_solver_sensitivity_shape_and_centered_difference(self) -> None:
        histories = np.full((2, 2, 21), 0.6, dtype=np.float32)
        parameters = np.asarray([[1.0, 1.2], [1.25, 0.95]], dtype=np.float32)
        history_times = np.linspace(-1.0, 0.0, 21)
        output_times = np.linspace(0.0, 0.1, 3)
        steps = (0.005, 0.005)
        values = generate_parameter_sensitivities(
            histories, parameters, history_times, output_times,
            self.config["equation"], 0.01, 1, 2, steps,
        )
        self.assertEqual(values.shape, (2, 3, 2, 2))
        self.assertTrue(np.isfinite(values).all())
        plus = parameters.copy()
        minus = parameters.copy()
        plus[:, 0] += steps[0]
        minus[:, 0] -= steps[0]
        expected = (
            solve_batch(histories, plus, history_times, output_times, self.equation, 0.01)
            - solve_batch(histories, minus, history_times, output_times, self.equation, 0.01)
        ) / (2.0 * steps[0])
        np.testing.assert_allclose(values[..., 0], expected, rtol=2.0e-5, atol=2.0e-6)

    def test_jvp_matches_autograd_without_span_scaling(self) -> None:
        model = build_operator(
            "separate_parameter_shared", 21, 1.0, self.tiny_operator_config(),
            torch.device("cpu"), 456,
        )
        histories = torch.rand(2, 2, 21) * 0.2 + 0.5
        parameters = torch.tensor([[0.95, 1.25], [1.20, 0.90]], requires_grad=True)
        times = torch.linspace(0.0, 1.0, 4)
        sensitivities = model_parameter_sensitivities(model, histories, parameters, times)
        self.assertEqual(tuple(sensitivities.shape), (2, 4, 2, 2))
        output = model(histories, parameters, times)
        derivative = torch.autograd.grad(output[0, 2, 1], parameters, retain_graph=True)[0]
        self.assertAlmostEqual(
            float(sensitivities[0, 2, 1, 0].detach()), float(derivative[0, 0]), places=5
        )
        self.assertAlmostEqual(
            float(sensitivities[0, 2, 1, 1].detach()), float(derivative[0, 1]), places=5
        )

    def test_smoke_resolves_sensitivity_points(self) -> None:
        resolved = resolve_stage_config(self.config, True)
        self.assertEqual(
            resolved["operator"]["sensitivity_points"],
            resolved["smoke"]["sensitivity_points"],
        )


if __name__ == "__main__":
    unittest.main()
