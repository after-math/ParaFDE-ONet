"""Numerical and structural regression tests for the variable-delay project."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "code"))

from equation import CompetitionConfig, delay_numpy, solve_batch
from model import MODEL_TYPES, build_operator, count_trainable_parameters
from training import operator_physics_residual


class CoreTests(unittest.TestCase):
    """Check the exact equation, four structures, gradients and physics residual."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the single source of formal configuration for all test cases.

        The method takes no explicit input and returns ``None``.  It stores config
        and validated equation on the test class; e.g. delay mean is 0.8.  unittest
        calls it once before the methods below.
        """
        cls.config = json.loads((PROJECT_DIR / "configs" / "experiment.json").read_text())
        cls.equation = CompetitionConfig.from_mapping(cls.config["equation"])

    def test_delay_range_and_reference_solver(self) -> None:
        """Verify sinusoidal range and finite positive RK4 trajectory output.

        No input is required and return is ``None``.  It solves two one-second cases
        and asserts shape ``[2,21,2]`` plus delay extrema 0.6/1.0.  unittest invokes
        it as an equation regression test.
        """
        period_grid = np.linspace(0.0, 5.0, 1001)
        delays = delay_numpy(period_grid, self.equation)
        self.assertAlmostEqual(float(delays.min()), 0.6, places=6)
        self.assertAlmostEqual(float(delays.max()), 1.0, places=6)
        histories = np.full((2, 2, 101), 0.6, dtype=np.float32)
        parameters = np.asarray([[0.8, 1.4], [1.4, 0.8]], dtype=np.float32)
        solution = solve_batch(
            histories, parameters, np.linspace(-1.0, 0.0, 101),
            np.linspace(0.0, 1.0, 21), self.equation, 0.01,
        )
        self.assertEqual(solution.shape, (2, 21, 2))
        self.assertTrue(np.isfinite(solution).all())
        self.assertGreater(float(solution.min()), 0.0)

    def test_all_model_shapes_and_gradients(self) -> None:
        """Verify every registered structure predicts and receives branch gradients.

        No input is required and return is ``None``.  For each tiny model, histories
        ``[3,2,21]`` and six times produce ``[3,6,2]``; backward must create finite,
        nonzero gradients.  unittest calls it to catch accidental method aliasing.
        """
        tiny = dict(self.config["operator"])
        tiny.update({
            "latent_dim": 8, "history_width": 12, "history_depth": 2,
            "parameter_width": 12, "parameter_depth": 2,
            "trunk_width": 12, "trunk_depth": 2, "fourier_modes": 2,
        })
        histories = torch.rand(3, 2, 21) + 0.2
        parameters = torch.rand(3, 2) * 0.6 + 0.8
        times = torch.linspace(0.0, 1.0, 6)
        counts = []
        for model_type in MODEL_TYPES:
            model = build_operator(model_type, 21, 1.0, tiny, torch.device("cpu"), 123)
            prediction = model(histories, parameters, times)
            self.assertEqual(tuple(prediction.shape), (3, 6, 2))
            prediction.square().mean().backward()
            gradient_sum = sum(
                float(parameter.grad.abs().sum())
                for parameter in model.parameters()
                if parameter.grad is not None
            )
            self.assertTrue(np.isfinite(gradient_sum))
            self.assertGreater(gradient_sum, 0.0)
            counts.append(count_trainable_parameters(model))
        self.assertEqual(len(set(counts)), 4)

    def test_variable_delay_physics_residual(self) -> None:
        """Verify mixed history/operator delay queries yield finite weight gradients.

        No input is required and return is ``None``.  A tiny Separate-Parameter model
        evaluates residual ``[2,5,2]`` at times spanning below/above the delay and
        backpropagates its MSE.  unittest calls it to protect the formal physics path.
        """
        tiny = dict(self.config["operator"])
        tiny.update({
            "latent_dim": 8, "history_width": 12, "history_depth": 2,
            "parameter_width": 12, "parameter_depth": 2,
            "trunk_width": 12, "trunk_depth": 2, "fourier_modes": 2,
        })
        model = build_operator(
            "separate_parameter_shared", 21, 1.0, tiny, torch.device("cpu"), 321
        )
        histories = torch.rand(2, 2, 21) + 0.2
        parameters = torch.rand(2, 2) * 0.6 + 0.8
        times = torch.linspace(0.05, 1.0, 5).repeat(2, 1).requires_grad_(True)
        residual = operator_physics_residual(model, histories, parameters, times, self.equation)
        self.assertEqual(tuple(residual.shape), (2, 5, 2))
        self.assertTrue(torch.isfinite(residual).all())
        residual.square().mean().backward()
        self.assertTrue(any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))


if __name__ == "__main__":
    unittest.main()
