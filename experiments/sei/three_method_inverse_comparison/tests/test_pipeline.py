"""Regression tests for the DelayedSEI3D three-method inverse comparison."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(1, str(SOURCE / "code"))

import pipeline
from model import build_operator


class InverseComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(
            (ROOT / "configs" / "experiment.json").read_text(encoding="utf-8")
        )

    def test_config_has_three_methods_and_shared_lhs_count(self) -> None:
        pipeline.validate_config(self.config)
        self.assertEqual(tuple(self.config["methods"]), pipeline.METHODS)
        self.assertEqual(self.config["case_count"], 10)
        self.assertEqual(self.config["histories_per_case"], 4)
        self.assertEqual(len(pipeline.condition_definitions(self.config)), 10)
        self.assertEqual(
            self.config["projected_lm"]["random_initializations"],
            self.config["pinndde"]["random_initializations"],
        )

    def test_normalization_projection_and_lhs_are_reproducible(self) -> None:
        bounds = np.asarray([[0.3, 0.7], [0.5, 2.0]])
        normalized = torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
        physical = pipeline.normalized_to_physical(normalized, bounds)
        expected = torch.tensor([[0.3, 0.5], [0.5, 1.25], [0.7, 2.0]])
        torch.testing.assert_close(physical, expected)
        torch.testing.assert_close(
            pipeline.physical_to_normalized(physical, bounds), normalized
        )
        first = pipeline.lhs_physical(5, 123, bounds)
        second = pipeline.lhs_physical(5, 123, bounds)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(first > bounds[:, 0]))
        self.assertTrue(np.all(first < bounds[:, 1]))

    @staticmethod
    def equation_mapping() -> dict[str, object]:
        return {
            "maximum_history": 1.0,
            "transmission_bounds": [0.3, 0.7],
            "convexity_bounds": [0.5, 2.0],
            "delay": 1.0,
            "natural_loss_rate": 0.02,
            "progression_rate": 0.1,
            "recovery_rate": 0.2,
        }

    def test_pinndde_fixed_delay_loss_is_finite_and_differentiable(self) -> None:
        equation = pipeline.DelayedSEIConfig.from_mapping(self.equation_mapping())
        bounds = np.asarray(equation.parameter_bounds)
        initial = pipeline.lhs_physical(2, 9, bounds)
        raw = torch.tensor(
            pipeline.physical_to_logits(initial, bounds), dtype=torch.float32
        )
        model = pipeline.MultiStartPINNDDE(
            2, [4, 5, 4], 1.0, raw, equation.parameter_bounds, 10, 4
        )
        one_history = torch.stack(
            [
                torch.full((2, 17), 0.80),
                torch.full((2, 17), 0.025),
                torch.full((2, 17), 0.012),
            ],
            dim=1,
        )
        histories = one_history.unsqueeze(1).expand(-1, 4, -1, -1).contiguous()
        history_grid = torch.linspace(-1.0, 0.0, 17)
        collocation = torch.tensor(
            [[[0.1, 0.4, 0.8]] * 4, [[0.1, 0.4, 0.8]] * 4],
            requires_grad=True,
        )
        observation_times = torch.tensor([[[0.0, 0.5]] * 4] * 2)
        observations = torch.ones((2, 4, 2, 3))
        objective, components = pipeline.pinndde_losses(
            model,
            histories,
            history_grid,
            collocation,
            observation_times,
            observations,
            [0, 1, 2],
            equation,
            True,
            False,
        )
        self.assertEqual(tuple(objective.shape), (2,))
        self.assertTrue(torch.isfinite(objective).all())
        objective.sum().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
        self.assertEqual(set(components), {"physics_loss", "initial_loss", "data_loss"})

    def create_tiny_checkpoint(self, path: Path) -> None:
        equation = self.equation_mapping()
        operator = {
            "activation": "gelu",
            "latent_dim": 8,
            "history_width": 12,
            "history_depth": 2,
            "parameter_width": 12,
            "parameter_depth": 2,
            "trunk_width": 12,
            "trunk_depth": 2,
            "fourier_modes": 2,
            "parameter_bounds": [[0.3, 0.7], [0.5, 2.0]],
        }
        model = build_operator(
            "separate_parameter_shared", 17, 1.0, operator, torch.device("cpu"), 123
        )
        torch.save(
            {
                "network_format": pipeline.OPERATOR_NETWORK_FORMAT,
                "model_type": "separate_parameter_shared",
                "model_config": model.model_config(),
                "model_state_dict": model.state_dict(),
                "training_seed": 123,
                "best_iteration": 3,
                "best_validation": 1.0,
                "resolved_config": {
                    "data": {
                        "history_level_ranges": [
                            [0.72, 0.92], [0.01, 0.04], [0.004, 0.02]
                        ],
                        "history_sigmas": [0.03, 0.005, 0.003],
                        "history_length_scale": 0.25,
                        "history_bounds": [
                            [0.30, 1.00], [0.001, 0.10], [0.001, 0.06]
                        ],
                        "history_total_upper": 1.0,
                        "history_sensors": 17,
                        "horizon": 1.0,
                        "output_points": 21,
                        "internal_step": 0.01,
                    },
                    "equation": equation,
                    "operator": operator,
                },
            },
            path,
        )

    def test_complete_three_method_cpu_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_text:
            temporary = Path(temporary_text)
            checkpoint = temporary / "best_model.pt"
            output = temporary / "output"
            self.create_tiny_checkpoint(checkpoint)
            status = pipeline.main(
                [
                    "--checkpoint", str(checkpoint),
                    "--config", str(ROOT / "configs" / "experiment.json"),
                    "--output-dir", str(output),
                    "--devices", "cpu",
                    "--processes", "1",
                    "--cpus", "2",
                    "--seed", "20260901",
                    "--smoke-only",
                    "--no-ntfy",
                ]
            )
            self.assertEqual(status, 0)
            pipeline_status = json.loads(
                (output / "pipeline_status.json").read_text(encoding="utf-8")
            )
            self.assertTrue(pipeline_status["success"])
            self.assertEqual(pipeline_status["stage"], "smoke_complete")
            results = list((output / "smoke").rglob("result.json"))
            self.assertEqual(len(results), 30)
            methods = {
                json.loads(path.read_text(encoding="utf-8"))["method_key"]
                for path in results
            }
            self.assertEqual(methods, set(pipeline.METHODS))
            four_history_results = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in results
                if json.loads(path.read_text(encoding="utf-8"))["condition"]
                == "four_histories"
            ]
            self.assertEqual(len(four_history_results), 3)
            self.assertTrue(
                all(row["scalar_observation_count"] == 160 for row in four_history_results)
            )
            reconstruction = np.load(
                next(
                    path.parent / "direct_reconstruction.npy"
                    for path in results
                    if "four_histories" in path.parts
                )
            )
            self.assertEqual(tuple(reconstruction.shape), (4, 21, 3))


if __name__ == "__main__":
    unittest.main()
