"""Regression tests for the normalized-sensitivity four-method experiment."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "code"))

from data import generate_dataset
from model import MODEL_TYPES, build_operator
from scripts.pipeline import build_round_robin_queues, validate_config
from training import normalized_model_parameter_sensitivities


class NormalizedSensitivityExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(
            (PROJECT_DIR / "configs" / "experiment.json").read_text(encoding="utf-8")
        )
        validate_config(cls.config)

    @staticmethod
    def tiny_operator_config(config: dict[str, object]) -> dict[str, object]:
        tiny = copy.deepcopy(config)
        tiny.update({
            "latent_dim": 8,
            "history_width": 12,
            "history_depth": 2,
            "parameter_width": 12,
            "parameter_depth": 2,
            "trunk_width": 12,
            "trunk_depth": 2,
            "fourier_modes": 2,
        })
        return tiny

    def test_five_seed_and_single_seed_schedules(self) -> None:
        seeds = self.config["randomness"]["training_seeds"]
        jobs = [(method, seed) for seed in seeds for method in MODEL_TYPES]
        self.assertEqual(len(jobs), 20)
        queues = build_round_robin_queues(jobs, 4)
        self.assertEqual([len(queue) for queue in queues], [5, 5, 5, 5])
        self.assertEqual(len([(method, seeds[0]) for method in MODEL_TYPES]), 4)

    def test_latent_dim_is_per_state_for_all_four_models(self) -> None:
        tiny = self.tiny_operator_config(dict(self.config["operator"]))
        histories = torch.rand(3, 4, 17) + 0.2
        physical_parameters = torch.tensor(
            [[3.0, 0.8], [5.0, 1.1], [7.0, 1.4]], dtype=torch.float32
        )
        times = torch.linspace(0.0, 1.0, 6)
        for model_type in MODEL_TYPES:
            model = build_operator(
                model_type, 17, 1.0, tiny, torch.device("cpu"), 20260901
            )
            if model_type == "single_branch_deeponet":
                self.assertEqual(model.joint_branch.output.out_features, 4 * 8)
            elif model_type == "four_branch_mionet":
                self.assertTrue(all(
                    branch.output.out_features == 4 * 8
                    for branch in model.history_parameter_branches
                ))
            else:
                self.assertTrue(all(
                    branch.output.out_features == 4 * 8
                    for branch in model.history_branches
                ))
                expected = 8 if model_type == "separate_parameter_shared" else 4 * 8
                self.assertEqual(model.parameter_branch.output.out_features, expected)
            self.assertEqual(model.time_trunk.output.out_features, 8)
            prediction = model(histories, physical_parameters, times)
            self.assertEqual(tuple(prediction.shape), (3, 6, 4))
            prediction.square().mean().backward()
            gradients = [
                parameter.grad for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(value).all() for value in gradients))

    def test_every_parameter_consuming_branch_receives_normalized_values(self) -> None:
        tiny = self.tiny_operator_config(dict(self.config["operator"]))
        histories = torch.rand(3, 4, 17) + 0.2
        parameters = torch.tensor(
            [[3.0, 0.8], [5.0, 1.1], [7.0, 1.4]], dtype=torch.float32
        )
        expected = torch.tensor(
            [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]], dtype=torch.float32
        )
        times = torch.linspace(0.0, 1.0, 4)
        for model_type in MODEL_TYPES:
            model = build_operator(model_type, 17, 1.0, tiny, torch.device("cpu"), 7)
            captured: list[torch.Tensor] = []

            def capture_input(
                module: torch.nn.Module, arguments: tuple[torch.Tensor, ...]
            ) -> None:
                del module
                captured.append(arguments[0].detach().clone())

            if model_type == "single_branch_deeponet":
                hooks = [model.joint_branch.register_forward_pre_hook(capture_input)]
            elif model_type == "four_branch_mionet":
                hooks = [
                    branch.register_forward_pre_hook(capture_input)
                    for branch in model.history_parameter_branches
                ]
            else:
                hooks = [model.parameter_branch.register_forward_pre_hook(capture_input)]
            model(histories, parameters, times)
            for hook in hooks:
                hook.remove()
            self.assertTrue(captured)
            for branch_input in captured:
                self.assertTrue(torch.allclose(branch_input[:, -2:], expected, atol=1e-6))

    def test_jvp_sensitivity_loss_backpropagates_to_model_weights(self) -> None:
        tiny = self.tiny_operator_config(dict(self.config["operator"]))
        model = build_operator(
            "separate_parameter_shared", 17, 1.0, tiny, torch.device("cpu"), 11
        )
        histories = torch.rand(2, 4, 17) + 0.2
        parameters = torch.tensor([[4.5, 1.0], [5.5, 1.2]], dtype=torch.float32)
        times = torch.linspace(0.0, 1.0, 3)
        sensitivity = normalized_model_parameter_sensitivities(
            model, histories, parameters, times
        )
        self.assertEqual(tuple(sensitivity.shape), (2, 3, 4, 2))
        self.assertTrue(torch.isfinite(sensitivity).all())
        sensitivity.square().mean().backward()
        gradients = [
            parameter.grad for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertGreater(sum(float(value.abs().sum()) for value in gradients), 0.0)

    def test_solver_central_differences_create_width_normalized_labels(self) -> None:
        config = copy.deepcopy(self.config)
        config["data"].update({
            "history_count": 5,
            "parameter_count": 5,
            "pairs_per_history": 2,
            "validation_count": 3,
            "test_count": 3,
            "history_sensors": 17,
            "horizon": 1.0,
            "output_points": 7,
        })

        def fake_solver(
            histories: np.ndarray,
            parameters: np.ndarray,
            history_times: np.ndarray,
            output_times: np.ndarray,
            equation_values: dict[str, object],
            internal_step: float,
            workers: int,
            chunk_size: int,
        ) -> np.ndarray:
            del history_times, equation_values, internal_step, workers, chunk_size
            baseline = histories[:, None, :, -1]
            parameter_term = (
                parameters[:, 0] ** 2 + 3.0 * parameters[:, 1]
            )[:, None, None]
            return np.broadcast_to(
                baseline + parameter_term,
                (histories.shape[0], output_times.size, 4),
            ).astype(np.float32)

        with patch("data.solve_parallel", side_effect=fake_solver):
            splits = generate_dataset(config, 123, 1, False)
        train = splits["train"]
        self.assertIsNotNone(train.normalized_parameter_sensitivities)
        sensitivity = train.normalized_parameter_sensitivities
        assert sensitivity is not None
        expected_beta = 8.0 * train.parameters[:, 0]
        self.assertTrue(np.allclose(
            sensitivity[..., 0], expected_beta[:, None, None], rtol=1e-4
        ))
        self.assertTrue(np.allclose(sensitivity[..., 1], 1.8, rtol=1e-4))
        self.assertIsNone(splits["validation"].normalized_parameter_sensitivities)
        self.assertIsNone(splits["test"].normalized_parameter_sensitivities)

    def test_tau_sensitivity_weight_is_strictly_larger(self) -> None:
        operator = self.config["operator"]
        self.assertGreater(
            operator["tau_sensitivity_weight"], operator["beta_sensitivity_weight"]
        )


if __name__ == "__main__":
    unittest.main()
