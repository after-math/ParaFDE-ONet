"""Regression tests for the fixed-parameter, three-method, five-seed experiment."""

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


class FixedParameterExperimentTests(unittest.TestCase):
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

    def test_five_seed_schedule_and_single_seed_subset(self) -> None:
        seeds = self.config["randomness"]["training_seeds"]
        jobs = [(method, seed) for seed in seeds for method in MODEL_TYPES]
        self.assertEqual(len(jobs), 15)
        queues = build_round_robin_queues(jobs, 4)
        self.assertEqual(sorted(len(queue) for queue in queues), [3, 4, 4, 4])
        single_seed_jobs = [(method, seeds[0]) for method in MODEL_TYPES]
        self.assertEqual(len(single_seed_jobs), 3)

    def test_latent_dim_is_per_state_and_all_models_backpropagate(self) -> None:
        tiny = self.tiny_operator_config(dict(self.config["operator"]))
        histories = torch.rand(3, 4, 17) + 0.2
        physical_parameters = torch.tensor([[5.0, 1.1]]).repeat(3, 1)
        times = torch.linspace(0.0, 1.0, 6)
        for model_type in MODEL_TYPES:
            model = build_operator(
                model_type, 17, 1.0, tiny, torch.device("cpu"), 20260901
            )
            self.assertEqual(model.history_branch.output.out_features, 4 * 8)
            expected_trunk = 8 if model_type == "split_branch_shared_trunk" else 4 * 8
            self.assertEqual(model.time_trunk.output.out_features, expected_trunk)
            prediction = model(histories, physical_parameters, times)
            self.assertEqual(tuple(prediction.shape), (3, 6, 4))
            prediction.square().mean().backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_parameter_branch_receives_real_fixed_parameter_and_is_trainable(self) -> None:
        tiny = self.tiny_operator_config(dict(self.config["operator"]))
        model = build_operator(
            "separate_parameter_shared_constant",
            17,
            1.0,
            tiny,
            torch.device("cpu"),
            7,
        )
        captured: list[torch.Tensor] = []

        def capture_input(module: torch.nn.Module, arguments: tuple[torch.Tensor, ...]) -> None:
            del module
            captured.append(arguments[0].detach().clone())

        hook = model.parameter_branch.register_forward_pre_hook(capture_input)
        histories = torch.rand(2, 4, 17) + 0.2
        times = torch.linspace(0.0, 1.0, 5)
        supplied_a = torch.tensor([[5.0, 1.1], [5.0, 1.1]])
        supplied_b = torch.tensor([[3.0, 0.8], [7.0, 1.4]])
        prediction_a = model(histories, supplied_a, times)
        prediction_b = model(histories, supplied_b, times)
        hook.remove()
        expected = torch.tensor([[5.0, 1.1], [5.0, 1.1]])
        self.assertTrue(all(torch.equal(values, expected) for values in captured))
        self.assertTrue(torch.equal(prediction_a, prediction_b))
        prediction_a.square().mean().backward()
        parameter_gradients = [
            parameter.grad
            for parameter in model.parameter_branch.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(parameter_gradients)
        self.assertGreater(
            sum(float(gradient.abs().sum()) for gradient in parameter_gradients), 0.0
        )

    def test_every_dataset_row_uses_five_and_one_point_one(self) -> None:
        config = copy.deepcopy(self.config)
        config["data"].update({
            "history_count": 5,
            "validation_count": 3,
            "test_count": 3,
            "history_sensors": 17,
            "horizon": 1.0,
            "output_points": 21,
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
            del parameters, history_times, equation_values, internal_step, workers, chunk_size
            return np.repeat(histories[:, None, :, -1], output_times.size, axis=1)

        with patch("data.solve_parallel", side_effect=fake_solver):
            splits = generate_dataset(config, 123, 1, False)
        expected = np.asarray([5.0, 1.1], dtype=np.float32)
        for split in splits.values():
            self.assertTrue(np.all(split.parameters == expected[None, :]))
        train_histories = {row.tobytes() for row in splits["train"].histories}
        self.assertTrue(
            all(row.tobytes() not in train_histories for row in splits["test"].histories)
        )


if __name__ == "__main__":
    unittest.main()
