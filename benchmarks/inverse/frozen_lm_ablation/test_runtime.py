from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent / "code"
for path in (str(CODE_ROOT), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from runtime import projected_damped_lm
from runtime import FrozenLMRuntime
from run import SYSTEM_ADAPTER_KEYS, SYSTEM_CONFIGS, resolve_histories_per_request


class _ToyAdapter:
    def encode_parameter(
        self, model: object, normalized: torch.Tensor, detach: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del model
        features = normalized.detach() if detach else normalized
        return features, normalized

    def predict_from_features(
        self,
        model: object,
        history_features: torch.Tensor,
        parameter_features: torch.Tensor,
        trunk_features: torch.Tensor,
    ) -> torch.Tensor:
        del model
        values = (
            parameter_features[:, 0, None, None, None]
            + parameter_features[:, 1, None, None, None]
            * trunk_features[None, None, :, None]
        )
        return values.expand(
            -1, history_features.shape[0], trunk_features.shape[0], 1
        )

    def normalized_to_physical_tensor(self, values: torch.Tensor) -> torch.Tensor:
        return values


class FrozenLMTests(unittest.TestCase):
    def test_d3d_maps_to_sei_and_requires_all_eight_histories(self) -> None:
        self.assertEqual(SYSTEM_CONFIGS["d3d"], "delayed_sei.json")
        self.assertEqual(SYSTEM_ADAPTER_KEYS["d3d"], "sei")
        self.assertEqual(resolve_histories_per_request("d3d", None, 8), 8)
        self.assertEqual(resolve_histories_per_request("d3d", 8, 8), 8)
        with self.assertRaisesRegex(ValueError, "all eight"):
            resolve_histories_per_request("d3d", 1, 8)
        with self.assertRaisesRegex(ValueError, "eight-history panel archive"):
            resolve_histories_per_request("d3d", None, 3)

    def test_existing_systems_keep_single_history_default(self) -> None:
        self.assertEqual(resolve_histories_per_request("v2d", None, 3), 1)
        self.assertEqual(resolve_histories_per_request("n4d", None, 3), 1)

    def test_exact_jvp_lm_converges_on_batched_quadratic_residual(self) -> None:
        config = {
            "lm_min_iterations": 3,
            "lm_max_iterations": 20,
            "lm_initial_damping": 0.01,
            "lm_damping_increase": 10.0,
            "lm_damping_decrease": 3.0,
            "lm_minimum_damping": 1.0e-12,
            "lm_maximum_damping": 1.0e12,
            "lm_relative_objective_tolerance": 1.0e-8,
            "lm_parameter_tolerance": 1.0e-6,
            "lm_early_stop_patience": 3,
        }
        starts = torch.tensor([[-0.8, 0.9], [0.7, -0.7]], dtype=torch.float32)

        def residual(values: torch.Tensor) -> torch.Tensor:
            return torch.stack(
                (values[:, 0] - 0.2, 2.0 * (values[:, 1] + 0.3)), dim=1
            )

        estimate, objective, trace, metadata = projected_damped_lm(
            residual, starts, config
        )
        self.assertTrue(torch.allclose(estimate, torch.tensor([0.2, -0.3]), atol=1e-5))
        self.assertLess(objective, 1e-10)
        self.assertGreaterEqual(len(trace), 3)
        self.assertEqual(metadata["jacobian_jvp_calls"], 2 * len(trace))
        self.assertTrue(torch.all((estimate >= -1.0) & (estimate <= 1.0)))

    def test_runtime_keeps_grid_screening_and_replaces_only_optimizer(self) -> None:
        runtime = FrozenLMRuntime.__new__(FrozenLMRuntime)
        runtime.device = torch.device("cpu")
        runtime.model = object()
        runtime.adapter = _ToyAdapter()
        runtime.config = {
            "grid_batch_size": 4,
            "top_k": 3,
            "lm_starts": 3,
            "lm_min_iterations": 3,
            "lm_max_iterations": 20,
            "lm_initial_damping": 0.01,
            "lm_damping_increase": 10.0,
            "lm_damping_decrease": 3.0,
            "lm_minimum_damping": 1.0e-12,
            "lm_maximum_damping": 1.0e12,
            "lm_relative_objective_tolerance": 1.0e-8,
            "lm_parameter_tolerance": 1.0e-6,
            "lm_early_stop_patience": 3,
        }
        runtime.grid_normalized = torch.tensor(
            [
                [-1.0, -1.0],
                [-0.5, -0.5],
                [0.0, 0.0],
                [0.5, 0.5],
                [1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        runtime.grid_parameter_features = runtime.grid_normalized.clone()
        runtime.trunk_features = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
        history_features = torch.zeros((2, 1), dtype=torch.float32)
        truth = torch.tensor([0.2, -0.3], dtype=torch.float32)
        observations = (
            truth[0] + truth[1] * runtime.trunk_features
        )[None, :, None].expand(2, -1, -1)

        estimate, objective, trace, timing, starts = runtime.invert(
            history_features, observations
        )
        self.assertTrue(np.allclose(estimate, truth.numpy(), atol=1e-5))
        self.assertLess(objective, 1e-10)
        self.assertEqual(len(starts), 3)
        self.assertGreaterEqual(len(trace), 3)
        self.assertIn("projected_lm_seconds", timing)
        self.assertNotIn("projected_adam_seconds", timing)


if __name__ == "__main__":
    unittest.main()
