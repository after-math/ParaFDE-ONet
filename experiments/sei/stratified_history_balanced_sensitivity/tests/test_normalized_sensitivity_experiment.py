"""Regression tests for the delayed-SEI normalized-sensitivity experiment."""

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

from data import (
    allocate_stratum_counts,
    generate_dataset,
    sample_histories,
    sample_stratified_histories,
)
from equation import DelayedSEIConfig, rhs_numpy, solve_batch
from model import MODEL_TYPES, build_operator
from scripts.pipeline import (
    build_round_robin_queues,
    format_final_metric_notification,
    resolve_active_methods,
    resolve_active_seeds,
    validate_config,
)
from training import normalized_model_parameter_sensitivities


class NormalizedSensitivityExperimentTests(unittest.TestCase):
    """Check equation, shapes, normalization, sensitivity labels and schedules."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load and validate the formal JSON once for all test methods.

        No parameters are accepted. The method returns ``None`` and stores the
        parsed config on the test class. For example later tests read fixed delay 1.
        ``unittest`` calls it automatically; reading one JSON file is its side effect.
        """
        cls.config = json.loads(
            (PROJECT_DIR / "configs" / "experiment.json").read_text(encoding="utf-8")
        )
        validate_config(cls.config)

    @staticmethod
    def tiny_operator_config(config: dict[str, object]) -> dict[str, object]:
        """Return a cheap architecture while preserving the SEI interface.

        ``config`` is an operator mapping. The returned independent mapping uses
        latent dimension 8 and width 12 while retaining two parameter bounds. The
        input is unchanged. Forward and JVP tests call this helper.
        """
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
        """Verify all-seed scheduling has 20 jobs and one seed has four.

        No inputs are accepted and ``None`` is returned. Four workers receive queue
        sizes ``[5,5,5,5]``. This has no external side effect and guards the actual
        round-robin scheduling contract used by the pipeline.
        """
        seeds = self.config["randomness"]["training_seeds"]
        jobs = [(method, seed) for seed in seeds for method in MODEL_TYPES]
        self.assertEqual(len(jobs), 20)
        self.assertEqual(
            [len(queue) for queue in build_round_robin_queues(jobs, 4)], [5, 5, 5, 5]
        )
        self.assertEqual(len([(method, seeds[0]) for method in MODEL_TYPES]), 4)

    def test_equation_and_reference_solver_are_finite(self) -> None:
        """Check delayed-SEI formula and a small causal RK4 solve.

        No inputs are accepted. A two-case, 17-sensor history and two parameter rows
        produce finite positive output ``[2,11,3]`` inside the unit simplex. The
        test allocates only local arrays and guards equation signs, shapes and delay.
        """
        equation = DelayedSEIConfig.from_mapping(self.config["equation"])
        current = np.asarray([[0.7, 0.1, 0.05]])
        delayed = np.asarray([[0.72, 0.09, 0.04]])
        parameters = np.asarray([[0.5, 1.25]])
        derivative = rhs_numpy(current, delayed, parameters, equation)
        self.assertEqual(derivative.shape, current.shape)
        histories = np.repeat(current[:, :, None], 17, axis=2)
        histories = np.repeat(histories, 2, axis=0)
        solution = solve_batch(
            histories,
            np.asarray([[0.35, 0.7], [0.65, 1.8]]),
            np.linspace(-1.0, 0.0, 17),
            np.linspace(0.0, 0.1, 11),
            equation,
            0.01,
        )
        self.assertEqual(solution.shape, (2, 11, 3))
        self.assertTrue(np.isfinite(solution).all())
        self.assertTrue(np.all(solution > 0.0))
        self.assertLessEqual(float(np.max(np.sum(solution, axis=-1))), 1.00001)

    def test_history_sampler_enforces_simplex(self) -> None:
        """Verify state-specific GRFs remain positive with ``S+E+I<=1``.

        No inputs are accepted. Thirty-two examples with 17 sensors return
        ``[32,3,17]``. Local RNG state advances and no files are written. This guards
        the proportional simplex projection used before reference solving.
        """
        data = self.config["data"]
        histories = sample_histories(
            32,
            np.linspace(-1.0, 0.0, 17),
            tuple(tuple(row) for row in data["history_level_ranges"]),
            tuple(data["history_sigmas"]),
            float(data["history_length_scale"]),
            tuple(tuple(row) for row in data["history_bounds"]),
            float(data["history_total_upper"]),
            np.random.default_rng(123),
        )
        self.assertEqual(histories.shape, (32, 3, 17))
        self.assertTrue(np.all(histories > 0.0))
        self.assertTrue(np.all(np.sum(histories, axis=1) <= 1.0 + 1e-6))

    def test_stratified_history_sampler_uses_realized_means_and_exact_quotas(self) -> None:
        """Check low/medium/high quotas after projection, not before projection.

        Thirty-two independent rows allocate as 11/11/10. Every third-state sensor
        mean lies in its labeled interval and all histories satisfy positivity and the
        configured total upper bound. Only local deterministic RNG state changes.
        """
        data = self.config["data"]
        strata = [dict(item) for item in data["history_state_3_strata"]]
        self.assertEqual(allocate_stratum_counts(32, strata), [11, 11, 10])
        histories, labels = sample_stratified_histories(
            32,
            np.linspace(-1.0, 0.0, 17),
            tuple(tuple(row) for row in data["history_level_ranges"]),
            tuple(data["history_sigmas"]),
            float(data["history_length_scale"]),
            tuple(tuple(row) for row in data["history_bounds"]),
            float(data["history_total_upper"]),
            strata,
            int(data["history_strata_candidate_multiplier"]),
            int(data["history_strata_max_rounds"]),
            np.random.default_rng(321),
        )
        self.assertEqual(histories.shape, (32, 3, 17))
        self.assertEqual(labels.shape, (32,))
        self.assertTrue(np.all(histories > 0.0))
        self.assertTrue(np.all(np.sum(histories, axis=1) <= 1.0 + 1e-6))
        means = np.mean(histories[:, 2, :], axis=1)
        expected_counts = dict(zip(("low", "medium", "high"), (11, 11, 10)))
        for item in strata:
            name = item["name"]
            low, high = item["mean_range"]
            selected = labels == name
            self.assertEqual(int(np.sum(selected)), expected_counts[name])
            self.assertTrue(np.all(means[selected] >= low - 1e-7))
            self.assertTrue(np.all(means[selected] <= high + 1e-7))

    def test_method_and_seed_command_line_selection(self) -> None:
        """Verify one/subset/all selectors reject unknown or conflicting values."""
        configured_seeds = list(self.config["randomness"]["training_seeds"])
        configured_methods = list(self.config["methods"])
        self.assertEqual(
            resolve_active_seeds(None, "20260901,20260903", configured_seeds),
            [20260901, 20260903],
        )
        self.assertEqual(
            resolve_active_methods(
                "separate_parameter_shared", None, configured_methods
            ),
            ["separate_parameter_shared"],
        )
        with self.assertRaises(ValueError):
            resolve_active_seeds(20260901, "20260902", configured_seeds)
        with self.assertRaises(ValueError):
            resolve_active_methods("missing", None, configured_methods)

    def test_four_models_have_two_parameter_inputs(self) -> None:
        """Check all four structures produce differentiable ``[B,Q,3]`` output.

        No inputs are accepted. Example tensors use B=3, M=17, Q=6 and physical
        lower/mid/upper combinations. Backward populates only temporary gradients.
        The test also verifies shared-p versus state-specific 3p Branch widths.
        """
        tiny = self.tiny_operator_config(dict(self.config["operator"]))
        histories = torch.rand(3, 3, 17) * 0.1 + 0.05
        parameters = torch.tensor([[0.3, 0.5], [0.5, 1.25], [0.7, 2.0]])
        times = torch.linspace(0.0, 1.0, 6)
        for model_type in MODEL_TYPES:
            model = build_operator(model_type, 17, 1.0, tiny, torch.device("cpu"), 7)
            if model_type == "single_branch_deeponet":
                self.assertEqual(model.joint_branch.input.in_features, 3 * 17 + 2)
            elif model_type == "three_branch_mionet":
                self.assertTrue(all(
                    branch.input.in_features == 17 + 2
                    for branch in model.history_parameter_branches
                ))
            else:
                expected = 8 if model_type == "separate_parameter_shared" else 24
                self.assertEqual(model.parameter_branch.input.in_features, 2)
                self.assertEqual(model.parameter_branch.output.out_features, expected)
            prediction = model(histories, parameters, times)
            self.assertEqual(tuple(prediction.shape), (3, 6, 3))
            prediction.square().mean().backward()
            gradients = [
                parameter.grad for parameter in model.parameters()
                if parameter.grad is not None
            ]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(value).all() for value in gradients))

    def test_parameter_normalization_and_jvp(self) -> None:
        """Verify two affine coordinates and two trainable JVP channels.

        No inputs are accepted. Parameter corners/midpoint map to -1/0/1 and B=2,
        Q=3 sensitivities have shape ``[2,3,3,2]``. Backward populates temporary
        weight gradients, guarding higher-order sensitivity supervision.
        """
        tiny = self.tiny_operator_config(dict(self.config["operator"]))
        model = build_operator(
            "separate_parameter_shared", 17, 1.0, tiny, torch.device("cpu"), 11
        )
        physical = torch.tensor([[0.3, 0.5], [0.5, 1.25], [0.7, 2.0]])
        expected = torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
        self.assertTrue(torch.allclose(
            model.normalize_parameters(physical), expected, atol=1e-6
        ))
        histories = torch.rand(2, 3, 17) * 0.1 + 0.05
        parameters = torch.tensor([[0.4, 0.8], [0.6, 1.7]])
        times = torch.linspace(0.0, 1.0, 3)
        with torch.no_grad():
            reporting_sensitivity = normalized_model_parameter_sensitivities(
                model, histories, parameters, times
            )
        self.assertEqual(tuple(reporting_sensitivity.shape), (2, 3, 3, 2))
        self.assertFalse(reporting_sensitivity.requires_grad)
        sensitivity = normalized_model_parameter_sensitivities(
            model,
            histories,
            parameters,
            times,
        )
        self.assertEqual(tuple(sensitivity.shape), (2, 3, 3, 2))
        self.assertTrue(torch.isfinite(sensitivity).all())
        sensitivity.square().mean().backward()
        gradients = [
            parameter.grad for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertGreater(sum(float(value.abs().sum()) for value in gradients), 0.0)

    def test_central_difference_creates_two_scaled_channels(self) -> None:
        """Check labels equal interval-width-scaled derivatives for ``b`` and ``a``.

        A patched trajectory ``endpoint+b^2+3a`` has targets ``0.4*2b`` and
        ``1.5*3``. Tiny train/validation/test arrays remain local; no numerical solve
        or file write occurs. The test guards channel order, scale and split leakage.
        """
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
            """Return ``endpoint+b^2+3a`` with deterministic ``[N,Q,3]`` shape.

            Arguments match ``data.solve_parallel``. The float32 return broadcasts
            across time/state; for b=1,a=1 it adds four. Unused arguments are read
            only and no side effects occur. The patched generator calls it five times.
            """
            del history_times, equation_values, internal_step, workers, chunk_size
            baseline = histories[:, None, :, -1]
            contribution = (
                parameters[:, 0, None, None] ** 2
                + 3.0 * parameters[:, 1, None, None]
            )
            return np.broadcast_to(
                baseline + contribution,
                (histories.shape[0], output_times.size, 3),
            ).astype(np.float32)

        with patch("data.solve_parallel", side_effect=fake_solver):
            splits = generate_dataset(config, 123, 1, False)
        sensitivity = splits["train"].normalized_parameter_sensitivities
        assert sensitivity is not None
        expected_b = 0.8 * splits["train"].parameters[:, 0]
        self.assertEqual(sensitivity.shape[-1], 2)
        self.assertTrue(np.allclose(
            sensitivity[..., 0], expected_b[:, None, None], rtol=3e-3, atol=3e-4
        ))
        self.assertTrue(np.allclose(sensitivity[..., 1], 4.5, rtol=3e-3, atol=3e-4))
        self.assertIsNone(splits["validation"].normalized_parameter_sensitivities)
        self.assertIsNotNone(splits["test"].normalized_parameter_sensitivities)
        self.assertIsNotNone(splits["train"].history_strata)
        self.assertIsNotNone(splits["validation"].history_strata)
        self.assertIsNotNone(splits["test"].history_strata)

    def test_final_ntfy_contains_four_mean_std_rows(self) -> None:
        """Check the final notification reports all four methods and sample SD.

        No inputs are accepted. A synthetic five-seed aggregate produces text with
        each method name, ``±`` values and n=5. No network message is sent. This
        protects the requested final ntfy content independently of network access.
        """
        methods = []
        for index, model_type in enumerate(MODEL_TYPES):
            methods.append({
                "method": self.config["method_display_names"][model_type],
                "seed_count": 5,
                "test_mse_mean": 1e-6 * (index + 1),
                "test_mse_std": 1e-7,
                "test_relative_l2_mean_mean": 1e-3 * (index + 1),
                "test_relative_l2_mean_std": 1e-4,
            })
        message = format_final_metric_notification({"methods": methods}, 7200.0)
        for method in methods:
            self.assertIn(method["method"], message)
        self.assertEqual(message.count("Test MSE:"), 4)
        self.assertEqual(message.count("Relative L2:"), 4)
        self.assertEqual(message.count(" ± "), 8)
        self.assertIn("n=5", message)


if __name__ == "__main__":
    unittest.main()
