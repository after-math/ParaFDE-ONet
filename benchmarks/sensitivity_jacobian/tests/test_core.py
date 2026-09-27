"""Unit tests for selection, finite differences, JVPs, and reported metrics."""

from __future__ import annotations

from pathlib import Path
import math
import sys
import tempfile
import unittest

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "code"))

from evaluation import (  # noqa: E402
    candidate_parameters,
    cosine_rows,
    evaluate_model_jacobian,
    jacobian_metric_rows,
)
from reference import ReferenceSolver  # noqa: E402
from systems import SystemData, canonical_history_grid, select_indices  # noqa: E402


def sample_system(count: int = 6) -> SystemData:
    times = np.linspace(0.0, 1.0, 5)
    histories = np.ones((count, 2, 3), dtype=np.float32)
    parameters = np.column_stack(
        (np.linspace(0.2, 0.7, count), np.linspace(1.2, 1.7, count))
    )
    return SystemData(
        name="toy",
        display_name="Toy",
        histories=histories,
        parameters=parameters,
        solutions=np.ones((count, times.size, 2), dtype=np.float32),
        history_times=np.linspace(-1.0, 0.0, 3),
        output_times=times,
        indices=np.arange(count),
        strata=None,
        parameter_names=("p1", "p2"),
        parameter_bounds=np.asarray([[0.0, 1.0], [1.0, 2.0]]),
        equation_values={},
    )


class FakeDataModule:
    @staticmethod
    def solve_parallel(
        histories: np.ndarray,
        parameters: np.ndarray,
        history_times: np.ndarray,
        output_times: np.ndarray,
        equation_values: dict,
        internal_step: float,
        workers: int,
        chunk_size: int,
    ) -> np.ndarray:
        del histories, history_times, equation_values, internal_step, workers, chunk_size
        time = output_times[None, :, None]
        p1 = parameters[:, 0, None, None]
        p2 = parameters[:, 1, None, None]
        state1 = 2.0 + p1 * time + p2 * time**2
        state2 = 3.0 + 2.0 * p1 * time - p2 * time**2
        return np.concatenate((state1, state2), axis=-1).astype(np.float32)


class ToyModel(torch.nn.Module):
    def forward(
        self, histories: torch.Tensor, parameters: torch.Tensor, times: torch.Tensor
    ) -> torch.Tensor:
        del histories
        time = times.view(1, -1, 1)
        p1 = parameters[:, 0].view(-1, 1, 1)
        p2 = parameters[:, 1].view(-1, 1, 1)
        return torch.cat(
            (2.0 + p1 * time + p2 * time**2, 3.0 + 2.0 * p1 * time - p2 * time**2),
            dim=-1,
        )


class CoreTests(unittest.TestCase):
    def test_float32_history_grid_is_restored_exactly(self) -> None:
        archived = np.linspace(-1.6, 0.0, 101, dtype=np.float32)
        restored = canonical_history_grid(archived, 1.6)
        self.assertEqual(restored.dtype, np.float64)
        self.assertEqual(float(restored[0]), -1.6)
        self.assertEqual(float(restored[-1]), 0.0)

    def test_balanced_selection(self) -> None:
        parameters = np.column_stack((np.linspace(0.1, 0.9, 12), np.linspace(1.1, 1.9, 12)))
        strata = np.repeat(np.asarray([0, 1, 2]), 4)
        selected = select_indices(
            parameters,
            np.asarray([[0.0, 1.0], [1.0, 2.0]]),
            [[0.01, 0.005], [0.01, 0.005]],
            6,
            7,
            strata,
        )
        values, counts = np.unique(strata[selected], return_counts=True)
        self.assertEqual(values.tolist(), [0, 1, 2])
        self.assertEqual(counts.tolist(), [2, 2, 2])

    def test_finite_difference_and_jvp_match(self) -> None:
        data = sample_system()
        with tempfile.TemporaryDirectory() as directory:
            solver = ReferenceSolver(FakeDataModule, Path(directory), 1, 2)
            directions = [
                solver.sensitivity(data, data.parameters, index, 1e-3, 0.01, "test")
                for index in range(2)
            ]
        reference = np.stack(directions, axis=-1)
        _, estimate = evaluate_model_jacobian(
            ToyModel(),
            data.histories,
            data.parameters,
            data.output_times,
            data.parameter_spans,
            torch.device("cpu"),
            3,
        )
        np.testing.assert_allclose(estimate, reference, rtol=2e-4, atol=2e-4)

    def test_metrics_and_gradient_cosine(self) -> None:
        data = sample_system()
        _, reference = evaluate_model_jacobian(
            ToyModel(),
            data.histories,
            data.parameters,
            data.output_times,
            data.parameter_spans,
            torch.device("cpu"),
            6,
        )
        estimates = {
            "without_sensitivity": reference + 0.1,
            "with_sensitivity": reference + 0.01,
        }
        _, summary = jacobian_metric_rows(data, reference, estimates, 1e-12, 20, 3)
        improved = next(
            row
            for row in summary
            if row["variant"] == "with_sensitivity" and row["parameter"] == "overall"
        )
        self.assertGreater(improved["aggregate_error_reduction_with_sensitivity"], 0.8)

        candidates = candidate_parameters(data.parameters, data.parameter_bounds, 0.05, 9)
        model = ToyModel()
        candidate_prediction, candidate_sensitivity = evaluate_model_jacobian(
            model,
            data.histories,
            candidates,
            data.output_times,
            data.parameter_spans,
            torch.device("cpu"),
            6,
        )
        truth_prediction, _ = evaluate_model_jacobian(
            model,
            data.histories,
            data.parameters,
            data.output_times,
            data.parameter_spans,
            torch.device("cpu"),
            6,
        )
        rows = cosine_rows(
            data,
            truth_prediction,
            candidate_prediction,
            candidate_sensitivity,
            {
                "without_sensitivity": candidate_prediction,
                "with_sensitivity": candidate_prediction,
            },
            {
                "without_sensitivity": candidate_sensitivity,
                "with_sensitivity": candidate_sensitivity,
            },
            4,
            1e-12,
        )
        self.assertTrue(
            all(
                math.isclose(row["jacobian_only_gradient_cosine"], 1.0, abs_tol=1e-6)
                for row in rows
            )
        )


if __name__ == "__main__":
    unittest.main()
