from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.panel_scheduler import panel_queues
from common.aggregation import all_pairwise_comparisons
from common.de_panel_runner import _direct_objectives, _reflect_unit, invert_de
from common.lm_panel_runner import _direct_residuals
from common.panel_archive import select_panel_histories
from common.parameters import build_parameter_design
from pipeline import parse_methods


class _ToyAdapter:
    state_dim = 1
    parameter_names = ("p0", "p1")
    parameter_bounds = np.asarray([[2.0, 4.0], [1.0, 2.0]], dtype=np.float64)

    def unit_to_physical_numpy(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        return self.parameter_bounds[:, 0] + values * (
            self.parameter_bounds[:, 1] - self.parameter_bounds[:, 0]
        )

    def physical_to_unit_numpy(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        return (values - self.parameter_bounds[:, 0]) / (
            self.parameter_bounds[:, 1] - self.parameter_bounds[:, 0]
        )

    def solve_batch(
        self,
        histories: np.ndarray,
        parameters: np.ndarray,
        history_grid: np.ndarray,
        output_times: np.ndarray,
        internal_step: float,
    ) -> np.ndarray:
        del history_grid, internal_step
        histories = np.asarray(histories, dtype=np.float64)
        parameters = np.asarray(parameters, dtype=np.float64)
        output_times = np.asarray(output_times, dtype=np.float64)
        values = (
            parameters[:, 0, None]
            + parameters[:, 1, None] * output_times[None]
            + 0.1 * histories[:, 0, -1, None]
        )
        return values[:, :, None]


def _toy_panel_and_observations() -> tuple[dict[str, np.ndarray], np.ndarray]:
    panel = {
        "histories": np.asarray(
            [[[0.0, 0.0, 0.0]], [[0.0, 0.5, 1.0]]], dtype=np.float64
        ),
        "history_grid": np.asarray([-1.0, -0.5, 0.0], dtype=np.float64),
        "output_times": np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        "observation_indices": np.asarray([0, 2], dtype=np.int64),
    }
    adapter = _ToyAdapter()
    truth = np.asarray([[3.1, 1.4], [3.1, 1.4]], dtype=np.float64)
    observations = adapter.solve_batch(
        panel["histories"],
        truth,
        panel["history_grid"],
        panel["output_times"][panel["observation_indices"]],
        0.01,
    )
    return panel, observations


class ProtocolTests(unittest.TestCase):
    def test_single_history_selection_keeps_reference_and_noise_paired(self) -> None:
        panel = {
            "histories": np.arange(6).reshape(2, 1, 3),
            "history_labels": np.asarray(["first", "second"]),
            "reference": np.arange(24).reshape(2, 2, 3, 2),
            "standard_normal_noise": np.arange(16).reshape(2, 2, 2, 2),
        }
        selected = select_panel_histories(panel, 1)
        self.assertEqual(selected["histories"].shape[0], 1)
        self.assertEqual(selected["reference"].shape[1], 1)
        self.assertEqual(selected["standard_normal_noise"].shape[1], 1)
        self.assertEqual(selected["history_indices_used"].tolist(), [0])
        self.assertTrue(np.array_equal(selected["histories"][0], panel["histories"][0]))
        self.assertEqual(panel["histories"].shape[0], 2)

    def test_method_parser_accepts_de_without_changing_default_methods(self) -> None:
        self.assertEqual(parse_methods("frozen,lm,de"), ["frozen", "lm", "de"])

    def test_observation_counts_are_system_specific(self) -> None:
        expected = {"variable_delay.json": 20, "nicholson.json": 20, "delayed_sei.json": 40}
        for filename, count in expected.items():
            values = json.loads((ROOT / "configs" / filename).read_text(encoding="utf-8"))
            self.assertEqual(values["observation_count"], count)

    def test_formal_parameter_design_is_40_plus_10(self) -> None:
        common = json.loads((ROOT / "configs" / "common.json").read_text(encoding="utf-8"))
        points, labels = build_parameter_design(50, common["parameter_design"])
        self.assertEqual(points.shape, (50, 2))
        self.assertEqual(np.sum(labels == "interior"), 40)
        self.assertEqual(np.sum(labels == "edge"), 10)
        self.assertTrue(np.all((points >= 0.0) & (points <= 1.0)))

    def test_panel_is_atomic_scheduler_item(self) -> None:
        queues = panel_queues(20, 8)
        self.assertEqual(queues[0], [0, 8, 16])
        self.assertEqual(queues[7], [7, 15])
        flattened = [panel for queue in queues for panel in queue]
        self.assertEqual(sorted(flattened), list(range(20)))
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_de_and_lm_direct_objectives_match(self) -> None:
        panel, observations = _toy_panel_and_observations()
        adapter = _ToyAdapter()
        unit = np.asarray([[0.2, 0.7], [0.8, 0.1]], dtype=np.float64)
        de_objectives, _, physical = _direct_objectives(
            adapter, unit, panel, observations, 0.01
        )
        lm_residuals, _ = _direct_residuals(
            adapter, physical, panel, observations, 0.01
        )
        self.assertTrue(
            np.allclose(de_objectives, np.mean(lm_residuals**2, axis=1))
        )

    def test_de_is_deterministic_bounded_and_budgeted(self) -> None:
        panel, observations = _toy_panel_and_observations()
        config = {
            "strategy": "rand1bin",
            "coordinate_system": "normalized",
            "population_size": 6,
            "mutation": 0.8,
            "recombination": 0.9,
            "initialization": "latin_hypercube",
            "bound_handling": "reflection",
            "maximum_generations": 2,
            "maximum_candidate_evaluations": 18,
            "internal_step": 0.01,
            "polish": False,
        }
        first = invert_de(_ToyAdapter(), panel, observations, config, 1234)
        second = invert_de(_ToyAdapter(), panel, observations, config, 1234)
        self.assertTrue(np.allclose(first[0], second[0]))
        self.assertEqual(first[1], second[1])
        self.assertTrue(np.all((first[4] >= 0.0) & (first[4] <= 1.0)))
        self.assertEqual(len(first[2]), 3)
        self.assertEqual(first[3]["candidate_objective_evaluations"], 18)
        self.assertEqual(first[3]["direct_trajectory_solves"], 36)
        self.assertEqual(first[3]["solver_batch_calls"], 3)

    def test_de_reflection_handles_values_outside_unit_interval(self) -> None:
        reflected = _reflect_unit(np.asarray([-1.2, -0.2, 0.4, 1.2, 2.2]))
        self.assertTrue(np.all((reflected >= 0.0) & (reflected <= 1.0)))
        self.assertTrue(np.allclose(reflected, [0.8, 0.2, 0.4, 0.8, 0.2]))

    def test_all_three_method_pairs_are_reported(self) -> None:
        rows = []
        for panel in (0, 1):
            for method, value in (("frozen", 0.10), ("lm", 0.11), ("de", 0.09)):
                rows.append(
                    {
                        "method_key": method,
                        "noise_standard_deviation": 0.01,
                        "panel_index": panel,
                        "parameter_normalized_rmse_panel_mean": value + panel * 0.001,
                    }
                )
        comparisons = all_pairwise_comparisons(rows, 100, 7, 0.05)
        pairs = {(row["method_a"], row["method_b"]) for row in comparisons}
        self.assertEqual(
            pairs, {("frozen", "lm"), ("frozen", "de"), ("lm", "de")}
        )


if __name__ == "__main__":
    unittest.main()
