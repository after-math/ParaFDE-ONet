"""Regression tests for the online speed benchmark."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "four_methods_sensitivity_5seeds"
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(1, str(SOURCE / "code"))

import benchmark
from model import OPERATOR_NETWORK_FORMAT, build_operator


class SpeedBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(
            (ROOT / "configs" / "benchmark.json").read_text(encoding="utf-8")
        )

    def test_formal_protocol_is_fixed(self) -> None:
        benchmark.validate_config(self.config)
        self.assertEqual(
            self.config["case_counts"],
            [1, 10, 50, 100, 500, 1000, 2048, 4096],
        )
        self.assertEqual(self.config["repeats"], 5)
        self.assertEqual(self.config["model_type"], "separate_parameter_shared")

    @staticmethod
    def create_tiny_checkpoint(path: Path) -> None:
        equation = {
            "maximum_history": 1.0,
            "parameter_bounds": [[0.8, 1.4], [0.8, 1.4]],
            "carrying_capacities": [1.0, 1.0],
            "competition_coefficients": [0.35, 0.30],
            "delay_mean": 0.8,
            "delay_amplitude": 0.2,
            "delay_angular_frequency": 1.2566370614359172,
        }
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
        }
        model = build_operator(
            "separate_parameter_shared",
            21,
            1.0,
            operator,
            torch.device("cpu"),
            123,
        )
        torch.save(
            {
                "network_format": OPERATOR_NETWORK_FORMAT,
                "model_type": "separate_parameter_shared",
                "model_config": model.model_config(),
                "model_state_dict": model.state_dict(),
                "training_seed": 123,
                "best_iteration": 3,
                "resolved_config": {
                    "data": {
                        "history_sensors": 21,
                        "history_mean": [0.58, 0.60],
                        "history_sigma": 0.08,
                        "history_length_scale": 0.35,
                        "history_bounds": [0.2, 1.2],
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

    def test_complete_cpu_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_text:
            temporary = Path(temporary_text)
            checkpoint = temporary / "best_model.pt"
            output = temporary / "output"
            self.create_tiny_checkpoint(checkpoint)
            status = benchmark.main(
                [
                    "--checkpoint", str(checkpoint),
                    "--config", str(ROOT / "configs" / "benchmark.json"),
                    "--output-dir", str(output),
                    "--device", "cpu",
                    "--cpus", "2",
                    "--smoke-only",
                ]
            )
            self.assertEqual(status, 0)
            state = json.loads(
                (output / "pipeline_status.json").read_text(encoding="utf-8")
            )
            self.assertTrue(state["success"])
            self.assertEqual(state["stage"], "smoke_complete")
            summary = json.loads(
                (output / "smoke" / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["case_counts"], [1, 10])
            self.assertEqual(len(summary["timings"]), 2)
            self.assertTrue((output / "smoke" / "raw_timings.csv").is_file())
            self.assertTrue((output / "smoke" / "figures" / "online_solution_time.png").is_file())


if __name__ == "__main__":
    unittest.main()
