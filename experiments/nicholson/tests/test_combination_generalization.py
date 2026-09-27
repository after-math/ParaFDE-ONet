"""Leakage and scheduling tests for Nicholson four-patch combination generalization."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "code"))

from data import generate_or_load_combination_dataset
from model import MODEL_TYPES
from scripts.combination_generalization import (
    resolve_combination_stage,
    validate_combination_config,
)
from scripts.pipeline import build_round_robin_queues


class CombinationTests(unittest.TestCase):
    """Protect the four semantic holdouts and four-job formal protocol."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load and validate the single formal combination configuration.

        No explicit input is required and the return is ``None``.  It stores config;
        for example there is one training seed and four methods.  unittest invokes
        it once before all tests and no file is written.
        """
        cls.config = json.loads(
            (PROJECT_DIR / "configs" / "combination_generalization.json").read_text(
                encoding="utf-8"
            )
        )
        validate_combination_config(cls.config)

    @staticmethod
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
        """Return finite positive trajectories determined by histories and parameters.

        Inputs match ``data.solve_parallel`` and include histories ``[N,4,M]`` and
        parameters ``[N,2]``.  The float32 return is ``[N,Q,4]``; e.g. eight smoke
        cases and 21 times return ``[8,21,4]``.  It has no side effect and ignores
        resource/equation fields.  The smoke dataset test uses it instead of RK4.
        """
        del history_times, equation_values, internal_step, workers, chunk_size
        base = histories[:, :, -1][:, None, :]
        time = output_times[None, :, None] * 1.0e-4
        parameter = parameters.mean(axis=1)[:, None, None] * 1.0e-3
        return np.asarray(base + time + parameter, dtype=np.float32)

    def test_four_jobs_round_robin_on_four_gpus(self) -> None:
        """Verify one seed by four methods produces four complete GPU queues.

        No input is required and the return is ``None``. Four pairs on four workers
        yield lengths ``[1,1,1,1]`` with no missing or duplicate job. unittest invokes
        it to protect the intended four-GPU execution pattern.
        """
        jobs = [
            (method, seed)
            for seed in self.config["randomness"]["training_seeds"]
            for method in MODEL_TYPES
        ]
        queues = build_round_robin_queues(jobs, 4)
        self.assertEqual([len(queue) for queue in queues], [1, 1, 1, 1])
        self.assertEqual(set(job for queue in queues for job in queue), set(jobs))

    def test_smoke_dataset_has_no_combination_leakage(self) -> None:
        """Verify SH--SP edges are held out and UH/UP nodes are genuinely new.

        No input is required and the return is ``None``.  A temporary smoke dataset
        has four eight-case splits; no SH--SP value pair occurs in training, and the
        UH--UP histories/parameters have no exact training rows.  It writes only a
        temporary directory and unittest removes it automatically.
        """
        resolved = resolve_combination_stage(copy.deepcopy(self.config), True)
        with tempfile.TemporaryDirectory() as temporary:
            with patch("data.solve_parallel", side_effect=self.fake_solver):
                training, named, manifest = generate_or_load_combination_dataset(
                    Path(temporary), resolved, 20260818, 1, True
                )
            self.assertTrue(all(split.parameters.shape[0] == 8 for split in named.values()))
            self.assertEqual(manifest["leakage_audit"]["training_edge_overlap"], 0)
            train_pairs = {
                (history.tobytes(), parameter.tobytes())
                for history, parameter in zip(
                    training["train"].histories, training["train"].parameters
                )
            }
            shsp = named["seen_history_seen_parameter"]
            self.assertTrue(all(
                (history.tobytes(), parameter.tobytes()) not in train_pairs
                for history, parameter in zip(shsp.histories, shsp.parameters)
            ))
            train_histories = {history.tobytes() for history in training["train"].histories}
            train_parameters = {parameter.tobytes() for parameter in training["train"].parameters}
            uhup = named["unseen_history_unseen_parameter"]
            self.assertTrue(all(history.tobytes() not in train_histories for history in uhup.histories))
            self.assertTrue(all(parameter.tobytes() not in train_parameters for parameter in uhup.parameters))

    def test_online_cartesian_physics_is_rejected(self) -> None:
        """Verify the validator prevents physics-loss combination leakage.

        No input is required and the return is ``None``.  A copied config with
        ``online_cartesian`` raises ``ValueError`` while the formal config passes.
        There is no side effect.  unittest invokes it as the key scientific protocol
        regression test.
        """
        altered = copy.deepcopy(self.config)
        altered["operator"]["physics_pairing_mode"] = "online_cartesian"
        with self.assertRaises(ValueError):
            validate_combination_config(altered)


if __name__ == "__main__":
    unittest.main()
