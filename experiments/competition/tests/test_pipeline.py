"""Regression tests for five-seed selection and round-robin scheduling."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "code"))

from model import MODEL_TYPES
from scripts.pipeline import build_round_robin_queues, validate_config


class PipelineTests(unittest.TestCase):
    """Protect the exact five-seed by four-method execution protocol."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the formal JSON configuration used by scheduling tests.

        The method has no explicit input and returns ``None``.  It stores the parsed
        mapping on the class; for example it contains five training seeds.  unittest
        calls it once before the tests below.
        """
        cls.config = json.loads(
            (PROJECT_DIR / "configs" / "experiment.json").read_text(encoding="utf-8")
        )

    def test_formal_config_has_five_unique_seeds(self) -> None:
        """Verify that the formal config passes exact multi-seed validation.

        No input is required and the return is ``None``.  The assertion checks five
        unique seed integers and invokes ``validate_config``; unittest uses it to
        catch accidental regression to one seed.
        """
        validate_config(self.config)
        seeds = self.config["randomness"]["training_seeds"]
        self.assertEqual(len(seeds), 5)
        self.assertEqual(len(set(seeds)), 5)

    def test_twenty_jobs_on_eight_workers(self) -> None:
        """Verify complete, duplicate-free round-robin assignment on eight GPUs.

        No input is required and the return is ``None``.  Twenty method-seed pairs
        produce queue lengths ``[3,3,3,3,2,2,2,2]`` and every pair occurs once.
        unittest calls it as the formal scheduling regression test.
        """
        seeds = self.config["randomness"]["training_seeds"]
        jobs = [(method, seed) for seed in seeds for method in MODEL_TYPES]
        queues = build_round_robin_queues(jobs, 8)
        self.assertEqual([len(queue) for queue in queues], [3, 3, 3, 3, 2, 2, 2, 2])
        flattened = [job for queue in queues for job in queue]
        self.assertEqual(len(flattened), 20)
        self.assertEqual(set(flattened), set(jobs))

    def test_one_selected_seed_uses_four_workers(self) -> None:
        """Verify the optional one-seed path still schedules all four methods.

        No input is required and the return is ``None``.  Four jobs on four workers
        create four singleton queues, matching a targeted seed run.  unittest calls
        it to protect debugging and补-run semantics.
        """
        seed = self.config["randomness"]["training_seeds"][0]
        jobs = [(method, seed) for method in MODEL_TYPES]
        queues = build_round_robin_queues(jobs, 4)
        self.assertEqual([len(queue) for queue in queues], [1, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()
