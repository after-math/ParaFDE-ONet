"""Regression test for the multiprocessing import collision seen on the server."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest


PROJECT = Path(__file__).resolve().parents[1]


class SpawnImportTests(unittest.TestCase):
    def test_spawn_keeps_benchmark_reporting_isolated(self) -> None:
        environment = dict(os.environ)
        existing = environment.get("PYTHONPATH")
        paths = [str(PROJECT / "code")]
        if existing:
            paths.append(existing)
        environment["PYTHONPATH"] = os.pathsep.join(paths)
        completed = subprocess.run(
            [sys.executable, str(PROJECT / "tests" / "spawn_probe.py")],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("spawn import isolation passed", completed.stdout)


if __name__ == "__main__":
    unittest.main()
