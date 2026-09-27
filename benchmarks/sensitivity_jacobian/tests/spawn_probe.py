"""Executable probe whose child inherits a conflicting source directory first."""

from __future__ import annotations

from pathlib import Path
import sys

from benchmark_reporting import make_all_figures


EXTERNAL = Path(__file__).resolve().parent / "fixtures" / "collision_source"
sys.path.insert(0, str(EXTERNAL))

from collision_worker import run_spawn


if __name__ == "__main__":
    assert callable(make_all_figures)
    assert run_spawn() == [1, 4, 9]
    print("spawn import isolation passed")
