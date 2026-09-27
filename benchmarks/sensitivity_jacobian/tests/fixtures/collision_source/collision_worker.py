"""Small spawn-based worker used by the import collision regression test."""

from __future__ import annotations

import multiprocessing as mp


def square(value: int) -> int:
    return value * value


def run_spawn() -> list[int]:
    context = mp.get_context("spawn")
    with context.Pool(processes=2) as pool:
        return pool.map(square, [1, 2, 3])
