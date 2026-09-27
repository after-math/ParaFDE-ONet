"""Delayed SEI adapter using the checkpoint-compatible High-75 panel."""

from __future__ import annotations

import numpy as np

from .base import BaseSystemAdapter


class DelayedSEIAdapter(BaseSystemAdapter):
    config_class_name = "DelayedSEIConfig"

    def sample_panel_histories(
        self, count: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, list[str]]:
        ranges = list(self.config["high75_level_ranges"])
        names = list(self.config["high75_names"])
        if count != len(ranges):
            if count < 1 or count > len(ranges):
                raise ValueError(
                    "SEI smoke/formal history count must lie between 1 and 8"
                )
            ranges = ranges[:count]
            names = names[:count]
        rows = []
        for level_ranges in ranges:
            sampled = self.data_module.sample_histories(
                1,
                self.history_grid(),
                tuple(tuple(float(value) for value in row) for row in level_ranges),
                tuple(float(value) for value in self.data_config["history_sigmas"]),
                float(self.data_config["history_length_scale"]),
                tuple(
                    tuple(float(value) for value in row)
                    for row in self.data_config["history_bounds"]
                ),
                float(self.data_config["history_total_upper"]),
                rng,
            )
            rows.append(sampled[0])
        return np.stack(rows).astype(np.float32), [str(value) for value in names]

