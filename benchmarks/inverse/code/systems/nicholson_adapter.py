"""Four-patch Nicholson adapter."""

from __future__ import annotations

import numpy as np

from .base import BaseSystemAdapter


class NicholsonAdapter(BaseSystemAdapter):
    config_class_name = "NicholsonConfig"

    def sample_panel_histories(
        self, count: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, list[str]]:
        histories = self.data_module.sample_histories(
            count,
            self.history_grid(),
            tuple(float(value) for value in self.data_config["history_mean"]),
            float(self.data_config["history_sigma"]),
            float(self.data_config["history_length_scale"]),
            tuple(float(value) for value in self.data_config["history_bounds"]),
            rng,
        )
        return histories, [f"training_distribution_{index:02d}" for index in range(count)]

