from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parents[1] / "experiments" / "nicholson"
SOURCE_CODE = SOURCE / "four_methods_normalized_sensitivity_5seeds" / "code"
sys.path.insert(0, str(SOURCE_CODE))
SPEC = importlib.util.spec_from_file_location("feature_reuse_benchmark", ROOT / "code" / "benchmark.py")
BENCHMARK = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BENCHMARK)

from model import Operator4D


class FeatureReuseBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.device = torch.device("cpu")
        self.histories = torch.randn(2, 4, 7)
        self.times = torch.linspace(0.0, 1.0, 5)
        self.observations = torch.randn(2, 5, 4)
        self.states = torch.arange(4)
        self.candidate = torch.tensor([4.5, 1.0])

    def build(self, method: str) -> Operator4D:
        model = Operator4D(
            model_type=method,
            history_sensors=7,
            horizon=1.0,
            latent_dim=8,
            history_width=12,
            history_depth=2,
            parameter_width=12,
            parameter_depth=2,
            trunk_width=12,
            trunk_depth=2,
            activation="gelu",
            fourier_modes=2,
            initialization_seed=11,
            parameter_bounds=[[3.0, 7.0], [0.8, 1.4]],
        )
        model.eval().requires_grad_(False)
        return model

    def test_cached_paths_match_full_forward_and_gradient(self) -> None:
        for method in BENCHMARK.METHOD_ORDER:
            model = self.build(method)
            cache = BENCHMARK.build_cache(model, self.histories, self.times)
            cached_output, _, cached_gradient = BENCHMARK.loss_and_parameter_gradient(
                model,
                self.histories,
                self.observations,
                self.states,
                cache,
                self.candidate,
            )
            full_output, full_gradient = BENCHMARK.full_loss_and_gradient(
                model,
                self.histories,
                self.times,
                self.observations,
                self.states,
                self.candidate,
            )
            torch.testing.assert_close(cached_output, full_output, rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(cached_gradient, full_gradient, rtol=5e-4, atol=2e-6)

    def test_only_pfdeonet_caches_history_features(self) -> None:
        for method in BENCHMARK.METHOD_ORDER:
            cache = BENCHMARK.build_cache(self.build(method), self.histories, self.times)
            self.assertIn("trunk", cache)
            self.assertEqual("history" in cache, method == "separate_parameter_shared")

    def test_pfdeonet_evaluates_one_shared_parameter_feature(self) -> None:
        model = self.build("separate_parameter_shared")
        cache = BENCHMARK.build_cache(model, self.histories, self.times)
        observed_batch_sizes = []

        def capture_batch(_module, inputs):
            observed_batch_sizes.append(int(inputs[0].shape[0]))

        hook = model.parameter_branch.register_forward_pre_hook(capture_batch)
        try:
            BENCHMARK.loss_and_parameter_gradient(
                model,
                self.histories,
                self.observations,
                self.states,
                cache,
                self.candidate,
            )
        finally:
            hook.remove()
        self.assertEqual(observed_batch_sizes, [1])

    def test_nested_parameter_prefixes_are_reproducible(self) -> None:
        config = BENCHMARK.load_json(ROOT / "configs" / "benchmark.json")
        BENCHMARK.validate_config(config, smoke=False)
        self.assertEqual(
            config["queries"]["counts"],
            [1, 10, 100, 1000, 3000, 10000, 100000],
        )

    def test_timing_summary_and_plot(self) -> None:
        import tempfile

        raw = []
        for method_index, method in enumerate(BENCHMARK.METHOD_ORDER, start=1):
            for count in (1, 2):
                for repeat in (1, 2):
                    raw.append(
                        {
                            "model_type": method,
                            "method": method,
                            "query_count": count,
                            "repeat": repeat,
                            "cache_seconds": 0.001 * method_index,
                            "first_query_seconds": 0.002 * method_index,
                            "steady_seconds": 0.01 * method_index * count + repeat * 1e-4,
                            "seconds_per_query": 0.01 * method_index,
                            "peak_memory_bytes": 1000 * method_index,
                            "cold_checksum": 1.0,
                            "steady_checksum": 1.0,
                        }
                    )
        names = {method: method for method in BENCHMARK.METHOD_ORDER}
        summary = BENCHMARK.summarize_timings(raw, names)
        self.assertEqual(len(summary), 6)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            BENCHMARK.plot_results(summary, names, output, ["png"], 80)
            self.assertTrue((output / "figures" / "feature_reuse_timing.png").is_file())


if __name__ == "__main__":
    unittest.main()
