"""Smoke test for the manuscript figure pipeline."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "code"))

from reference import write_rows  # noqa: E402
from benchmark_reporting import make_all_figures  # noqa: E402


class ReportingTests(unittest.TestCase):
    def test_all_figures_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            system_dir = output / "competition"
            jacobian_rows = []
            gradient_rows = []
            convergence_rows = []
            for variant, scale in (
                ("without_sensitivity", 1.0),
                ("with_sensitivity", 0.5),
            ):
                for parameter in ("r1", "r2"):
                    for case_index in range(4):
                        jacobian_rows.append(
                            {
                                "parameter": parameter,
                                "variant": variant,
                                "relative_error": scale * (0.1 + 0.01 * case_index),
                            }
                        )
                for case_index in range(4):
                    gradient_rows.append(
                        {
                            "variant": variant,
                            "jacobian_only_gradient_cosine": 0.4
                            + 0.1 * case_index
                            + (0.1 if variant == "with_sensitivity" else 0.0),
                        }
                    )
            for study in ("finite_difference_step", "solver_step"):
                for comparison_index in range(2):
                    convergence_rows.append(
                        {
                            "study": study,
                            "aggregate_relative_change": 0.02
                            / (comparison_index + 1),
                        }
                    )
            write_rows(system_dir / "per_case_jacobian_metrics.csv", jacobian_rows)
            write_rows(system_dir / "gradient_direction.csv", gradient_rows)
            write_rows(system_dir / "convergence.csv", convergence_rows)
            make_all_figures(output, ["competition"])
            for stem in (
                "jacobian_error_comparison",
                "gradient_cosine_comparison",
                "step_convergence",
            ):
                self.assertTrue((output / "figures" / f"{stem}.pdf").is_file())
                self.assertTrue((output / "figures" / f"{stem}.png").is_file())


if __name__ == "__main__":
    unittest.main()
