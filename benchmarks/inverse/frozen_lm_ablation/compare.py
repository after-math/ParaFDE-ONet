#!/usr/bin/env python3
"""Create a paired Frozen-LM versus Frozen-Adam ablation summary."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np  # noqa: E402

from common.io_utils import write_csv, write_json  # noqa: E402


def _read(path: Path) -> dict[tuple[int, int, float], dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {
        (
            int(row["panel_index"]),
            int(row["case_index"]),
            float(row["noise_standard_deviation"]),
        ): row
        for row in rows
    }
    if len(result) != len(rows):
        raise RuntimeError(f"duplicate paired keys in {path}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adam-csv", type=Path, required=True)
    parser.add_argument("--lm-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261141)
    args = parser.parse_args()

    adam = _read(args.adam_csv.expanduser().resolve())
    lm = _read(args.lm_csv.expanduser().resolve())
    if set(adam) != set(lm):
        raise RuntimeError("Adam and LM CSV files do not contain identical paired requests")
    for key in adam:
        for field in (
            "system_key",
            "panel_archive_sha256",
            "checkpoint_sha256",
            "history_count",
            "history_indices_used",
            "noise_standard_deviation",
        ):
            if adam[key].get(field) != lm[key].get(field):
                raise RuntimeError(f"unpaired field {field} at request {key}")

    grouped: dict[float, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    for (panel, _, noise), adam_row in adam.items():
        lm_row = lm[(panel, int(adam_row["case_index"]), noise)]
        grouped[noise].append(
            (
                panel,
                float(adam_row["parameter_normalized_rmse"]),
                float(lm_row["parameter_normalized_rmse"]),
                float(adam_row["warm_online_seconds"]),
                float(lm_row["warm_online_seconds"]),
            )
        )

    rng = np.random.default_rng(args.bootstrap_seed)
    summary: list[dict[str, Any]] = []
    for noise, values in sorted(grouped.items()):
        by_panel: dict[int, list[tuple[float, float]]] = defaultdict(list)
        adam_times = []
        lm_times = []
        for panel, adam_error, lm_error, adam_time, lm_time in values:
            by_panel[panel].append((adam_error, lm_error))
            adam_times.append(adam_time)
            lm_times.append(lm_time)
        panel_pairs = np.asarray(
            [
                (
                    np.mean([value[0] for value in pairs]),
                    np.mean([value[1] for value in pairs]),
                )
                for _, pairs in sorted(by_panel.items())
            ],
            dtype=np.float64,
        )
        differences = panel_pairs[:, 1] - panel_pairs[:, 0]
        draws = np.empty(args.bootstrap_repetitions, dtype=np.float64)
        for index in range(args.bootstrap_repetitions):
            selected = rng.integers(0, len(panel_pairs), size=len(panel_pairs))
            draws[index] = float(np.mean(differences[selected]))
        adam_error = float(np.mean(panel_pairs[:, 0]))
        lm_error = float(np.mean(panel_pairs[:, 1]))
        adam_time = float(np.median(adam_times))
        lm_time = float(np.median(lm_times))
        summary.append(
            {
                "noise_standard_deviation": float(noise),
                "paired_panel_count": len(panel_pairs),
                "paired_request_count": len(values),
                "adam_parameter_nrmse_panel_macro_mean": adam_error,
                "lm_parameter_nrmse_panel_macro_mean": lm_error,
                "lm_minus_adam_parameter_nrmse": float(np.mean(differences)),
                "lm_minus_adam_ci95_lower": float(np.percentile(draws, 2.5)),
                "lm_minus_adam_ci95_upper": float(np.percentile(draws, 97.5)),
                "lm_to_adam_parameter_nrmse_ratio": float(
                    lm_error / max(adam_error, 1.0e-15)
                ),
                "adam_warm_online_seconds_median": adam_time,
                "lm_warm_online_seconds_median": lm_time,
                "adam_to_lm_speedup": float(adam_time / max(lm_time, 1.0e-15)),
            }
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "paired_optimizer_ablation.csv", summary)
    write_json(
        output_dir / "paired_optimizer_ablation.json",
        {
            "adam_csv": str(args.adam_csv.expanduser().resolve()),
            "lm_csv": str(args.lm_csv.expanduser().resolve()),
            "bootstrap_repetitions": int(args.bootstrap_repetitions),
            "bootstrap_seed": int(args.bootstrap_seed),
            "summary": summary,
        },
    )
    print(f"Wrote paired ablation summary to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
