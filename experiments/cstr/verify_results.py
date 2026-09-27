"""Verify that the packaged supplementary-experiment results are complete."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULT = ROOT.parents[1] / "results/paper/cstr/industrial_lm_20events"


def main() -> None:
    summary_path = RESULT / "industrial_summary.json"
    inversion_path = RESULT / "inversion_results.csv"
    event_dir = RESULT / "events"
    if not summary_path.is_file() or not inversion_path.is_file():
        raise FileNotFoundError("formal summary or inversion table is absent")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with inversion_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    events = sorted(event_dir.glob("event_*.json"))

    if len(rows) != 252:
        raise AssertionError(f"expected 252 inversion records, found {len(rows)}")
    if len(events) != 20:
        raise AssertionError(f"expected 20 event files, found {len(events)}")
    if summary["aggregate"]["event_count"] != 20:
        raise AssertionError("aggregate event count is not 20")

    numeric_fields = (
        "time_seconds",
        "normalized_parameter_error",
        "estimated_k",
        "estimated_kappa",
    )
    for row in rows:
        for field in numeric_fields:
            if not math.isfinite(float(row[field])):
                raise FloatingPointError(f"non-finite {field} in inversion table")

    lm_rows = [
        row
        for row in rows
        if int(row["event"]) >= 0 and row["method"] == "ParaFDEONet LM"
    ]
    if len(lm_rows) != 80:
        raise AssertionError(f"expected 80 regular LM inversions, found {len(lm_rows)}")
    if not all(row["lm_early_stopped"] == "True" for row in lm_rows):
        raise AssertionError("at least one LM inversion did not converge early")

    para = summary["aggregate"]["pipelines"]["ParaFDEONet"]
    if para["unsafe_recommendation_count"] != 0:
        raise AssertionError("packaged ParaFDEONet result contains an unsafe recommendation")
    print("RESULTS_VERIFIED")
    print(f"inversion_records={len(rows)}")
    print(f"event_files={len(events)}")
    print(f"median_cycle_seconds={para['cycle_time_median_seconds']:.6f}")
    print(f"median_cycle_speedup={summary['aggregate']['median_cycle_speedup']:.6f}")


if __name__ == "__main__":
    main()
