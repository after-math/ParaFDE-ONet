"""Run each original test suite in a separate process to isolate module names."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SUITES = {
    "competition": ["experiments/competition/four_methods_sensitivity_5seeds", "experiments/competition/four_methods_normalized_no_sensitivity_5seeds"],
    "sei": ["experiments/sei/stratified_history_balanced_sensitivity", "experiments/sei/stratified_history_balanced_normalized_no_sensitivity"],
    "nicholson": ["experiments/nicholson/four_methods_normalized_sensitivity_5seeds"],
    "smart_grid": ["experiments/smart_grid"],
    "cstr": ["experiments/cstr"],
    "inverse": ["benchmarks/inverse"],
    "jacobian": ["benchmarks/sensitivity_jacobian"],
    "feature-reuse": ["benchmarks/feature_reuse"],
}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=[*SUITES, "all"], default="all", nargs="?")
    args = parser.parse_args()
    names = list(SUITES) if args.suite == "all" else [args.suite]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLBACKEND="Agg", WANDB_MODE="disabled")
    failures = []
    for name in names:
        for relative in SUITES[name]:
            path = ROOT / relative
            targets = ["tests"]
            if name == "inverse":
                targets.append("frozen_lm_ablation/test_runtime.py")
            print(f"\n{name}: {relative}", flush=True)
            result = subprocess.run([sys.executable, "-m", "pytest", "-q", *targets], cwd=path, env=env)
            if result.returncode:
                failures.append(relative)
    if failures:
        print("Failed suites:", ", ".join(failures))
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
