#!/usr/bin/env python3
"""Portable entry points for the ParaFDEONet research code."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = {
    "competition": "experiments/competition/four_methods_sensitivity_5seeds",
    "sei": "experiments/sei/stratified_history_balanced_sensitivity",
    "nicholson": "experiments/nicholson/four_methods_normalized_sensitivity_5seeds",
    "smart_grid": "experiments/smart_grid",
    "cstr": "experiments/cstr",
}
ENTRIES = {
    "inverse": "benchmarks/inverse/code/pipeline.py",
    "inverse-lm": "benchmarks/inverse/frozen_lm_ablation/run.py",
    "inverse-compare": "benchmarks/inverse/frozen_lm_ablation/compare.py",
    "jacobian": "benchmarks/sensitivity_jacobian/code/benchmark.py",
    "feature-reuse": "benchmarks/feature_reuse/code/benchmark.py",
    "grid-screen": "experiments/smart_grid/code/scripts/run_survivability.py",
    "grid-multiseed": "experiments/smart_grid/code/scripts/run_multiseed_survivability.py",
    "grid-fresh": "experiments/smart_grid/code/scripts/run_fresh_history_audit.py",
    "grid-gbt": "experiments/smart_grid/code/scripts/run_literature_gbt.py",
    "cstr-online": "experiments/cstr/run_industrial_benchmark.py",
    "cstr-pilot": "experiments/cstr/run_industrial_pilot.py",
}
for _name in ("competition", "sei", "nicholson"):
    ENTRIES[f"speed-{_name}"] = (
        f"experiments/{_name}/separate_parameter_speed_benchmark/code/benchmark.py"
    )


def execute(command: list[str], cwd: Path = ROOT) -> None:
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("MPLBACKEND", "Agg")
    # Public entry points are local by default. No author-specific notification endpoint.
    env["WANDB_MODE"] = "disabled"
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def run_training(args: argparse.Namespace) -> None:
    names = list(EXPERIMENTS) if args.system == "all" else [args.system]
    for name in names:
        project = ROOT / EXPERIMENTS[name]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = (
            args.output_dir.expanduser().resolve()
            if args.output_dir
            else ROOT / "outputs" / f"{args.command}_{name}_{stamp}"
        )
        if args.system == "all" and args.output_dir:
            output = output / name
        config = project / "configs/experiment.json"
        if name == "cstr":
            command = [sys.executable, str(project / "run_pipeline.py"),
                       "--config", str(config), "--output-dir", str(output),
                       "--workers", str(args.cpus), "--device", args.device]
            if args.command == "smoke":
                command.append("--smoke")
            if args.iterations is not None:
                command.extend(["--iterations", str(args.iterations)])
        else:
            device = args.device.removeprefix("cuda:")
            command = [sys.executable, str(project / "code/scripts/pipeline.py"),
                       "--config", str(config), "--output-dir", str(output),
                       "--devices", device, "--processes", "1", "--cpus", str(args.cpus),
                       "--wandb-mode", "disabled", "--no-ntfy"]
            seed = args.seed
            if args.command == "smoke":
                command.append("--smoke-only")
                if seed is None:
                    seed = json.loads(config.read_text())["randomness"]["training_seeds"][0]
            if seed is not None:
                command.extend(["--seed", str(seed)])
            if args.iterations is not None:
                raise SystemExit("--iterations is supported only for cstr; edit a copied configuration for other systems.")
        if args.resume:
            command.append("--resume")
        execute(command, project)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="Show manuscript experiment and advanced entry paths")
    for mode in ("smoke", "train"):
        p = sub.add_parser(mode, help="Reduced CPU check" if mode == "smoke" else "Full configured training")
        p.add_argument("system", choices=[*EXPERIMENTS, "all"] if mode == "smoke" else list(EXPERIMENTS))
        p.add_argument("--device", default="cpu" if mode == "smoke" else "cuda:0")
        p.add_argument("--cpus", type=int, default=1)
        p.add_argument("--seed", type=int)
        p.add_argument("--iterations", type=int)
        p.add_argument("--output-dir", type=Path)
        p.add_argument("--resume", action="store_true")
    p = sub.add_parser("entry", help="Forward all remaining arguments to an original entry point")
    p.add_argument("name", choices=list(ENTRIES))
    p.add_argument("arguments", nargs=argparse.REMAINDER)
    sub.add_parser("check", help="Check syntax, configurations, release files and provenance")
    p = sub.add_parser("test", help="Run isolated existing experiment test suites")
    p.add_argument("suite", nargs="?", default="all", choices=[*EXPERIMENTS, "inverse", "jacobian", "feature-reuse", "all"])
    p = sub.add_parser("figures", help="Redraw paper figures from bundled source data")
    p.add_argument("name", nargs="?", default="all", choices=["all", "representative", "jacobian", "smart-grid", "feature-reuse", "pinndde", "cstr"])
    args = parser.parse_args()
    if args.command == "list":
        for name, path in EXPERIMENTS.items():
            print(f"{name:16} {path}")
        for name, path in ENTRIES.items():
            print(f"{name:16} {path}")
    elif args.command in {"smoke", "train"}:
        if args.cpus < 1:
            parser.error("--cpus must be positive")
        if args.resume and args.output_dir is None:
            parser.error("--resume requires --output-dir")
        run_training(args)
    elif args.command == "entry":
        remainder = args.arguments
        if remainder[:1] == ["--"]:
            remainder = remainder[1:]
        if args.name in {"cstr-online", "cstr-pilot"}:
            config_root = ROOT / "experiments/cstr/configs"
            defaults = ["--experiment-config", str(config_root / "experiment.json")]
            kind = "industrial" if args.name == "cstr-online" else "pilot"
            defaults.extend([f"--{kind}-config", str(config_root / f"{kind}.json")])
            remainder = defaults + remainder
        execute([sys.executable, str(ROOT / ENTRIES[args.name]), *remainder])
    elif args.command == "check":
        execute([sys.executable, str(ROOT / "scripts/check_release.py")])
    elif args.command == "test":
        execute([sys.executable, str(ROOT / "scripts/run_tests.py"), args.suite])
    elif args.command == "figures":
        execute([sys.executable, str(ROOT / "scripts/render_figures.py"), args.name])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode)
