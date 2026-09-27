"""Command-line pipeline for formal CSTR forward training."""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch

from .data import generate_or_load_dataset
from .training import save_json, train_operator


LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--data-cache",
        type=Path,
        default=None,
        help="optional compatible cache to reuse without regenerating DDE trajectories",
    )
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run(args: argparse.Namespace) -> None:
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.iterations is not None and args.iterations < 1:
        raise ValueError("iterations override must be positive")
    config = copy.deepcopy(load_config(args.config))
    if args.iterations is not None:
        config["operator"]["iterations"] = int(args.iterations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(args.output_dir / "pipeline.log"),
            logging.StreamHandler(),
        ],
    )
    save_json(args.output_dir / "resolved_config.json", config)
    save_json(
        args.output_dir / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
        },
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    save_json(
        args.output_dir / "pipeline_status.json",
        {"status": "running", "device": str(device), "smoke": bool(args.smoke)},
    )
    started = time.perf_counter()
    try:
        cache_dir = (
            args.data_cache
            if args.data_cache is not None
            else args.output_dir / "data_cache"
        )
        splits, metadata = generate_or_load_dataset(
            config,
            cache_dir,
            workers=1 if args.smoke else args.workers,
            smoke=bool(args.smoke),
        )
        save_json(args.output_dir / "data_diagnostics.json", metadata)
        save_json(
            args.output_dir / "data_cache_source.json",
            {"path": str(cache_dir.resolve()), "external": args.data_cache is not None},
        )
        summary = train_operator(
            splits,
            config,
            args.output_dir / "forward",
            device,
            smoke=bool(args.smoke),
            resume=bool(args.resume),
        )
        save_json(
            args.output_dir / "pipeline_status.json",
            {
                "status": "completed",
                "passed": bool(summary["passed"]),
                "elapsed_seconds": time.perf_counter() - started,
                "summary": summary,
            },
        )
        LOGGER.info("PIPELINE_COMPLETED passed=%s", summary["passed"])
    except Exception as error:
        save_json(
            args.output_dir / "pipeline_status.json",
            {
                "status": "failed",
                "elapsed_seconds": time.perf_counter() - started,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            },
        )
        LOGGER.exception("PIPELINE_FAILED")
        raise


def main() -> None:
    run(build_parser().parse_args())
