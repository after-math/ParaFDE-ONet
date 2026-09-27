"""Runtime helpers for loading audited experiment code without module collisions."""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Iterator

import numpy as np
import torch


SOURCE_MODULE_NAMES = (
    "_source_patch",
    "equation",
    "data",
    "model",
    "training",
    "reporting",
    "wandb_tracker",
)


def load_json(path: Path) -> dict[str, Any]:
    """Read a JSON object and reject missing or non-object content."""
    if not path.is_file():
        raise FileNotFoundError(path)
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return values


@contextmanager
def source_environment(code_dir: Path) -> Iterator[dict[str, Any]]:
    """Import one source tree under its original absolute module names.

    The three audited projects all use top-level names such as ``model`` and
    ``equation``.  This context removes those names temporarily, inserts exactly
    one source directory, and restores the previous interpreter state afterward.
    Loaded model instances remain valid because their classes retain references to
    the defining module dictionaries.
    """
    code_dir = code_dir.resolve()
    if not (code_dir / "model.py").is_file() or not (code_dir / "equation.py").is_file():
        raise FileNotFoundError(f"invalid source code directory: {code_dir}")
    saved_modules = {name: sys.modules.get(name) for name in SOURCE_MODULE_NAMES}
    saved_path = list(sys.path)
    for name in SOURCE_MODULE_NAMES:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(code_dir))
    importlib.invalidate_caches()
    loaded: dict[str, Any] = {}
    try:
        loaded["equation"] = importlib.import_module("equation")
        loaded["data"] = importlib.import_module("data")
        loaded["model"] = importlib.import_module("model")
        yield loaded
    finally:
        for name in SOURCE_MODULE_NAMES:
            sys.modules.pop(name, None)
        for name, module in saved_modules.items():
            if module is not None:
                sys.modules[name] = module
        sys.path[:] = saved_path
        importlib.invalidate_caches()


def load_model_from_source(
    code_dir: Path, checkpoint_path: Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load one checkpoint with the exact model module that created it."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    with source_environment(code_dir) as modules:
        model, checkpoint = modules["model"].load_operator_checkpoint(
            checkpoint_path, device
        )
    model.eval()
    if str(checkpoint.get("model_type")) != "separate_parameter_shared":
        raise RuntimeError(f"unexpected model type in {checkpoint_path}")
    return model, checkpoint


def checkpoint_identity(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Return the immutable checkpoint fields recorded in the result archive."""
    return {
        "network_format": checkpoint.get("network_format"),
        "model_type": checkpoint.get("model_type"),
        "training_seed": checkpoint.get("training_seed"),
        "iteration": checkpoint.get("iteration"),
        "best_validation": checkpoint.get("best_validation"),
        "state_scalar_count_including_buffers": int(
            sum(
                value.numel()
                for value in checkpoint["model_state_dict"].values()
                if torch.is_tensor(value)
            )
        ),
    }


def validate_checkpoint_pair(
    baseline: dict[str, Any], improved: dict[str, Any], expected_seed: int
) -> None:
    """Reject seed, method, or architecture mismatches in a paired comparison."""
    for name, checkpoint in (("baseline", baseline), ("improved", improved)):
        if int(checkpoint.get("training_seed", -1)) != expected_seed:
            raise RuntimeError(f"{name} checkpoint does not use seed {expected_seed}")
        if checkpoint.get("model_type") != "separate_parameter_shared":
            raise RuntimeError(f"{name} checkpoint is not PFDEONet")
    baseline_config = dict(baseline.get("model_config", {}))
    improved_config = dict(improved.get("model_config", {}))
    ignored = {"parameter_bounds"}
    comparable_baseline = {k: v for k, v in baseline_config.items() if k not in ignored}
    comparable_improved = {k: v for k, v in improved_config.items() if k not in ignored}
    if comparable_baseline != comparable_improved:
        differing = sorted(
            key
            for key in set(comparable_baseline) | set(comparable_improved)
            if comparable_baseline.get(key) != comparable_improved.get(key)
        )
        raise RuntimeError(f"checkpoint architecture differs at: {differing}")


def write_environment(path: Path) -> None:
    """Save a compact software and hardware record."""
    cuda_name = None
    if torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(torch.cuda.current_device())
    values = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device": cuda_name,
        "cpu_count": os.cpu_count(),
    }
    path.write_text(json.dumps(values, indent=2), encoding="utf-8")
