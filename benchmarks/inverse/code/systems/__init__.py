"""System-specific adapters around the three untouched source projects."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def build_adapter(
    system_key: str,
    config: Mapping[str, Any],
    benchmark_root: Path,
    source_root_override: Path | None = None,
):
    """Build exactly one adapter in the current process.

    Each benchmark invocation handles one system, which avoids collisions among
    the three legacy source trees whose modules are all named ``data``,
    ``equation`` and ``model``.
    """
    if system_key == "v2d":
        from .variable_delay_adapter import VariableDelayAdapter

        return VariableDelayAdapter(config, benchmark_root, source_root_override)
    if system_key == "n4d":
        from .nicholson_adapter import NicholsonAdapter

        return NicholsonAdapter(config, benchmark_root, source_root_override)
    if system_key == "sei":
        from .delayed_sei_adapter import DelayedSEIAdapter

        return DelayedSEIAdapter(config, benchmark_root, source_root_override)
    raise KeyError(f"unknown system: {system_key}")

