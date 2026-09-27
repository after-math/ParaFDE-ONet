"""Fail-fast helpers for isolated extensions of the audited parent project."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def parent_code_path(current_file: str, relative: str) -> Path:
    """Resolve a source file in the untouched parent project's ``code`` tree."""
    current = Path(current_file).resolve()
    experiment_root = next(
        (
            parent
            for parent in current.parents
            if parent.name == "four_methods_sensitivity_5seeds"
        ),
        None,
    )
    if experiment_root is None:
        raise RuntimeError(f"cannot locate sensitivity experiment root from {current}")
    project_root = experiment_root.parent
    path = project_root / "code" / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def patched_source(
    source_path: Path,
    replacements: Iterable[tuple[str, str]],
) -> str:
    """Apply exact one-occurrence replacements and reject parent-source drift."""
    source = source_path.read_text(encoding="utf-8")
    for old, new in replacements:
        count = source.count(old)
        if count != 1:
            raise RuntimeError(
                f"expected one exact patch anchor in {source_path}, found {count}: "
                f"{old[:100]!r}"
            )
        source = source.replace(old, new, 1)
    return source


def execute_source(source: str, namespace: dict[str, object], filename: str) -> None:
    """Execute validated source as the current module for spawn-safe imports."""
    exec(compile(source, filename, "exec"), namespace)
