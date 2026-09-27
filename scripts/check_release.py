"""Validate a clean source release using only the Python standard library."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {".git", ".venv", "__pycache__", ".pytest_cache", "outputs", "artifacts", "wandb"}

def files():
    for path in ROOT.rglob("*"):
        if path.is_file() and not any(p in IGNORED for p in path.relative_to(ROOT).parts):
            yield path

def main() -> None:
    errors = []
    paths = list(files())
    for path in paths:
        relative = path.relative_to(ROOT)
        if path.stat().st_size > 95 * 1024**2:
            errors.append(f"Oversized release file: {relative}")
        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(), filename=str(relative))
            except (SyntaxError, UnicodeError) as error:
                errors.append(str(error))
        if path.suffix == ".json":
            try:
                config = json.loads(path.read_text())
                if isinstance(config, dict):
                    for key in ("wandb", "ntfy"):
                        if isinstance(config.get(key), dict) and config[key].get("enabled"):
                            errors.append(f"Tracking enabled in {relative}")
            except (ValueError, UnicodeError) as error:
                errors.append(f"{relative}: {error}")
        if path.suffix in {".py", ".json", ".md", ".txt", ".yml", ".cff"}:
            text = path.read_text()
            # Match real workstation paths, not prose such as train/data/physics.
            if re.search(r"/(?:Users|home|root|data[0-9]+)/[A-Za-z0-9_-]+/", text):
                errors.append(f"Personal absolute path: {relative}")
            if re.search(r"https://ntfy[.]sh/[A-Za-z0-9_-]+", text):
                errors.append(f"Personal notification endpoint: {relative}")
    manifest = json.loads((ROOT / "docs/source_manifest.json").read_text())
    for item in manifest:
        path = ROOT / item["file"]
        if not path.is_file():
            errors.append(f"Missing source file: {item['file']}")
        elif "release_sha256" in item and hashlib.sha256(path.read_bytes()).hexdigest() != item["release_sha256"]:
            errors.append(f"Release checksum changed: {item['file']}")
    for config in (ROOT / "benchmarks/inverse/configs").glob("*.json"):
        data = json.loads(config.read_text())
        for key in ("source_project", "training_resolved_config"):
            if key in data and not (config.parent.parent / data[key]).exists():
                errors.append(f"Unresolved {key} in {config.relative_to(ROOT)}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Release checks passed: {len(paths)} files, {sum(p.suffix == '.py' for p in paths)} Python modules, {len(manifest)} source records.")

if __name__ == "__main__":
    main()
