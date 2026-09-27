"""Redraw the archived paper plots without pretrained neural networks."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = {"representative": "representative.py", "jacobian": "jacobian.py", "cstr": "cstr.py",
           "smart-grid": "smart_grid.py", "feature-reuse": "feature_reuse.py", "pinndde": "pinndde.py"}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", nargs="?", default="all", choices=["all", *SCRIPTS])
    args = parser.parse_args()
    names = list(SCRIPTS) if args.name == "all" else [args.name]
    env = dict(os.environ, MPLBACKEND="Agg")
    for name in names:
        subprocess.run([sys.executable, str(ROOT / "scripts/figures" / SCRIPTS[name])],
                       cwd=ROOT, env=env, check=True)
    print(ROOT / "outputs/paper_figures")

if __name__ == "__main__":
    main()
