"""Copy completed formal forward-run inputs into the benchmark artifact layout."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SEEDS = {"competition": 20261005, "sei": 20260903, "nicholson": 20260901}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system", choices=list(SEEDS))
    parser.add_argument("variant", choices=["with_sensitivity", "without_sensitivity"])
    parser.add_argument("run_dir", type=Path, help="Completed forward pipeline directory containing full/")
    args = parser.parse_args()
    full = args.run_dir.expanduser().resolve() / "full"
    destination = ROOT / "artifacts" / args.system / args.variant
    if destination.exists():
        raise SystemExit(f"Destination already exists; use a new checkout or move the existing directory first: {destination}")
    config_path = full / "resolved_config.json"
    test_path = full / "dataset/test.npz"
    checkpoint_files = sorted((full / "methods").glob("*/seed_*/best_model.pt"))
    needed = full / "methods/separate_parameter_shared" / f"seed_{SEEDS[args.system]}" / "best_model.pt"
    for source in [config_path, test_path, needed]:
        if not source.is_file():
            raise SystemExit(f"Required formal input is missing: {source}")
    config = json.loads(config_path.read_text())
    if int(config["data"]["history_sensors"]) != 101 or int(config["operator"]["iterations"]) < 1000:
        raise SystemExit("This appears to be a smoke run; benchmarks require a full-size trained model.")
    transfers = [(config_path, destination / "resolved_config.json"),
                 (test_path, destination / "dataset/test.npz")]
    transfers.extend((p, destination / p.relative_to(full)) for p in checkpoint_files)
    records = []
    for source, target in transfers:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        with target.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        records.append({"file": str(target.relative_to(destination)), "bytes": target.stat().st_size, "sha256": digest})
    (destination / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"Prepared {len(records)} files in {destination}")


if __name__ == "__main__":
    main()
