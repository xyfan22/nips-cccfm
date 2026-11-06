"""Download the public input files listed in data/sources.json with curl resume."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def verify(path, entry):
    if path.stat().st_size != entry["size_bytes"]:
        raise ValueError(f"Size mismatch for {path.name}")
    expected = entry.get("md5")
    if expected:
        digest = hashlib.md5()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"MD5 mismatch for {path.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", default=["all"])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--list", action="store_true", help="Print URLs without downloading")
    args = parser.parse_args()
    entries = json.loads((ROOT / "data" / "sources.json").read_text())["sources"]
    names = [entry["id"] for entry in entries]
    if "all" not in args.dataset and set(args.dataset) - set(names):
        parser.error(f"Unknown dataset; choose from {names} or all")
    selected = [entry for entry in entries if "all" in args.dataset or entry["id"] in args.dataset]
    for entry in selected:
        if args.list:
            print(f'{entry["id"]}: {entry["filename"]}\n  {entry["url"]}')
            continue
        args.output_dir.mkdir(parents=True, exist_ok=True)
        target = args.output_dir / entry["filename"]
        if target.exists():
            verify(target, entry)
            print(f"Verified existing {target.name}")
            continue
        partial = target.with_suffix(target.suffix + ".part")
        if partial.exists() and partial.stat().st_size == entry["size_bytes"]:
            verify(partial, entry)
            partial.replace(target)
            print(f"Verified completed partial {target.name}")
            continue
        subprocess.run(
            ["curl", "--location", "--fail", "--retry", "5", "--continue-at", "-",
             "--output", str(partial), entry["url"]],
            check=True,
        )
        verify(partial, entry)
        partial.replace(target)
        print(f"Downloaded and verified {target.name}")


if __name__ == "__main__":
    main()
