"""Verify the numerical inputs, frozen definitions, and runtime in this release."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def verify(root=ROOT):
    manifest = json.loads((root / "provenance/release_inputs.json").read_text())
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Non-portable manifest path: {relative}")
        path = root / relative
        data = path.read_bytes()
        if len(data) != entry["size_bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"Checksum mismatch: {relative}")
    return len(manifest["files"])


if __name__ == "__main__":
    print(f"Verified {verify()} release inputs and runtime modules")
