"""Hash the H5AD datasets that Plan C actually consumes.

The digest is independent of unrelated H5AD groups such as cached encoder
representations in ``obsm``. It is intended for provenance checks, not as a
scientific analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import h5py
import numpy as np


DATASETS = (
    "X",
    "obs/cell_line/categories",
    "obs/cell_line/codes",
    "obs/pert/categories",
    "obs/pert/codes",
    "var/gene_id",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk_rows", type=int, default=2048)
    return parser.parse_args()


def update_length_prefixed(digest: hashlib._Hash, value: bytes) -> None:
    digest.update(struct.pack("<Q", len(value)))
    digest.update(value)


def normalized_header(path: str, dataset: h5py.Dataset) -> bytes:
    header = {
        "path": path,
        "shape": [int(value) for value in dataset.shape],
        "dtype": dataset.dtype.str,
    }
    return json.dumps(header, sort_keys=True, separators=(",", ":")).encode()


def hash_dataset(path: str, dataset: h5py.Dataset, chunk_rows: int) -> str:
    digest = hashlib.sha256()
    digest.update(b"cccfm-replogle-core-dataset-v1\0")
    update_length_prefixed(digest, normalized_header(path, dataset))

    if dataset.dtype.kind in {"O", "S", "U"}:
        values = np.asarray(dataset[:]).reshape(-1)
        for value in values:
            if isinstance(value, (bytes, np.bytes_)):
                encoded = bytes(value)
            else:
                encoded = str(value).encode("utf-8")
            update_length_prefixed(digest, encoded)
        return digest.hexdigest()

    if dataset.ndim == 0:
        digest.update(np.ascontiguousarray(dataset[()]).tobytes(order="C"))
        return digest.hexdigest()

    for start in range(0, dataset.shape[0], chunk_rows):
        stop = min(start + chunk_rows, dataset.shape[0])
        block = np.ascontiguousarray(dataset[start:stop])
        digest.update(block.tobytes(order="C"))
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.chunk_rows < 1:
        raise ValueError("--chunk_rows must be positive")
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    if args.output.exists():
        raise FileExistsError(args.output)

    with h5py.File(args.data, "r") as handle:
        missing = [path for path in DATASETS if path not in handle]
        if missing:
            raise KeyError(f"Missing required H5AD datasets: {missing}")
        dataset_hashes = {
            path: hash_dataset(path, handle[path], args.chunk_rows)
            for path in DATASETS
        }
        x_shape = [int(value) for value in handle["X"].shape]
        x_dtype = handle["X"].dtype.str
        obsm_keys = sorted(handle["obsm"].keys()) if "obsm" in handle else []

    combined = hashlib.sha256()
    combined.update(b"cccfm-replogle-core-v1\0")
    for path in DATASETS:
        update_length_prefixed(combined, path.encode())
        update_length_prefixed(combined, dataset_hashes[path].encode())

    record = {
        "schema": "cccfm-replogle-core-v1",
        "data": str(args.data),
        "file_size_bytes": args.data.stat().st_size,
        "x_shape": x_shape,
        "x_dtype": x_dtype,
        "obsm_keys_ignored_by_core_digest": obsm_keys,
        "dataset_sha256": dataset_hashes,
        "core_sha256": combined.hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
