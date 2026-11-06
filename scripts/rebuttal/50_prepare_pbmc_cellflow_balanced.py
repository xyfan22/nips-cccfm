"""Prepare balanced PBMC snapshots for the B2 CellFlow donor-transfer audit.

The selection rule is identical to the frozen B1 controls audit. The large
input matrix is opened in backed mode and only selected rows are materialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp


DEFAULT_CELL_TYPES = (
    "B Intermediate/Memory",
    "B Naive",
    "CD4 Memory",
    "CD8 Memory",
    "CD14 Mono",
    "NK",
)
FROZEN_SEEDS = (7, 42, 314)
FROZEN_EXCLUDED = ("IL-15", "IL-17E")
FROZEN_INPUT_SIZE = 13_101_592_422
FROZEN_INPUT_MD5 = "d89559875a6c75ab37e0adf84c6e177d"
EXPECTED_DONORS = 12
EXPECTED_NON_PBS_CYTOKINES = 88
N_CELLS_PER_GROUP = 50
MIN_CELLS = 50
TARGET_SUM = 1e4
TRAINING_SEED = 42
FROZEN_PCA_COMPONENTS = 100
FROZEN_PCA_SEED = 42
HASH_CHUNK_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class PrepareConfig:
    input_path: Path
    output_dir: Path
    seeds: tuple[int, ...] = FROZEN_SEEDS
    cell_types: tuple[str, ...] = DEFAULT_CELL_TYPES
    n_cells: int = N_CELLS_PER_GROUP
    min_cells: int = MIN_CELLS
    expected_donors: int = EXPECTED_DONORS
    expected_cytokines: int = EXPECTED_NON_PBS_CYTOKINES
    expected_excluded: tuple[str, ...] = FROZEN_EXCLUDED
    expected_input_size: int = FROZEN_INPUT_SIZE
    expected_input_md5: str = FROZEN_INPUT_MD5
    pca_n_components: int = FROZEN_PCA_COMPONENTS
    pca_seed: int = FROZEN_PCA_SEED
    allow_nonfrozen_fixture: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(FROZEN_SEEDS))
    parser.add_argument("--cell_types", nargs="+", default=list(DEFAULT_CELL_TYPES))
    parser.add_argument("--n_cells", type=int, default=N_CELLS_PER_GROUP)
    parser.add_argument("--min_cells", type=int, default=MIN_CELLS)
    parser.add_argument("--expected_donors", type=int, default=EXPECTED_DONORS)
    parser.add_argument(
        "--expected_cytokines", type=int, default=EXPECTED_NON_PBS_CYTOKINES
    )
    parser.add_argument("--expected_excluded", nargs="*", default=list(FROZEN_EXCLUDED))
    parser.add_argument("--expected_input_size", type=int, default=FROZEN_INPUT_SIZE)
    parser.add_argument("--expected_input_md5", default=FROZEN_INPUT_MD5)
    parser.add_argument("--pca_n_components", type=int, default=FROZEN_PCA_COMPONENTS)
    parser.add_argument("--pca_seed", type=int, default=FROZEN_PCA_SEED)
    parser.add_argument(
        "--allow_nonfrozen_fixture",
        action="store_true",
        help="Synthetic-test only; outputs are rejected by the real fold runner.",
    )
    return parser.parse_args()


def decode(values: np.ndarray) -> list[str]:
    decoded: list[str] = []
    for value in values.tolist():
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        decoded.append(str(value))
    return decoded


def categorical(handle: h5py.File, key: str) -> tuple[list[str], np.ndarray]:
    group = handle["obs"][key]
    if "categories" not in group or "codes" not in group:
        raise ValueError(f"obs/{key} is not categorical")
    return decode(group["categories"][:]), np.asarray(group["codes"][:], dtype=np.int32)


def hash_file(path: Path, algorithms: Iterable[str] = ("sha256",)) -> dict[str, str]:
    hashers = {name: hashlib.new(name) for name in algorithms}
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            for hasher in hashers.values():
                hasher.update(chunk)
    return {name: hasher.hexdigest() for name, hasher in hashers.items()}


def row_digest(row_indices: np.ndarray) -> str:
    rows = np.sort(np.asarray(row_indices, dtype="<i8"))
    return hashlib.sha256(rows.tobytes()).hexdigest()


def normalize_log1p(matrix: Any) -> Any:
    if sp.issparse(matrix):
        result = sp.csr_matrix(matrix, dtype=np.float32, copy=True)
        totals = np.asarray(result.sum(axis=1)).ravel()
        if np.any(totals <= 0):
            raise ValueError("Selected zero-count cells cannot be normalized")
        result = result.multiply((TARGET_SUM / totals)[:, None]).tocsr()
        np.log1p(result.data, out=result.data)
        return result

    result = np.asarray(matrix, dtype=np.float32).copy()
    totals = result.sum(axis=1)
    if np.any(totals <= 0):
        raise ValueError("Selected zero-count cells cannot be normalized")
    result *= (TARGET_SUM / totals)[:, None]
    np.log1p(result, out=result)
    return result


def mean_rows(matrix: Any, mask: np.ndarray) -> np.ndarray:
    if int(mask.sum()) == 0:
        raise ValueError("Cannot average an empty row selection")
    value = matrix[mask].mean(axis=0)
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _read_var_and_rows(input_path: Path, rows: np.ndarray) -> tuple[Any, pd.DataFrame]:
    backed = ad.read_h5ad(input_path, backed="r")
    try:
        matrix = backed.X[rows]
        if sp.issparse(matrix):
            matrix = sp.csr_matrix(matrix, dtype=np.float32, copy=True)
        else:
            matrix = np.asarray(matrix, dtype=np.float32).copy()
        var = backed.var.copy()
    finally:
        backed.file.close()
    return normalize_log1p(matrix), var


def _make_obs(
    rows: np.ndarray,
    seed: int,
    donors: list[str],
    donor_codes: np.ndarray,
    all_cell_types: list[str],
    selected_cell_types: tuple[str, ...],
    cell_codes: np.ndarray,
    all_cytokines: list[str],
    selected_conditions: list[str],
    cytokine_codes: np.ndarray,
    treatments: list[str],
    treatment_codes: np.ndarray,
) -> pd.DataFrame:
    donor_values = np.asarray(donors, dtype=object)[donor_codes[rows]]
    cell_values = np.asarray(all_cell_types, dtype=object)[cell_codes[rows]]
    cytokine_values = np.asarray(all_cytokines, dtype=object)[cytokine_codes[rows]]
    treatment_values = np.asarray(treatments, dtype=object)[treatment_codes[rows]]
    selected_treatments = [
        value for value in treatments if value in set(treatment_values.tolist())
    ]
    obs = pd.DataFrame(
        {
            "donor": pd.Categorical(donor_values, categories=donors),
            "cell_type": pd.Categorical(
                cell_values, categories=list(selected_cell_types)
            ),
            "cytokine": pd.Categorical(cytokine_values, categories=selected_conditions),
            "treatment": pd.Categorical(
                treatment_values, categories=selected_treatments
            ),
            "is_control": cytokine_values == "PBS",
            "source_row": rows.astype(np.int64),
            "sample_seed": np.full(len(rows), seed, dtype=np.int32),
        },
        index=pd.Index([f"source_{row}" for row in rows], name="cell_id"),
    )
    return obs


def _validate_frozen_cli(config: PrepareConfig) -> bool:
    frozen = (
        config.seeds == FROZEN_SEEDS
        and config.cell_types == DEFAULT_CELL_TYPES
        and config.n_cells == N_CELLS_PER_GROUP
        and config.min_cells == MIN_CELLS
        and config.expected_donors == EXPECTED_DONORS
        and config.expected_cytokines == EXPECTED_NON_PBS_CYTOKINES
        and set(config.expected_excluded) == set(FROZEN_EXCLUDED)
        and config.expected_input_size == FROZEN_INPUT_SIZE
        and config.expected_input_md5 == FROZEN_INPUT_MD5
        and config.pca_n_components == FROZEN_PCA_COMPONENTS
        and config.pca_seed == FROZEN_PCA_SEED
    )
    if not frozen and not config.allow_nonfrozen_fixture:
        raise ValueError(
            "Non-frozen protocol arguments require --allow_nonfrozen_fixture"
        )
    return frozen


def prepare_balanced(config: PrepareConfig) -> dict[str, Any]:
    frozen_arguments = _validate_frozen_cli(config)
    if not config.input_path.is_file():
        raise FileNotFoundError(config.input_path)
    if config.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {config.output_dir}")
    if config.n_cells <= 1 or config.min_cells < config.n_cells:
        raise ValueError("Require 1 < n_cells <= min_cells")
    if len(set(config.seeds)) != len(config.seeds) or TRAINING_SEED not in config.seeds:
        raise ValueError("Seeds must be unique and include 42")
    if len(set(config.cell_types)) != len(config.cell_types):
        raise ValueError("Duplicate cell types")
    if config.pca_n_components < 1 or config.pca_seed < 0:
        raise ValueError("PCA components must be positive and the seed non-negative")

    input_size = config.input_path.stat().st_size
    input_hashes = hash_file(config.input_path, ("md5", "sha256"))
    input_matches = (
        input_size == config.expected_input_size
        and input_hashes["md5"] == config.expected_input_md5
    )
    if not input_matches and not config.allow_nonfrozen_fixture:
        raise ValueError(
            "Input fingerprint mismatch: "
            f"size={input_size}, md5={input_hashes['md5']}"
        )

    with h5py.File(config.input_path, "r") as handle:
        donors, donor_codes = categorical(handle, "donor")
        cell_types_all, cell_codes = categorical(handle, "cell_type")
        cytokines_all, cytokine_codes = categorical(handle, "cytokine")
        treatments, treatment_codes = categorical(handle, "treatment")
        x_node = handle["X"]
        shape_value = x_node.attrs["shape"] if "shape" in x_node.attrs else x_node.shape
        shape = tuple(int(value) for value in shape_value)
        n_obs, n_vars = shape

        if len(donors) != config.expected_donors:
            raise ValueError(
                f"Observed {len(donors)} donors, expected {config.expected_donors}"
            )
        code_arrays = (donor_codes, cell_codes, cytokine_codes, treatment_codes)
        if any(len(codes) != n_obs for codes in code_arrays):
            raise ValueError("obs code length does not match X")
        if "PBS" not in cytokines_all or "PBS" not in treatments:
            raise ValueError("PBS control is missing")
        missing_cell_types = sorted(set(config.cell_types) - set(cell_types_all))
        if missing_cell_types:
            raise ValueError(f"Missing cell types: {missing_cell_types}")

        selected_cell_codes = [cell_types_all.index(name) for name in config.cell_types]
        cell_to_local = np.full(len(cell_types_all), -1, dtype=np.int16)
        cell_to_local[selected_cell_codes] = np.arange(
            len(selected_cell_codes), dtype=np.int16
        )
        valid = (
            (donor_codes >= 0)
            & (cell_codes >= 0)
            & (cytokine_codes >= 0)
            & (treatment_codes >= 0)
        )
        selected_cell_local = cell_to_local[
            np.clip(cell_codes, 0, len(cell_to_local) - 1)
        ]
        selected_mask = valid & (selected_cell_local >= 0)

        full_flat = (
            donor_codes[selected_mask].astype(np.int64) * len(config.cell_types)
            + selected_cell_local[selected_mask].astype(np.int64)
        ) * len(cytokines_all) + cytokine_codes[selected_mask].astype(np.int64)
        counts = np.bincount(
            full_flat,
            minlength=len(donors) * len(config.cell_types) * len(cytokines_all),
        ).reshape(len(donors), len(config.cell_types), len(cytokines_all))

        pbs_full_idx = cytokines_all.index("PBS")
        if int(counts[:, :, pbs_full_idx].min()) < config.min_cells:
            raise ValueError("At least one selected PBS group is below min_cells")
        eligible_full_idxs = [
            idx
            for idx, name in enumerate(cytokines_all)
            if name != "PBS" and int(counts[:, :, idx].min()) >= config.min_cells
        ]
        excluded = [
            cytokines_all[idx]
            for idx in range(len(cytokines_all))
            if cytokines_all[idx] != "PBS" and idx not in eligible_full_idxs
        ]
        if len(eligible_full_idxs) != config.expected_cytokines:
            raise ValueError(
                f"Observed {len(eligible_full_idxs)} eligible cytokines, "
                f"expected {config.expected_cytokines}; excluded={excluded}"
            )
        if set(excluded) != set(config.expected_excluded):
            raise ValueError(
                f"Excluded cytokines {sorted(excluded)} != expected "
                f"{sorted(config.expected_excluded)}"
            )

        cytokines = [cytokines_all[idx] for idx in eligible_full_idxs]
        selected_conditions = ["PBS", *cytokines]
        selected_full_cytokine_idxs = [pbs_full_idx, *eligible_full_idxs]
        full_cyto_to_local = np.full(len(cytokines_all), -1, dtype=np.int16)
        full_cyto_to_local[selected_full_cytokine_idxs] = np.arange(
            len(selected_full_cytokine_idxs), dtype=np.int16
        )
        selected_cyto_local = full_cyto_to_local[
            np.clip(cytokine_codes, 0, len(full_cyto_to_local) - 1)
        ]
        final_mask = selected_mask & (selected_cyto_local >= 0)

        n_cell_types = len(config.cell_types)
        n_conditions = len(selected_full_cytokine_idxs)
        n_groups = len(donors) * n_cell_types * n_conditions
        group_ids = (
            donor_codes[final_mask].astype(np.int64) * n_cell_types
            + selected_cell_local[final_mask].astype(np.int64)
        ) * n_conditions + selected_cyto_local[final_mask].astype(np.int64)
        global_rows = np.flatnonzero(final_mask)
        order = np.argsort(group_ids, kind="stable")
        sorted_group_ids = group_ids[order]
        sorted_global_rows = global_rows[order]
        group_counts = np.bincount(sorted_group_ids, minlength=n_groups)
        if np.any(group_counts < config.n_cells):
            raise ValueError(
                f"Groups below n_cells: {np.flatnonzero(group_counts < config.n_cells)[:20].tolist()}"
            )
        offsets = np.concatenate(([0], np.cumsum(group_counts)))

        selected_rows: dict[int, np.ndarray] = {}
        for seed in config.seeds:
            sampled: list[np.ndarray] = []
            for group_id in range(n_groups):
                candidates = sorted_global_rows[
                    offsets[group_id] : offsets[group_id + 1]
                ]
                rng = np.random.default_rng(np.random.SeedSequence([seed, group_id]))
                sampled.append(rng.choice(candidates, config.n_cells, replace=False))
            rows = np.sort(np.concatenate(sampled).astype(np.int64))
            expected_rows = n_groups * config.n_cells
            if len(rows) != expected_rows or len(np.unique(rows)) != expected_rows:
                raise AssertionError("Balanced selection row count/uniqueness failed")
            selected_rows[seed] = rows

        esm_group = handle["uns"]["esm2_embeddings"]
        missing_embeddings = sorted(set(cytokines) - set(esm_group.keys()))
        if missing_embeddings:
            raise ValueError(f"Missing ESM2 embeddings: {missing_embeddings}")
        if "PBS" in esm_group:
            raise ValueError("PBS must use the null token, not an ESM2 embedding")
        esm2_embeddings = {
            name: np.asarray(esm_group[name][:], dtype=np.float32).reshape(-1)
            for name in cytokines
        }
        bad_dims = {
            name: value.shape
            for name, value in esm2_embeddings.items()
            if value.shape != (2560,)
        }
        if bad_dims:
            raise ValueError(f"Unexpected ESM2 dimensions: {bad_dims}")

    def build_snapshot(seed: int) -> ad.AnnData:
        rows = selected_rows[seed]
        matrix, var = _read_var_and_rows(config.input_path, rows)
        if matrix.shape != (len(rows), n_vars):
            raise AssertionError("Selected matrix shape mismatch")
        obs = _make_obs(
            rows,
            seed,
            donors,
            donor_codes,
            cell_types_all,
            config.cell_types,
            cell_codes,
            cytokines_all,
            selected_conditions,
            cytokine_codes,
            treatments,
            treatment_codes,
        )
        snapshot = ad.AnnData(X=matrix, obs=obs, var=var)
        snapshot.uns["esm2_embeddings"] = {
            key: value.copy() for key, value in esm2_embeddings.items()
        }
        return snapshot

    seed42_snapshot = build_snapshot(TRAINING_SEED)
    donor_embeddings: dict[str, np.ndarray] = {}
    donor_pbs_rows: dict[str, list[int]] = {}
    donor_pbs_digests: dict[str, str] = {}
    for donor in donors:
        mask = (
            seed42_snapshot.obs["donor"].astype(str).to_numpy() == donor
        ) & seed42_snapshot.obs["is_control"].to_numpy(dtype=bool)
        expected_pbs = config.n_cells * len(config.cell_types)
        if int(mask.sum()) != expected_pbs:
            raise ValueError(f"{donor}: expected {expected_pbs} seed-42 PBS cells")
        rows = seed42_snapshot.obs.loc[mask, "source_row"].to_numpy(dtype=np.int64)
        donor_pbs_rows[donor] = rows.tolist()
        donor_pbs_digests[donor] = row_digest(rows)
        donor_embeddings[donor] = mean_rows(seed42_snapshot.X, mask)
        if donor_embeddings[donor].shape != (n_vars,):
            raise AssertionError("Donor embedding dimension mismatch")

    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{config.output_dir.name}.staging-",
            dir=config.output_dir.parent,
        )
    )
    try:
        seed_manifest: dict[str, Any] = {}
        for seed in config.seeds:
            snapshot = (
                seed42_snapshot if seed == TRAINING_SEED else build_snapshot(seed)
            )
            snapshot.uns["donor_embeddings_b2"] = {
                key: value.copy() for key, value in donor_embeddings.items()
            }
            expected_rows = (
                len(donors)
                * len(config.cell_types)
                * (len(cytokines) + 1)
                * config.n_cells
            )
            if snapshot.n_obs != expected_rows:
                raise AssertionError(
                    f"seed {seed}: {snapshot.n_obs} rows != {expected_rows}"
                )
            group_sizes = snapshot.obs.groupby(
                ["donor", "cell_type", "cytokine"], observed=True
            ).size()
            if len(group_sizes) != n_groups or not np.all(
                group_sizes.to_numpy() == config.n_cells
            ):
                raise AssertionError(f"seed {seed}: groups are not exactly balanced")
            output_path = staging / f"seed{seed}.h5ad"
            snapshot.write_h5ad(output_path, compression="gzip")
            output_sha = hash_file(output_path)["sha256"]
            seed_manifest[str(seed)] = {
                "path": output_path.name,
                "n_rows": int(snapshot.n_obs),
                "source_row_location": "obs/source_row",
                "row_digest_sha256": row_digest(selected_rows[seed]),
                "sha256": output_sha,
                "size_bytes": output_path.stat().st_size,
            }

        manifest: dict[str, Any] = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "plan": "PLAN_B2_full_pbmc_cellflow.md",
            "frozen_protocol": bool(frozen_arguments and input_matches),
            "input_path": str(config.input_path),
            "input_size_bytes": input_size,
            "input_md5": input_hashes["md5"],
            "input_sha256": input_hashes["sha256"],
            "input_shape": [n_obs, n_vars],
            "donors": donors,
            "cell_types": list(config.cell_types),
            "cytokines": cytokines,
            "excluded_cytokines": sorted(excluded),
            "n_cells_per_group": config.n_cells,
            "normalization": "per-cell normalize_total(1e4) then log1p",
            "pca_n_components": config.pca_n_components,
            "pca_solver": "randomized",
            "pca_seed": config.pca_seed,
            "pca_fit_donors": "source-only per fold",
            "training_sample_seed": TRAINING_SEED,
            "sample_seeds": list(config.seeds),
            "seeds": seed_manifest,
            "esm2_embedding_dim": 2560,
            "donor_embedding_source": "seed-42 PBS mean over six fixed cell types",
            "donor_embedding_dim": n_vars,
            "donor_embeddings_b2": {
                key: value.tolist() for key, value in donor_embeddings.items()
            },
            "donor_pbs_source_rows": donor_pbs_rows,
            "donor_pbs_row_digest_sha256": donor_pbs_digests,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(config.output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    args = parse_args()
    config = PrepareConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        seeds=tuple(args.seeds),
        cell_types=tuple(args.cell_types),
        n_cells=args.n_cells,
        min_cells=args.min_cells,
        expected_donors=args.expected_donors,
        expected_cytokines=args.expected_cytokines,
        expected_excluded=tuple(args.expected_excluded),
        expected_input_size=args.expected_input_size,
        expected_input_md5=args.expected_input_md5,
        pca_n_components=args.pca_n_components,
        pca_seed=args.pca_seed,
        allow_nonfrozen_fixture=args.allow_nonfrozen_fixture,
    )
    manifest = prepare_balanced(config)
    print(f"Balanced snapshots written to {config.output_dir}")
    print(f"Frozen protocol: {manifest['frozen_protocol']}")
    for seed in config.seeds:
        info = manifest["seeds"][str(seed)]
        print(f"  seed{seed}.h5ad: {info['n_rows']} rows, {info['size_bytes']} bytes")


if __name__ == "__main__":
    main()
