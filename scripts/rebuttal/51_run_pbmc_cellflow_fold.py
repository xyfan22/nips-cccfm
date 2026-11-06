"""Run one strict leave-one-donor-out B2 CellFlow fold.

The module keeps JAX and CellFlow imports lazy so per-fold cache paths are set
before the runtime initializes. A completed fold contains all three methods:
target-control-copy, shifted-control, and CellFlow.
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


FROZEN_SAMPLE_SEEDS = (7, 42, 314)
FROZEN_PCA_COMPONENTS = 100
FROZEN_PCA_SEED = 42
FROZEN_MODEL_SEED = 42
FROZEN_ITERATIONS = 100_000
FROZEN_BATCH_CANDIDATES = (1024, 512, 256, 128)
FROZEN_METHODS = ("target_control_copy", "shifted_control", "cellflow")
FROZEN_DONORS = tuple(f"Donor{index}" for index in range(1, 13))
FROZEN_CELL_TYPES = (
    "B Intermediate/Memory",
    "B Naive",
    "CD4 Memory",
    "CD8 Memory",
    "CD14 Mono",
    "NK",
)
METRICS = (
    "pearson_ctrl",
    "pearson_pert",
    "mse",
    "centroid_accuracy",
    "mmd",
    "energy",
    "centered_energy",
    "moment",
)
ROW_FIELDS = (
    "method",
    "sample_seed",
    "target_donor",
    "cell_type",
    "cytokine",
    "n_cells",
    "n_source_donors",
    "condition_id",
    *METRICS,
)
HASH_CHUNK_BYTES = 8 * 1024 * 1024
CELLFLOW_SPLIT_CELL_TYPE_KEY = "z_b2_cell_type"
CELLFLOW_COVARIATE_ADAPTER: dict[str, Any] = {
    "name": "cellflow-0.5.3-split-key-order",
    "donor_key": "donor",
    "original_cell_type_key": "cell_type",
    "aliased_cell_type_key": CELLFLOW_SPLIT_CELL_TYPE_KEY,
    "split_covariates": ["donor", CELLFLOW_SPLIT_CELL_TYPE_KEY],
    "changes_values": False,
}

FROZEN_ARCHITECTURE: dict[str, Any] = {
    "solver": "otfm",
    "condition_mode": "deterministic",
    "regularization": 0.0,
    "pooling": "attention_token",
    "pooling_kwargs": {},
    "layers_before_pool": {
        "cytokine_treatment": {
            "layer_type": "mlp",
            "dims": [1024, 1024],
            "dropout_rate": 0.5,
        },
        "donor": {
            "layer_type": "mlp",
            "dims": [256, 256],
            "dropout_rate": 0.0,
        },
    },
    "layers_after_pool": {
        "layer_type": "mlp",
        "dims": [1024, 1024],
        "dropout_rate": 0.0,
    },
    "condition_embedding_dim": 256,
    "cond_output_dropout": 0.9,
    "condition_encoder_kwargs": {},
    "pool_sample_covariates": True,
    "time_freqs": 1024,
    "time_encoder_dims": [1024, 1024, 1024],
    "time_encoder_dropout": 0.0,
    "hidden_dims": [2048, 2048, 2048],
    "hidden_dropout": 0.0,
    "conditioning": "concatenation",
    "conditioning_kwargs": {},
    "decoder_dims": [4096, 4096, 4096],
    "decoder_dropout": 0.0,
    "activation": "silu",
    "vf_kwargs": None,
    "probability_path": {"constant_noise": 0.5},
    "match_epsilon": 0.5,
    "match_tau_a": 1.0,
    "match_tau_b": 1.0,
    "optimizer": "MultiSteps(Adam(5e-5), every_k=20)",
    "solver_kwargs": {},
    "layer_norm_before_concatenation": False,
    "linear_projection_before_concatenation": False,
    "iterations": FROZEN_ITERATIONS,
    "model_seed": FROZEN_MODEL_SEED,
}


@dataclass
class FrozenPCA:
    mean_: np.ndarray
    components_: np.ndarray
    explained_variance_: np.ndarray
    explained_variance_ratio_: np.ndarray

    def transform(self, matrix: Any) -> np.ndarray:
        dense = to_dense(matrix)
        return ((dense - self.mean_) @ self.components_.T).astype(np.float32)

    def inverse_transform(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=np.float32)
        return (values @ self.components_ + self.mean_).astype(np.float32)


@dataclass
class FoldContext:
    manifest: dict[str, Any]
    source_donors: list[str]
    train_data: ad.AnnData
    pca: FrozenPCA
    pca_audit: dict[str, Any]
    donor_embeddings: dict[str, np.ndarray]


@dataclass
class SeedCloudIndex:
    data: ad.AnnData
    donors: list[str]
    cell_types: list[str]
    conditions: list[str]
    n_cells: int
    grouped_matrix: Any
    group_means: np.ndarray

    def group_id(self, donor: str, cell_type: str, condition: str) -> int:
        donor_index = self.donors.index(donor)
        cell_index = self.cell_types.index(cell_type)
        condition_index = self.conditions.index(condition)
        return (donor_index * len(self.cell_types) + cell_index) * len(
            self.conditions
        ) + condition_index

    def cloud(self, donor: str, cell_type: str, condition: str) -> np.ndarray:
        group_id = self.group_id(donor, cell_type, condition)
        start = group_id * self.n_cells
        return to_dense(self.grouped_matrix[start : start + self.n_cells])

    def centroid(self, donor: str, cell_type: str, condition: str) -> np.ndarray:
        return self.group_means[self.group_id(donor, cell_type, condition)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--balanced_dir", type=Path, required=True)
    parser.add_argument("--target_donor", required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path)
    parser.add_argument("--batch_size", type=int, choices=FROZEN_BATCH_CANDIDATES)
    parser.add_argument("--resource_probe_result", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--resource_probe", action="store_true")
    modes.add_argument("--train_only", action="store_true")
    modes.add_argument("--eval_only", action="store_true")
    modes.add_argument(
        "--resource_probe_candidate", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--allow_nonfrozen_fixture", action="store_true", help=argparse.SUPPRESS
    )
    return parser.parse_args()


def hash_file(path: Path, algorithms: Iterable[str] = ("sha256",)) -> dict[str, str]:
    hashers = {name: hashlib.new(name) for name in algorithms}
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            for hasher in hashers.values():
                hasher.update(chunk)
    return {name: hasher.hexdigest() for name, hasher in hashers.items()}


def row_digest(rows: np.ndarray) -> str:
    values = np.sort(np.asarray(rows, dtype="<i8"))
    return hashlib.sha256(values.tobytes()).hexdigest()


def to_dense(matrix: Any) -> np.ndarray:
    if sp.issparse(matrix):
        return matrix.toarray().astype(np.float32, copy=False)
    return np.asarray(matrix, dtype=np.float32)


def mean_rows(matrix: Any, mask: np.ndarray) -> np.ndarray:
    if int(mask.sum()) == 0:
        raise ValueError("Cannot average an empty selection")
    return np.asarray(matrix[mask].mean(axis=0), dtype=np.float32).reshape(-1)


def attach_cellflow_split_alias(frame: pd.DataFrame) -> None:
    """Add a value-identical cell-type key that avoids a CellFlow 0.5.3 order bug."""
    if "cell_type" not in frame:
        raise ValueError("CellFlow input is missing cell_type")
    original = frame["cell_type"].astype(str).to_numpy()
    if CELLFLOW_SPLIT_CELL_TYPE_KEY in frame:
        observed = frame[CELLFLOW_SPLIT_CELL_TYPE_KEY].astype(str).to_numpy()
        if not np.array_equal(original, observed):
            raise ValueError("CellFlow cell-type compatibility alias changes values")
        return
    frame[CELLFLOW_SPLIT_CELL_TYPE_KEY] = frame["cell_type"].copy()
    observed = frame[CELLFLOW_SPLIT_CELL_TYPE_KEY].astype(str).to_numpy()
    if not np.array_equal(original, observed):
        raise AssertionError(
            "Failed to construct CellFlow cell-type compatibility alias"
        )


def verify_cellflow_covariate_adapter() -> None:
    """Fail closed unless CellFlow's sorted split-key lookup has stable ordering."""
    condition_key_order = ["cytokine", "donor", CELLFLOW_SPLIT_CELL_TYPE_KEY]
    lookup_key_order = [
        "cytokine",
        *sorted(["donor", CELLFLOW_SPLIT_CELL_TYPE_KEY]),
    ]
    if condition_key_order != lookup_key_order:
        raise AssertionError(
            "CellFlow compatibility alias does not stabilize condition-key ordering"
        )


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    if left.size < 2 or right.size < 2:
        return 0.0
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else 0.0


def centroid_accuracy(pred: np.ndarray, centroids: np.ndarray, true_idx: int) -> float:
    if len(centroids) <= 1:
        raise ValueError("Centroid accuracy requires at least two perturbations")
    distances = np.linalg.norm(centroids - pred[None, :], axis=1)
    return float(np.sum(distances > distances[true_idx]) / (len(distances) - 1))


def batch_sqdist(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    left_sq = np.sum(left * left, axis=-1, keepdims=True)
    right_sq = np.sum(right * right, axis=-1, keepdims=True)
    distances = (
        left_sq
        + np.swapaxes(right_sq, 1, 2)
        - 2.0 * np.matmul(left, np.swapaxes(right, 1, 2))
    )
    return np.maximum(distances, 0.0)


def batch_distribution_metrics(
    pred: np.ndarray, target: np.ndarray
) -> dict[str, np.ndarray]:
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError(
            f"Expected equal [conditions,cells,genes] arrays, got {pred.shape}, {target.shape}"
        )
    dpp = batch_sqdist(pred, pred)
    dtt = batch_sqdist(target, target)
    dpt = batch_sqdist(pred, target)
    energy = (
        2.0 * np.sqrt(dpt + 1e-12).mean(axis=(1, 2))
        - np.sqrt(dpp + 1e-12).mean(axis=(1, 2))
        - np.sqrt(dtt + 1e-12).mean(axis=(1, 2))
    )
    mmd = np.zeros(len(pred), dtype=np.float32)
    for bandwidth in (0.1, 1.0, 10.0):
        scale = 2.0 * bandwidth * bandwidth
        mmd += (
            np.exp(-dpp / scale).mean(axis=(1, 2))
            + np.exp(-dtt / scale).mean(axis=(1, 2))
            - 2.0 * np.exp(-dpt / scale).mean(axis=(1, 2))
        )
    mmd /= 3.0

    pred_centered = pred - pred.mean(axis=1, keepdims=True)
    target_centered = target - target.mean(axis=1, keepdims=True)
    dpp_c = batch_sqdist(pred_centered, pred_centered)
    dtt_c = batch_sqdist(target_centered, target_centered)
    dpt_c = batch_sqdist(pred_centered, target_centered)
    centered_energy = (
        2.0 * np.sqrt(dpt_c + 1e-12).mean(axis=(1, 2))
        - np.sqrt(dpp_c + 1e-12).mean(axis=(1, 2))
        - np.sqrt(dtt_c + 1e-12).mean(axis=(1, 2))
    )

    def moments(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean_value = values.mean(axis=1)
        centered = values - mean_value[:, None, :]
        variance = (centered * centered).mean(axis=1)
        std = np.sqrt(variance)
        denominator = np.maximum(std**3, 1e-12)
        skew = np.where(
            std > 1e-12,
            (centered * centered * centered).mean(axis=1) / denominator,
            0.0,
        )
        return mean_value, variance, skew

    pred_mean, pred_var, pred_skew = moments(pred)
    target_mean, target_var, target_skew = moments(target)
    moment = (
        np.abs(pred_mean - target_mean).mean(axis=1)
        + np.abs(pred_var - target_var).mean(axis=1)
        + np.abs(pred_skew - target_skew).mean(axis=1)
    )
    return {
        "mmd": mmd,
        "energy": energy,
        "centered_energy": centered_energy,
        "moment": moment,
    }


def evaluate_cloud_batch(
    pred_clouds: np.ndarray,
    true_clouds: np.ndarray,
    target_control_mean: np.ndarray,
) -> list[dict[str, float]]:
    if pred_clouds.shape != true_clouds.shape:
        raise ValueError("Predicted and target cloud batches differ in shape")
    pred_centroids = pred_clouds.mean(axis=1)
    true_centroids = true_clouds.mean(axis=1)
    perturbation_reference = true_centroids.mean(axis=0)
    distribution = batch_distribution_metrics(pred_clouds, true_clouds)
    rows: list[dict[str, float]] = []
    for index, (pred_profile, true_profile) in enumerate(
        zip(pred_centroids, true_centroids, strict=True)
    ):
        row = {
            "pearson_ctrl": safe_pearson(
                pred_profile - target_control_mean,
                true_profile - target_control_mean,
            ),
            "pearson_pert": safe_pearson(
                pred_profile - perturbation_reference,
                true_profile - perturbation_reference,
            ),
            "mse": float(np.mean((pred_profile - true_profile) ** 2)),
            "centroid_accuracy": centroid_accuracy(pred_profile, true_centroids, index),
            "mmd": float(distribution["mmd"][index]),
            "energy": float(distribution["energy"][index]),
            "centered_energy": float(distribution["centered_energy"][index]),
            "moment": float(distribution["moment"][index]),
        }
        if not all(math.isfinite(value) for value in row.values()):
            raise ValueError(f"Non-finite metrics for condition {index}: {row}")
        rows.append(row)
    return rows


def configure_runtime(cache_dir: Path) -> None:
    cache_dir = cache_dir.resolve()
    paths = {
        "TMPDIR": cache_dir / "tmp",
        "JAX_COMPILATION_CACHE_DIR": cache_dir / "jax",
        "NUMBA_CACHE_DIR": cache_dir / "numba",
        "MPLCONFIGDIR": cache_dir / "matplotlib",
        "XDG_CACHE_HOME": cache_dir / "xdg",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    for key, path in paths.items():
        os.environ[key] = str(path)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_PLATFORMS", "cuda")
    tempfile.tempdir = str(paths["TMPDIR"])


def import_cellflow_runtime() -> dict[str, Any]:
    import jax
    import optax
    from flax import linen as nn
    from cellflow.model import CellFlow
    from cellflow.utils import match_linear

    try:
        jax.config.update(
            "jax_compilation_cache_dir", os.environ["JAX_COMPILATION_CACHE_DIR"]
        )
    except Exception as error:
        raise RuntimeError("Failed to configure the JAX compilation cache") from error
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError(f"Expected exactly one JAX GPU, observed {devices}")
    return {
        "jax": jax,
        "optax": optax,
        "nn": nn,
        "CellFlow": CellFlow,
        "match_linear": match_linear,
        "devices": devices,
    }


def load_manifest(balanced_dir: Path, allow_nonfrozen_fixture: bool) -> dict[str, Any]:
    manifest_path = balanced_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("frozen_protocol", False) and not allow_nonfrozen_fixture:
        raise ValueError("Balanced input is not marked as the frozen B2 protocol")
    seeds = tuple(int(seed) for seed in manifest["sample_seeds"])
    if not allow_nonfrozen_fixture:
        frozen_checks = {
            "sample seeds": seeds == FROZEN_SAMPLE_SEEDS,
            "donors": tuple(manifest.get("donors", ())) == FROZEN_DONORS,
            "cell types": tuple(manifest.get("cell_types", ())) == FROZEN_CELL_TYPES,
            "cytokine count": len(manifest.get("cytokines", ())) == 88,
            "excluded cytokines": set(manifest.get("excluded_cytokines", ()))
            == {"IL-15", "IL-17E"},
            "cells per group": int(manifest.get("n_cells_per_group", -1)) == 50,
            "PCA components": int(manifest.get("pca_n_components", -1))
            == FROZEN_PCA_COMPONENTS,
            "PCA seed": int(manifest.get("pca_seed", -1)) == FROZEN_PCA_SEED,
            "training sample seed": int(manifest.get("training_sample_seed", -1)) == 42,
            "ESM2 dimension": int(manifest.get("esm2_embedding_dim", -1)) == 2560,
        }
        failed = [name for name, passed in frozen_checks.items() if not passed]
        if failed:
            raise ValueError(f"Balanced manifest violates frozen fields: {failed}")
    if len(set(seeds)) != len(seeds) or 42 not in seeds:
        raise ValueError(f"Invalid sample seeds: {seeds}")
    for seed in seeds:
        info = manifest["seeds"][str(seed)]
        path = balanced_dir / info["path"]
        if not path.is_file() or path.stat().st_size != int(info["size_bytes"]):
            raise ValueError(f"Balanced seed file missing/size mismatch: {path}")
        if hash_file(path)["sha256"] != info["sha256"]:
            raise ValueError(f"Balanced seed hash mismatch: {path}")
    return manifest


def fit_fold_context(
    balanced_dir: Path,
    target_donor: str,
    allow_nonfrozen_fixture: bool = False,
) -> FoldContext:
    manifest = load_manifest(balanced_dir, allow_nonfrozen_fixture)
    donors = list(manifest["donors"])
    if target_donor not in donors:
        raise ValueError(f"Unknown target donor {target_donor}")
    source_donors = [donor for donor in donors if donor != target_donor]
    seed42_path = balanced_dir / manifest["seeds"]["42"]["path"]
    full = ad.read_h5ad(seed42_path)
    donor_values = full.obs["donor"].astype(str).to_numpy()
    source_mask = np.isin(donor_values, source_donors)
    target_mask = donor_values == target_donor
    if np.any(source_mask & target_mask) or int(target_mask.sum()) == 0:
        raise AssertionError("Invalid source/target fold masks")
    if int(source_mask.sum()) == 0:
        raise AssertionError("Fold has no source rows")

    from sklearn import __version__ as sklearn_version
    from sklearn.decomposition import PCA

    source_matrix = to_dense(full.X[source_mask])
    n_components = int(manifest["pca_n_components"])
    if n_components > min(source_matrix.shape):
        raise ValueError(
            f"PCA components {n_components} exceed source matrix {source_matrix.shape}"
        )
    estimator = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=int(manifest["pca_seed"]),
    )
    source_pca = estimator.fit_transform(source_matrix).astype(np.float32)
    pca = FrozenPCA(
        mean_=np.asarray(estimator.mean_, dtype=np.float32),
        components_=np.asarray(estimator.components_, dtype=np.float32),
        explained_variance_=np.asarray(estimator.explained_variance_, dtype=np.float32),
        explained_variance_ratio_=np.asarray(
            estimator.explained_variance_ratio_, dtype=np.float32
        ),
    )
    del source_matrix

    source_rows = full.obs.loc[source_mask, "source_row"].to_numpy(dtype=np.int64)
    target_rows = full.obs.loc[target_mask, "source_row"].to_numpy(dtype=np.int64)
    train_data = full[source_mask].copy()
    train_data.obsm["X_pca"] = source_pca
    verify_cellflow_covariate_adapter()
    attach_cellflow_split_alias(train_data.obs)

    donor_embeddings: dict[str, np.ndarray] = {}
    for donor in donors:
        donor_pbs_mask = (donor_values == donor) & full.obs["is_control"].to_numpy(
            dtype=bool
        )
        expected = int(manifest["n_cells_per_group"]) * len(manifest["cell_types"])
        if int(donor_pbs_mask.sum()) != expected:
            raise ValueError(f"{donor}: expected {expected} seed-42 PBS rows")
        observed_rows = full.obs.loc[donor_pbs_mask, "source_row"].to_numpy(
            dtype=np.int64
        )
        expected_digest = manifest["donor_pbs_row_digest_sha256"][donor]
        if row_digest(observed_rows) != expected_digest:
            raise ValueError(f"{donor}: PBS source-row digest mismatch")
        embedding = mean_rows(full.X, donor_pbs_mask)
        manifest_embedding = np.asarray(
            manifest["donor_embeddings_b2"][donor], dtype=np.float32
        )
        if not np.allclose(embedding, manifest_embedding, rtol=0.0, atol=1e-6):
            raise ValueError(f"{donor}: donor embedding differs from manifest")
        donor_embeddings[donor] = embedding

    train_data.uns["donor_embeddings_b2"] = {
        key: donor_embeddings[key].copy() for key in source_donors
    }
    if set(train_data.uns["esm2_embeddings"]) != set(manifest["cytokines"]):
        raise ValueError("Training ESM2 representation keys differ from manifest")
    if target_donor in set(train_data.obs["donor"].astype(str)):
        raise AssertionError("Target donor leaked into CellFlow training rows")
    if target_donor in train_data.uns["donor_embeddings_b2"]:
        raise AssertionError(
            "Target donor embedding leaked into CellFlow training data"
        )

    pca_audit = {
        "target_donor": target_donor,
        "source_donors": source_donors,
        "fit_row_count": int(source_mask.sum()),
        "fit_source_row_digest_sha256": row_digest(source_rows),
        "target_rows_available": int(target_mask.sum()),
        "target_rows_in_fit": 0,
        "target_source_row_digest_sha256": row_digest(target_rows),
        "n_components": int(pca.components_.shape[0]),
        "solver": "randomized",
        "random_state": int(manifest["pca_seed"]),
        "scikit_learn_version": sklearn_version,
        "target_donor_embedding_source": "seed-42 PBS controls only",
        "target_donor_embedding_row_count": len(
            manifest["donor_pbs_source_rows"][target_donor]
        ),
        "target_donor_embedding_row_digest_sha256": manifest[
            "donor_pbs_row_digest_sha256"
        ][target_donor],
    }
    return FoldContext(
        manifest=manifest,
        source_donors=source_donors,
        train_data=train_data,
        pca=pca,
        pca_audit=pca_audit,
        donor_embeddings=donor_embeddings,
    )


def prepare_cellflow_model(train_data: ad.AnnData, runtime: dict[str, Any]) -> Any:
    CellFlow = runtime["CellFlow"]
    optax = runtime["optax"]
    nn = runtime["nn"]
    match_linear = runtime["match_linear"]
    verify_cellflow_covariate_adapter()
    attach_cellflow_split_alias(train_data.obs)
    model = CellFlow(train_data, solver="otfm")
    model.prepare_data(
        sample_rep="X_pca",
        control_key="is_control",
        perturbation_covariates={"cytokine_treatment": ("cytokine",)},
        perturbation_covariate_reps={"cytokine_treatment": "esm2_embeddings"},
        sample_covariates=["donor"],
        sample_covariate_reps={"donor": "donor_embeddings_b2"},
        split_covariates=["donor", CELLFLOW_SPLIT_CELL_TYPE_KEY],
        max_combination_length=1,
        null_value=0.0,
    )
    match_fn = functools.partial(
        match_linear,
        epsilon=FROZEN_ARCHITECTURE["match_epsilon"],
        tau_a=FROZEN_ARCHITECTURE["match_tau_a"],
        tau_b=FROZEN_ARCHITECTURE["match_tau_b"],
    )
    model.prepare_model(
        condition_mode="deterministic",
        regularization=0.0,
        pooling="attention_token",
        pooling_kwargs={},
        layers_before_pool=FROZEN_ARCHITECTURE["layers_before_pool"],
        layers_after_pool=FROZEN_ARCHITECTURE["layers_after_pool"],
        condition_embedding_dim=256,
        cond_output_dropout=0.9,
        condition_encoder_kwargs={},
        pool_sample_covariates=True,
        time_freqs=1024,
        time_encoder_dims=[1024, 1024, 1024],
        time_encoder_dropout=0.0,
        hidden_dims=[2048, 2048, 2048],
        hidden_dropout=0.0,
        conditioning="concatenation",
        conditioning_kwargs={},
        decoder_dims=[4096, 4096, 4096],
        decoder_dropout=0.0,
        vf_act_fn=nn.silu,
        vf_kwargs=None,
        probability_path={"constant_noise": 0.5},
        match_fn=match_fn,
        optimizer=optax.MultiSteps(optax.adam(5e-5), 20),
        solver_kwargs={},
        layer_norm_before_concatenation=False,
        linear_projection_before_concatenation=False,
        seed=FROZEN_MODEL_SEED,
    )
    return model


def save_pca(path: Path, pca: FrozenPCA) -> None:
    np.savez_compressed(
        path,
        mean=pca.mean_,
        components=pca.components_,
        explained_variance=pca.explained_variance_,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )


def load_pca(path: Path) -> FrozenPCA:
    with np.load(path) as values:
        return FrozenPCA(
            mean_=values["mean"].astype(np.float32),
            components_=values["components"].astype(np.float32),
            explained_variance_=values["explained_variance"].astype(np.float32),
            explained_variance_ratio_=values["explained_variance_ratio"].astype(
                np.float32
            ),
        )


def build_seed_cloud_index(
    data: ad.AnnData, manifest: dict[str, Any]
) -> SeedCloudIndex:
    donors = list(manifest["donors"])
    cell_types = list(manifest["cell_types"])
    conditions = ["PBS", *manifest["cytokines"]]
    n_cells = int(manifest["n_cells_per_group"])
    expected_rows = len(donors) * len(cell_types) * len(conditions) * n_cells
    if data.n_obs != expected_rows:
        raise ValueError(
            f"Balanced snapshot has {data.n_obs} rows, expected {expected_rows}"
        )

    observed_categories = {
        "donor": list(data.obs["donor"].cat.categories),
        "cell_type": list(data.obs["cell_type"].cat.categories),
        "cytokine": list(data.obs["cytokine"].cat.categories),
    }
    expected_categories = {
        "donor": donors,
        "cell_type": cell_types,
        "cytokine": conditions,
    }
    if observed_categories != expected_categories:
        raise ValueError("Balanced snapshot categories differ from the manifest")
    donor_codes = data.obs["donor"].cat.codes.to_numpy(dtype=np.int64)
    cell_codes = data.obs["cell_type"].cat.codes.to_numpy(dtype=np.int64)
    condition_codes = data.obs["cytokine"].cat.codes.to_numpy(dtype=np.int64)
    if np.any(donor_codes < 0) or np.any(cell_codes < 0) or np.any(condition_codes < 0):
        raise ValueError("Balanced snapshot contains missing group categories")
    group_ids = (donor_codes * len(cell_types) + cell_codes) * len(
        conditions
    ) + condition_codes
    n_groups = len(donors) * len(cell_types) * len(conditions)
    counts = np.bincount(group_ids, minlength=n_groups)
    if len(counts) != n_groups or not np.all(counts == n_cells):
        raise ValueError("Balanced snapshot groups are not uniformly complete")
    order = np.argsort(group_ids, kind="stable")
    grouped_ids = group_ids[order]
    if not np.array_equal(grouped_ids, np.repeat(np.arange(n_groups), n_cells)):
        raise AssertionError("Balanced group ordering failed")
    grouped_matrix = data.X[order]
    assignment = sp.csr_matrix(
        (
            np.full(data.n_obs, 1.0 / n_cells, dtype=np.float32),
            (group_ids, np.arange(data.n_obs)),
        ),
        shape=(n_groups, data.n_obs),
    )
    group_means = to_dense(assignment @ data.X)
    if (
        group_means.shape != (n_groups, data.n_vars)
        or not np.isfinite(group_means).all()
    ):
        raise ValueError("Balanced snapshot group means are invalid")
    return SeedCloudIndex(
        data=data,
        donors=donors,
        cell_types=cell_types,
        conditions=conditions,
        n_cells=n_cells,
        grouped_matrix=grouped_matrix,
        group_means=group_means,
    )


def build_control_and_target_clouds(
    cloud_index: SeedCloudIndex,
    target_donor: str,
    source_donors: list[str],
    cell_type: str,
    cytokines: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_control = cloud_index.cloud(target_donor, cell_type, "PBS")
    true_clouds = np.stack(
        [cloud_index.cloud(target_donor, cell_type, cyto) for cyto in cytokines]
    ).astype(np.float32)
    donor_deltas: list[np.ndarray] = []
    for donor in source_donors:
        source_control_mean = cloud_index.centroid(donor, cell_type, "PBS")
        source_centroids = np.stack(
            [cloud_index.centroid(donor, cell_type, cyto) for cyto in cytokines]
        )
        donor_deltas.append(source_centroids - source_control_mean[None, :])
    source_delta = np.stack(donor_deltas).mean(axis=0).astype(np.float32)
    copy_clouds = np.broadcast_to(target_control[None, :, :], true_clouds.shape).copy()
    shifted_clouds = copy_clouds + source_delta[:, None, :]
    return target_control, true_clouds, copy_clouds, shifted_clouds


def predict_cellflow_clouds(
    model: Any,
    data: ad.AnnData,
    pca: FrozenPCA,
    target_donor: str,
    cell_type: str,
    cytokines: list[str],
    sample_seed: int,
    n_cells: int,
    donor_embeddings: dict[str, np.ndarray],
) -> np.ndarray:
    target_control_mask = (
        (data.obs["donor"].astype(str).to_numpy() == target_donor)
        & (data.obs["cell_type"].astype(str).to_numpy() == cell_type)
        & data.obs["is_control"].to_numpy(dtype=bool)
    )
    if int(target_control_mask.sum()) != n_cells:
        raise ValueError(
            f"{target_donor}/{cell_type}: target PBS count is not {n_cells}"
        )
    controls = data[target_control_mask].copy()
    verify_cellflow_covariate_adapter()
    attach_cellflow_split_alias(controls.obs)
    controls.obsm["X_pca"] = pca.transform(controls.X)
    controls.uns["donor_embeddings_b2"] = {
        target_donor: donor_embeddings[target_donor].copy()
    }
    condition_ids = [
        f"seed{sample_seed}|{target_donor}|{cell_type}|{cytokine}"
        for cytokine in cytokines
    ]
    covariates = pd.DataFrame(
        {
            "donor": [target_donor] * len(cytokines),
            "cell_type": [cell_type] * len(cytokines),
            "cytokine": cytokines,
            "treatment": ["cytokine"] * len(cytokines),
            "is_control": [False] * len(cytokines),
            "condition_id": condition_ids,
        }
    )
    attach_cellflow_split_alias(covariates)
    predictions = model.predict(
        adata=controls,
        covariate_data=covariates,
        sample_rep="X_pca",
        condition_id_key="condition_id",
        throw=True,
    )
    if not isinstance(predictions, dict) or set(predictions) != set(condition_ids):
        observed = (
            set(predictions) if isinstance(predictions, dict) else type(predictions)
        )
        raise ValueError(f"CellFlow prediction keys differ: {observed}")
    clouds: list[np.ndarray] = []
    for condition_id in condition_ids:
        values = np.asarray(predictions[condition_id], dtype=np.float32)
        while values.ndim > 2 and values.shape[0] == 1:
            values = values[0]
        if values.shape != (n_cells, pca.components_.shape[0]):
            raise ValueError(f"{condition_id}: prediction shape {values.shape}")
        gene_values = pca.inverse_transform(values)
        if (
            gene_values.shape != (n_cells, pca.mean_.shape[0])
            or not np.isfinite(gene_values).all()
        ):
            raise ValueError(f"{condition_id}: invalid reconstructed prediction")
        clouds.append(gene_values)
    return np.stack(clouds).astype(np.float32)


def evaluate_fold(
    model: Any,
    balanced_dir: Path,
    context: FoldContext,
    target_donor: str,
) -> list[dict[str, Any]]:
    manifest = context.manifest
    cytokines = list(manifest["cytokines"])
    cell_types = list(manifest["cell_types"])
    n_cells = int(manifest["n_cells_per_group"])
    rows: list[dict[str, Any]] = []
    for sample_seed in (int(seed) for seed in manifest["sample_seeds"]):
        seed_path = balanced_dir / manifest["seeds"][str(sample_seed)]["path"]
        data = ad.read_h5ad(seed_path)
        data.uns["donor_embeddings_b2"] = {
            key: value.copy() for key, value in context.donor_embeddings.items()
        }
        cloud_index = build_seed_cloud_index(data, manifest)
        for cell_type in cell_types:
            (
                target_control,
                true_clouds,
                copy_clouds,
                shifted_clouds,
            ) = build_control_and_target_clouds(
                cloud_index,
                target_donor,
                context.source_donors,
                cell_type,
                cytokines,
            )
            cellflow_clouds = predict_cellflow_clouds(
                model,
                data,
                context.pca,
                target_donor,
                cell_type,
                cytokines,
                sample_seed,
                n_cells,
                context.donor_embeddings,
            )
            method_clouds = {
                "target_control_copy": copy_clouds,
                "shifted_control": shifted_clouds,
                "cellflow": cellflow_clouds,
            }
            target_control_mean = target_control.mean(axis=0)
            for method in FROZEN_METHODS:
                metric_rows = evaluate_cloud_batch(
                    method_clouds[method], true_clouds, target_control_mean
                )
                for cytokine, metrics in zip(cytokines, metric_rows, strict=True):
                    rows.append(
                        {
                            "method": method,
                            "sample_seed": sample_seed,
                            "target_donor": target_donor,
                            "cell_type": cell_type,
                            "cytokine": cytokine,
                            "n_cells": n_cells,
                            "n_source_donors": len(context.source_donors),
                            "condition_id": (
                                f"seed{sample_seed}|{target_donor}|{cell_type}|{cytokine}"
                            ),
                            **metrics,
                        }
                    )
    validate_fold_rows(rows, manifest, target_donor)
    return rows


def validate_fold_rows(
    rows: list[dict[str, Any]], manifest: dict[str, Any], target_donor: str
) -> None:
    expected_per_method = (
        len(manifest["sample_seeds"])
        * len(manifest["cell_types"])
        * len(manifest["cytokines"])
    )
    if len(rows) != expected_per_method * len(FROZEN_METHODS):
        raise ValueError(
            f"Fold has {len(rows)} rows, expected {expected_per_method * 3}"
        )
    expected_keys = {
        (int(seed), target_donor, cell_type, cytokine)
        for seed in manifest["sample_seeds"]
        for cell_type in manifest["cell_types"]
        for cytokine in manifest["cytokines"]
    }
    expected_source_donors = len(manifest["donors"]) - 1
    reference_keys: set[tuple[Any, ...]] | None = None
    for method in FROZEN_METHODS:
        selected = [row for row in rows if row["method"] == method]
        keys = {
            (
                int(row["sample_seed"]),
                row["target_donor"],
                row["cell_type"],
                row["cytokine"],
            )
            for row in selected
        }
        if (
            len(selected) != expected_per_method
            or len(keys) != expected_per_method
            or keys != expected_keys
        ):
            raise ValueError(f"{method}: incomplete or duplicate fold rows")
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise ValueError("Method evaluation key sets differ")
        for row in selected:
            if row["target_donor"] != target_donor:
                raise ValueError("Fold row has the wrong target donor")
            if int(row["n_cells"]) != int(manifest["n_cells_per_group"]):
                raise ValueError("Fold row violates uniform n")
            if int(row["n_source_donors"]) != expected_source_donors:
                raise ValueError("Fold row has the wrong source-donor count")
            expected_condition_id = (
                f"seed{int(row['sample_seed'])}|{target_donor}|"
                f"{row['cell_type']}|{row['cytokine']}"
            )
            if row["condition_id"] != expected_condition_id:
                raise ValueError("Fold row has an invalid condition_id")
            if not all(math.isfinite(float(row[metric])) for metric in METRICS):
                raise ValueError("Fold row has non-finite metrics")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def load_resource_probe_provenance(
    path: Path | None,
    batch_size: int,
    allow_nonfrozen_fixture: bool,
) -> dict[str, Any]:
    if path is None:
        if allow_nonfrozen_fixture:
            return {
                "path": "synthetic fixture",
                "sha256": "0" * 64,
                "target_donor": "Donor1",
                "selected_batch_size": batch_size,
            }
        raise ValueError("Training requires --resource_probe_result")
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if (
        payload.get("target_donor") != "Donor1"
        or int(payload.get("selected_batch_size", -1)) != batch_size
        or not isinstance(payload.get("attempts"), list)
        or not payload["attempts"]
    ):
        raise ValueError("Resource-probe result does not authorize this batch size")
    successful = [
        int(attempt["batch_size"])
        for attempt in payload["attempts"]
        if int(attempt.get("returncode", -1)) == 0
    ]
    if successful != [batch_size]:
        raise ValueError("Resource-probe success record is incomplete or ambiguous")
    attempted_sizes = [int(attempt["batch_size"]) for attempt in payload["attempts"]]
    expected_prefix = list(FROZEN_BATCH_CANDIDATES[: len(attempted_sizes)])
    if attempted_sizes != expected_prefix or attempted_sizes[-1] != batch_size:
        raise ValueError("Resource-probe candidates were not tried in frozen order")
    if any(
        int(attempt.get("returncode", -1)) != 86 for attempt in payload["attempts"][:-1]
    ):
        raise ValueError("Resource-probe contains a non-resource failed attempt")
    return {
        "path": str(path.resolve()),
        "sha256": hash_file(path)["sha256"],
        "target_donor": payload["target_donor"],
        "selected_batch_size": batch_size,
    }


def train_model(
    context: FoldContext,
    runtime: dict[str, Any],
    batch_size: int,
) -> tuple[Any, dict[str, Any], list[float]]:
    model = prepare_cellflow_model(context.train_data, runtime)
    started = time.time()
    model.train(
        num_iterations=FROZEN_ITERATIONS,
        batch_size=batch_size,
        valid_freq=FROZEN_ITERATIONS + 1,
    )
    wall_seconds = time.time() - started
    losses = [float(value) for value in model.trainer.training_logs.get("loss", [])]
    if len(losses) != FROZEN_ITERATIONS or not np.isfinite(losses).all():
        raise ValueError(
            f"Expected {FROZEN_ITERATIONS} finite losses, observed {len(losses)}"
        )
    window = max(1, len(losses) // 10)
    first_mean = float(np.mean(losses[:window]))
    last_mean = float(np.mean(losses[-window:]))
    summary = {
        "iterations": FROZEN_ITERATIONS,
        "batch_size": batch_size,
        "model_seed": FROZEN_MODEL_SEED,
        "wall_seconds": wall_seconds,
        "first_10pct_mean_loss": first_mean,
        "last_10pct_mean_loss": last_mean,
        "loss_decreased": bool(last_mean < first_mean),
        "all_losses_finite": True,
    }
    return model, summary, losses


def save_training_artifacts(
    stage: Path,
    model: Any,
    context: FoldContext,
    target_donor: str,
    batch_size: int,
    training_summary: dict[str, Any],
    losses: list[float],
    runtime: dict[str, Any],
    resource_probe: dict[str, Any],
) -> dict[str, Any]:
    stage.mkdir(parents=True, exist_ok=False)
    model.save(stage, overwrite=False)
    checkpoint = stage / "CellFlow.pkl"
    save_pca(stage / "pca.npz", context.pca)
    write_json(stage / "pca_audit.json", context.pca_audit)
    config = {
        "target_donor": target_donor,
        "source_donors": context.source_donors,
        "sample_seeds": context.manifest["sample_seeds"],
        "cell_types": context.manifest["cell_types"],
        "cytokines": context.manifest["cytokines"],
        "n_cells_per_group": context.manifest["n_cells_per_group"],
        "gene_dim": int(context.pca.mean_.shape[0]),
        "pca": {
            "n_components": int(context.manifest["pca_n_components"]),
            "solver": "randomized",
            "random_state": int(context.manifest["pca_seed"]),
        },
        "architecture": FROZEN_ARCHITECTURE,
        "cellflow_covariate_adapter": CELLFLOW_COVARIATE_ADAPTER,
        "batch_size": batch_size,
        "resource_probe": resource_probe,
        "jax_devices": [str(device) for device in runtime["devices"]],
    }
    write_json(stage / "config.json", config)
    write_json(stage / "training_summary.json", training_summary)
    with (stage / "training_loss.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", "loss"])
        writer.writerows(enumerate(losses, start=1))
    artifact_hashes = {
        "checkpoint_sha256": hash_file(checkpoint)["sha256"],
        "pca_sha256": hash_file(stage / "pca.npz")["sha256"],
        "pca_audit_sha256": hash_file(stage / "pca_audit.json")["sha256"],
        "config_sha256": hash_file(stage / "config.json")["sha256"],
        "training_summary_sha256": hash_file(stage / "training_summary.json")["sha256"],
        "training_loss_sha256": hash_file(stage / "training_loss.csv")["sha256"],
    }
    training_audit = {
        "target_donor": target_donor,
        "target_rows_in_pca_fit": context.pca_audit["target_rows_in_fit"],
        "target_rows_in_training": 0,
        "target_donor_embedding_registered_in_training": False,
        "train_rows": int(context.train_data.n_obs),
        "checkpoint": checkpoint.name,
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "pca_size_bytes": (stage / "pca.npz").stat().st_size,
        **artifact_hashes,
        "target_donor_embedding_source": context.pca_audit[
            "target_donor_embedding_source"
        ],
        "target_donor_embedding_row_count": context.pca_audit[
            "target_donor_embedding_row_count"
        ],
        "target_donor_embedding_row_digest_sha256": context.pca_audit[
            "target_donor_embedding_row_digest_sha256"
        ],
        "loss_decreased": training_summary["loss_decreased"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(stage / "training_audit.json", training_audit)
    return training_audit


def save_evaluation_artifacts(
    evaluation_dir: Path,
    rows: list[dict[str, Any]],
    target_donor: str,
    manifest: dict[str, Any],
) -> None:
    evaluation_dir.mkdir(parents=True, exist_ok=False)
    write_csv(evaluation_dir / "per_condition_metrics.csv", rows)
    summary_rows: list[dict[str, Any]] = []
    for method in FROZEN_METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        summary_rows.append(
            {
                "method": method,
                "n_rows": len(method_rows),
                **{
                    metric: float(np.mean([row[metric] for row in method_rows]))
                    for metric in METRICS
                },
            }
        )
    with (evaluation_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "n_rows", *METRICS])
        writer.writeheader()
        writer.writerows(summary_rows)
    audit = {
        "target_donor": target_donor,
        "methods": list(FROZEN_METHODS),
        "n_rows": len(rows),
        "expected_rows": (
            len(FROZEN_METHODS)
            * len(manifest["sample_seeds"])
            * len(manifest["cell_types"])
            * len(manifest["cytokines"])
        ),
        "uniform_n": int(manifest["n_cells_per_group"]),
        "method_key_sets_equal": True,
        "all_metrics_finite": True,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(evaluation_dir / "evaluation_audit.json", audit)


def finalize_evaluation(
    fold_dir: Path,
    rows: list[dict[str, Any]],
    target_donor: str,
    manifest: dict[str, Any],
    evaluation_wall_seconds: float,
) -> dict[str, str]:
    evaluation_final = fold_dir / "evaluation"
    if evaluation_final.exists():
        raise FileExistsError(evaluation_final)
    staging_parent = Path(tempfile.mkdtemp(prefix=".evaluation.staging-", dir=fold_dir))
    stage = staging_parent / "evaluation"
    try:
        save_evaluation_artifacts(stage, rows, target_donor, manifest)
        evaluation_hashes = {
            "metrics_sha256": hash_file(stage / "per_condition_metrics.csv")["sha256"],
            "evaluation_audit_sha256": hash_file(stage / "evaluation_audit.json")[
                "sha256"
            ],
            "evaluation_summary_sha256": hash_file(stage / "summary.csv")["sha256"],
        }
        stage.rename(evaluation_final)
        staging_parent.rmdir()
    except BaseException:
        shutil.rmtree(staging_parent, ignore_errors=True)
        raise

    fold_audit_path = fold_dir / "fold_audit.json"
    fold_audit = json.loads(fold_audit_path.read_text())
    fold_audit.update(
        {
            "evaluation_complete": True,
            "evaluation_wall_seconds": evaluation_wall_seconds,
            "evaluation_completed_at_utc": datetime.now(timezone.utc).isoformat(),
            **evaluation_hashes,
        }
    )
    temporary_audit = fold_dir / ".fold_audit.json.tmp"
    write_json(temporary_audit, fold_audit)
    temporary_audit.replace(fold_audit_path)
    return evaluation_hashes


def verify_saved_training_artifacts(
    fold_dir: Path,
    context: FoldContext,
    target_donor: str,
) -> dict[str, Any]:
    config = json.loads((fold_dir / "config.json").read_text())
    training_audit = json.loads((fold_dir / "training_audit.json").read_text())
    pca_audit = json.loads((fold_dir / "pca_audit.json").read_text())
    expected_config = {
        "target_donor": target_donor,
        "source_donors": context.source_donors,
        "sample_seeds": context.manifest["sample_seeds"],
        "cell_types": context.manifest["cell_types"],
        "cytokines": context.manifest["cytokines"],
        "n_cells_per_group": context.manifest["n_cells_per_group"],
        "gene_dim": int(context.train_data.n_vars),
        "pca": {
            "n_components": int(context.manifest["pca_n_components"]),
            "solver": "randomized",
            "random_state": int(context.manifest["pca_seed"]),
        },
        "architecture": FROZEN_ARCHITECTURE,
        "cellflow_covariate_adapter": CELLFLOW_COVARIATE_ADAPTER,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise ValueError(f"Saved training config differs for {key}")
    if int(config.get("batch_size", -1)) not in FROZEN_BATCH_CANDIDATES:
        raise ValueError("Saved training config has an invalid batch size")
    resource_probe = config.get("resource_probe", {})
    if (
        resource_probe.get("target_donor") != "Donor1"
        or int(resource_probe.get("selected_batch_size", -1))
        != int(config["batch_size"])
        or len(str(resource_probe.get("sha256", ""))) != 64
    ):
        raise ValueError("Saved resource-probe provenance is invalid")
    probe_path = Path(str(resource_probe.get("path", "")))
    if context.manifest.get("frozen_protocol", False):
        if not probe_path.is_file():
            raise FileNotFoundError("Saved resource-probe result is unavailable")
        if hash_file(probe_path)["sha256"] != resource_probe["sha256"]:
            raise ValueError("Saved resource-probe result hash differs")
    if int(training_audit.get("target_rows_in_training", -1)) != 0:
        raise ValueError("Saved training audit reports target-row leakage")
    if training_audit.get("target_donor_embedding_registered_in_training") is not False:
        raise ValueError("Saved training audit reports target-donor embedding leakage")
    if int(pca_audit.get("target_rows_in_fit", -1)) != 0:
        raise ValueError("Saved PCA audit reports target-row leakage")
    expected_target_digest = context.manifest["donor_pbs_row_digest_sha256"][
        target_donor
    ]
    if (
        training_audit.get("target_donor_embedding_row_digest_sha256")
        != expected_target_digest
        or pca_audit.get("target_donor_embedding_row_digest_sha256")
        != expected_target_digest
    ):
        raise ValueError("Saved target-donor embedding audit differs from the manifest")

    artifact_names = {
        "checkpoint_sha256": "CellFlow.pkl",
        "pca_sha256": "pca.npz",
        "pca_audit_sha256": "pca_audit.json",
        "config_sha256": "config.json",
        "training_summary_sha256": "training_summary.json",
        "training_loss_sha256": "training_loss.csv",
    }
    for audit_key, name in artifact_names.items():
        observed = hash_file(fold_dir / name)["sha256"]
        if observed != training_audit.get(audit_key):
            raise ValueError(f"Saved artifact hash mismatch: {name}")
    return config


def _resource_error(error: BaseException) -> bool:
    message = f"{type(error).__name__}: {error}".lower()
    patterns = (
        "out of memory",
        "resource exhausted",
        "resource_exhausted",
        "cuda_error_out_of_memory",
        "failed to allocate",
    )
    return any(pattern in message for pattern in patterns)


def run_probe_candidate(args: argparse.Namespace) -> None:
    if args.batch_size is None:
        raise ValueError("Probe candidate requires --batch_size")
    cache_dir = args.cache_dir or args.out_dir.parent / f".{args.out_dir.name}.cache"
    configure_runtime(cache_dir)
    runtime = import_cellflow_runtime()
    context = fit_fold_context(
        args.balanced_dir, args.target_donor, args.allow_nonfrozen_fixture
    )
    first_cell_type = context.manifest["cell_types"][0]
    probe_cytokines = context.manifest["cytokines"][:4]
    obs = context.train_data.obs
    probe_mask = (obs["cell_type"].astype(str).to_numpy() == first_cell_type) & obs[
        "cytokine"
    ].astype(str).isin(["PBS", *probe_cytokines]).to_numpy()
    probe_train = context.train_data[probe_mask].copy()
    model = prepare_cellflow_model(probe_train, runtime)
    model.train(num_iterations=3, batch_size=args.batch_size, valid_freq=4)
    seed42 = ad.read_h5ad(args.balanced_dir / context.manifest["seeds"]["42"]["path"])
    predictions = predict_cellflow_clouds(
        model,
        seed42,
        context.pca,
        args.target_donor,
        first_cell_type,
        [probe_cytokines[0]],
        42,
        int(context.manifest["n_cells_per_group"]),
        context.donor_embeddings,
    )
    if predictions.shape[0] != 1 or not np.isfinite(predictions).all():
        raise ValueError("Probe prediction is invalid")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        args.out_dir / "candidate_success.json",
        {"batch_size": args.batch_size, "prediction_shape": list(predictions.shape)},
    )


def _remove_probe_runtime_dirs(stage: Path) -> None:
    for path in stage.glob("cache_batch*"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    for path in stage.glob("batch*"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)


def run_resource_probe_parent(args: argparse.Namespace) -> None:
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    if args.target_donor != "Donor1" and not args.allow_nonfrozen_fixture:
        raise ValueError("The frozen resource probe must use Donor1")
    args.out_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{args.out_dir.name}.staging-", dir=args.out_dir.parent
        )
    )
    try:
        attempts: list[dict[str, Any]] = []
        selected: int | None = None
        for batch_size in FROZEN_BATCH_CANDIDATES:
            candidate_dir = stage / f"batch{batch_size}"
            cache_dir = stage / f"cache_batch{batch_size}"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--balanced_dir",
                str(args.balanced_dir),
                "--target_donor",
                args.target_donor,
                "--out_dir",
                str(candidate_dir),
                "--cache_dir",
                str(cache_dir),
                "--batch_size",
                str(batch_size),
                "--resource_probe_candidate",
            ]
            if args.allow_nonfrozen_fixture:
                command.append("--allow_nonfrozen_fixture")
            started = time.time()
            process = subprocess.run(
                command, text=True, capture_output=True, check=False
            )
            log_text = process.stdout + "\n--- STDERR ---\n" + process.stderr
            (stage / f"batch{batch_size}.log").write_text(log_text)
            attempt = {
                "batch_size": batch_size,
                "returncode": process.returncode,
                "wall_seconds": time.time() - started,
            }
            attempts.append(attempt)
            if process.returncode == 0:
                selected = batch_size
                break
            if process.returncode != 86:
                raise RuntimeError(
                    f"Resource probe batch {batch_size} failed for a non-resource reason; "
                    f"see {stage / f'batch{batch_size}.log'}"
                )
        if selected is None:
            raise RuntimeError("All frozen batch sizes failed the resource probe")
        write_json(
            stage / "resource_probe.json",
            {
                "target_donor": args.target_donor,
                "selected_batch_size": selected,
                "attempts": attempts,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        _remove_probe_runtime_dirs(stage)
        stage.rename(args.out_dir)
    except BaseException as error:
        if stage.exists():
            write_json(
                stage / "resource_probe_failure.json",
                {
                    "target_donor": args.target_donor,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "attempts": locals().get("attempts", []),
                    "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            _remove_probe_runtime_dirs(stage)
            stage.rename(args.out_dir)
        raise


def _default_cache_dir(out_dir: Path, suffix: str) -> Path:
    return out_dir.parent / f".{out_dir.name}.{suffix}.cache"


def run_train_or_full(args: argparse.Namespace) -> None:
    if args.batch_size is None:
        raise ValueError(
            "Training requires the batch size selected by --resource_probe"
        )
    resource_probe = load_resource_probe_provenance(
        args.resource_probe_result,
        args.batch_size,
        args.allow_nonfrozen_fixture,
    )
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    args.out_dir.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or _default_cache_dir(args.out_dir, "train")
    configure_runtime(cache_dir)
    runtime = import_cellflow_runtime()
    context = fit_fold_context(
        args.balanced_dir, args.target_donor, args.allow_nonfrozen_fixture
    )
    model, training_summary, losses = train_model(context, runtime, args.batch_size)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{args.out_dir.name}.staging-", dir=args.out_dir.parent
        )
    )
    shutil.rmtree(stage)
    try:
        training_audit = save_training_artifacts(
            stage,
            model,
            context,
            args.target_donor,
            args.batch_size,
            training_summary,
            losses,
            runtime,
            resource_probe,
        )
        fold_audit = {
            **training_audit,
            "evaluation_complete": False,
            "balanced_manifest_sha256": hash_file(args.balanced_dir / "manifest.json")[
                "sha256"
            ],
            "cache_dir": str(cache_dir.resolve()),
        }
        write_json(stage / "fold_audit.json", fold_audit)
        stage.rename(args.out_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    if not args.train_only:
        evaluation_started = time.time()
        rows = evaluate_fold(model, args.balanced_dir, context, args.target_donor)
        finalize_evaluation(
            args.out_dir,
            rows,
            args.target_donor,
            context.manifest,
            time.time() - evaluation_started,
        )


def run_eval_only(args: argparse.Namespace) -> None:
    if not args.out_dir.is_dir():
        raise FileNotFoundError(args.out_dir)
    evaluation_final = args.out_dir / "evaluation"
    if evaluation_final.exists():
        raise FileExistsError(evaluation_final)
    required = (
        args.out_dir / "CellFlow.pkl",
        args.out_dir / "pca.npz",
        args.out_dir / "pca_audit.json",
        args.out_dir / "config.json",
        args.out_dir / "training_summary.json",
        args.out_dir / "training_loss.csv",
        args.out_dir / "training_audit.json",
        args.out_dir / "fold_audit.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing train-only artifacts: {missing}")
    config = json.loads((args.out_dir / "config.json").read_text())
    if config["target_donor"] != args.target_donor:
        raise ValueError("Eval target donor differs from training artifact")
    cache_dir = args.cache_dir or _default_cache_dir(args.out_dir, "eval")
    configure_runtime(cache_dir)
    runtime = import_cellflow_runtime()
    context = fit_fold_context(
        args.balanced_dir, args.target_donor, args.allow_nonfrozen_fixture
    )
    config = verify_saved_training_artifacts(args.out_dir, context, args.target_donor)
    context.pca = load_pca(args.out_dir / "pca.npz")
    expected_pca_shape = (
        int(config["pca"]["n_components"]),
        int(config["gene_dim"]),
    )
    if context.pca.components_.shape != expected_pca_shape:
        raise ValueError(
            f"Saved PCA components have shape {context.pca.components_.shape}, "
            f"expected {expected_pca_shape}"
        )
    pca_arrays = (
        context.pca.mean_,
        context.pca.components_,
        context.pca.explained_variance_,
        context.pca.explained_variance_ratio_,
    )
    if not all(np.isfinite(values).all() for values in pca_arrays):
        raise ValueError("Saved PCA contains non-finite values")
    model = runtime["CellFlow"].load(args.out_dir)
    evaluation_started = time.time()
    rows = evaluate_fold(model, args.balanced_dir, context, args.target_donor)
    finalize_evaluation(
        args.out_dir,
        rows,
        args.target_donor,
        context.manifest,
        time.time() - evaluation_started,
    )


def main() -> None:
    args = parse_args()
    if args.resource_probe:
        if args.batch_size is not None:
            raise ValueError("Parent resource probe chooses its own batch size")
        if args.resource_probe_result is not None:
            raise ValueError("Resource probe cannot consume a prior probe result")
        run_resource_probe_parent(args)
        return
    if args.resource_probe_candidate:
        try:
            run_probe_candidate(args)
        except BaseException as error:
            if _resource_error(error):
                args.out_dir.parent.mkdir(parents=True, exist_ok=True)
                write_json(
                    args.out_dir.parent / f"{args.out_dir.name}_resource_failure.json",
                    {"type": type(error).__name__, "message": str(error)},
                )
                print(
                    f"RESOURCE_FAILURE: {type(error).__name__}: {error}",
                    file=sys.stderr,
                )
                raise SystemExit(86) from error
            raise
        return
    if args.eval_only:
        if args.batch_size is not None:
            raise ValueError("Eval-only reads the frozen training configuration")
        if args.resource_probe_result is not None:
            raise ValueError("Eval-only reads resource provenance from the checkpoint")
        run_eval_only(args)
        return
    run_train_or_full(args)


if __name__ == "__main__":
    main()
