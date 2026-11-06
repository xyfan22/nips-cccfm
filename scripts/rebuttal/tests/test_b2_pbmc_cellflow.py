"""Production-path tests for the B2 PBMC CellFlow implementation."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import shutil
import sys
from pathlib import Path
from statistics import mean
from types import ModuleType, SimpleNamespace
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


REBUTTAL_DIR = Path(__file__).resolve().parents[1]


def load_script(filename: str, module_name: str) -> ModuleType:
    path = REBUTTAL_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


PREPARE = load_script("50_prepare_pbmc_cellflow_balanced.py", "b2_prepare")
RUNNER = load_script("51_run_pbmc_cellflow_fold.py", "b2_runner")
SUMMARIZE = load_script("52_summarize_pbmc_cellflow.py", "b2_summarize")

DONORS = ("Donor1", "Donor2", "Donor3")
CELL_TYPES = ("CT_A", "CT_B")
CYTOKINES = ("C1", "C2", "C3")
SEEDS = (7, 42, 314)
N_CELLS = 10
N_GENES = 8


def build_synthetic_h5ad(path: Path) -> None:
    """Build a valid sparse H5AD with extra categories and one excluded cytokine."""
    rng = np.random.default_rng(20260727)
    rows: list[dict[str, Any]] = []
    matrix_rows: list[np.ndarray] = []
    conditions = ("PBS", *CYTOKINES, "C_bad")
    donor_categories = list(DONORS)
    cell_categories = [*CELL_TYPES, "CT_UNUSED"]
    cytokine_categories = list(conditions)
    for donor_index, donor in enumerate(DONORS):
        for cell_index, cell_type in enumerate(CELL_TYPES):
            for condition_index, cytokine in enumerate(conditions):
                count = 12 if cytokine != "C_bad" else 4
                for replicate in range(count):
                    rows.append(
                        {
                            "donor": donor,
                            "cell_type": cell_type,
                            "cytokine": cytokine,
                            "treatment": "PBS" if cytokine == "PBS" else "cytokine",
                        }
                    )
                    signal = donor_index + cell_index + condition_index
                    counts = rng.poisson(3.0 + 0.15 * signal, size=N_GENES).astype(
                        np.float32
                    )
                    counts[(replicate + condition_index) % N_GENES] += 2.0
                    matrix_rows.append(counts)
    obs = pd.DataFrame(rows)
    obs["donor"] = pd.Categorical(obs["donor"], categories=donor_categories)
    obs["cell_type"] = pd.Categorical(obs["cell_type"], categories=cell_categories)
    obs["cytokine"] = pd.Categorical(obs["cytokine"], categories=cytokine_categories)
    obs["treatment"] = pd.Categorical(obs["treatment"], categories=["PBS", "cytokine"])
    obs.index = pd.Index([f"raw_{index}" for index in range(len(obs))], name="cell_id")
    var = pd.DataFrame(
        {"gene_id": [f"ENSG{index:05d}" for index in range(N_GENES)]},
        index=pd.Index([f"gene_{index}" for index in range(N_GENES)], name="feature"),
    )
    data = ad.AnnData(X=sp.csr_matrix(np.stack(matrix_rows)), obs=obs, var=var)
    data.uns["esm2_embeddings"] = {
        cytokine: rng.normal(size=2560).astype(np.float32)
        for cytokine in (*CYTOKINES, "C_bad")
    }
    data.write_h5ad(path, compression="gzip")


def make_prepare_config(input_path: Path, output_dir: Path) -> Any:
    hashes = PREPARE.hash_file(input_path, ("md5",))
    return PREPARE.PrepareConfig(
        input_path=input_path,
        output_dir=output_dir,
        seeds=SEEDS,
        cell_types=CELL_TYPES,
        n_cells=N_CELLS,
        min_cells=N_CELLS,
        expected_donors=len(DONORS),
        expected_cytokines=len(CYTOKINES),
        expected_excluded=("C_bad",),
        expected_input_size=input_path.stat().st_size,
        expected_input_md5=hashes["md5"],
        pca_n_components=4,
        pca_seed=42,
        allow_nonfrozen_fixture=True,
    )


@pytest.fixture()
def prepared_fixture(tmp_path: Path) -> SimpleNamespace:
    input_path = tmp_path / "synthetic_pbmc.h5ad"
    output_dir = tmp_path / "balanced"
    build_synthetic_h5ad(input_path)
    config = make_prepare_config(input_path, output_dir)
    manifest = PREPARE.prepare_balanced(config)
    return SimpleNamespace(
        input_path=input_path,
        balanced_dir=output_dir,
        config=config,
        manifest=manifest,
    )


def independent_selected_rows(input_path: Path, seed: int) -> np.ndarray:
    data = ad.read_h5ad(input_path)
    donor_codes = data.obs["donor"].cat.codes.to_numpy()
    cell_codes = data.obs["cell_type"].cat.codes.to_numpy()
    cytokine_codes = data.obs["cytokine"].cat.codes.to_numpy()
    all_cell_types = list(data.obs["cell_type"].cat.categories)
    all_cytokines = list(data.obs["cytokine"].cat.categories)
    cell_map = np.full(len(all_cell_types), -1, dtype=np.int16)
    for local, cell_type in enumerate(CELL_TYPES):
        cell_map[all_cell_types.index(cell_type)] = local
    condition_names = ("PBS", *CYTOKINES)
    condition_map = np.full(len(all_cytokines), -1, dtype=np.int16)
    for local, cytokine in enumerate(condition_names):
        condition_map[all_cytokines.index(cytokine)] = local
    cell_local = cell_map[cell_codes]
    condition_local = condition_map[cytokine_codes]
    mask = (cell_local >= 0) & (condition_local >= 0)
    group_ids = (donor_codes[mask] * len(CELL_TYPES) + cell_local[mask]) * len(
        condition_names
    ) + condition_local[mask]
    source_rows = np.flatnonzero(mask)
    selected: list[np.ndarray] = []
    for group_id in range(len(DONORS) * len(CELL_TYPES) * len(condition_names)):
        candidates = source_rows[group_ids == group_id]
        rng = np.random.default_rng(np.random.SeedSequence([seed, group_id]))
        selected.append(rng.choice(candidates, N_CELLS, replace=False))
    return np.sort(np.concatenate(selected).astype(np.int64))


def normalize_reference(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32).copy()
    values *= (1e4 / values.sum(axis=1))[:, None]
    return np.log1p(values)


def test_prepare_uses_exact_b1_rows_and_preserves_metadata(
    prepared_fixture: SimpleNamespace,
) -> None:
    manifest = prepared_fixture.manifest
    assert manifest["frozen_protocol"] is False
    assert manifest["excluded_cytokines"] == ["C_bad"]
    assert manifest["pca_n_components"] == 4
    raw = ad.read_h5ad(prepared_fixture.input_path)
    for seed in SEEDS:
        compact = ad.read_h5ad(prepared_fixture.balanced_dir / f"seed{seed}.h5ad")
        expected_rows = independent_selected_rows(prepared_fixture.input_path, seed)
        observed_rows = compact.obs["source_row"].to_numpy(dtype=np.int64)
        assert np.array_equal(observed_rows, expected_rows)
        assert compact.n_obs == len(DONORS) * len(CELL_TYPES) * 4 * N_CELLS
        assert compact.n_vars == N_GENES
        assert list(compact.var_names) == list(raw.var_names)
        assert list(compact.var["gene_id"]) == list(raw.var["gene_id"])
        assert list(compact.obs["cell_type"].cat.categories) == list(CELL_TYPES)
        assert list(compact.obs["cytokine"].cat.categories) == ["PBS", *CYTOKINES]
        assert set(compact.uns["esm2_embeddings"]) == set(CYTOKINES)
        assert "PBS" not in compact.uns["esm2_embeddings"]
        assert (
            PREPARE.row_digest(observed_rows)
            == manifest["seeds"][str(seed)]["row_digest_sha256"]
        )

    seed42 = ad.read_h5ad(prepared_fixture.balanced_dir / "seed42.h5ad")
    raw_dense = raw.X.toarray()
    for donor in DONORS:
        pbs_mask = (seed42.obs["donor"].astype(str).to_numpy() == donor) & seed42.obs[
            "is_control"
        ].to_numpy(dtype=bool)
        source_rows = seed42.obs.loc[pbs_mask, "source_row"].to_numpy(dtype=np.int64)
        expected_embedding = normalize_reference(raw_dense[source_rows]).mean(axis=0)
        observed_embedding = np.asarray(
            manifest["donor_embeddings_b2"][donor], dtype=np.float32
        )
        assert len(source_rows) == len(CELL_TYPES) * N_CELLS
        assert np.allclose(observed_embedding, expected_embedding, atol=3e-6, rtol=0)
        assert (
            PREPARE.row_digest(source_rows)
            == manifest["donor_pbs_row_digest_sha256"][donor]
        )


def test_prepare_is_deterministic_and_refuses_overwrite(
    prepared_fixture: SimpleNamespace, tmp_path: Path
) -> None:
    with pytest.raises(FileExistsError):
        PREPARE.prepare_balanced(prepared_fixture.config)
    second_dir = tmp_path / "balanced_second"
    second = PREPARE.prepare_balanced(
        make_prepare_config(prepared_fixture.input_path, second_dir)
    )
    for seed in SEEDS:
        first_rows = (
            ad.read_h5ad(prepared_fixture.balanced_dir / f"seed{seed}.h5ad")
            .obs["source_row"]
            .to_numpy()
        )
        second_rows = (
            ad.read_h5ad(second_dir / f"seed{seed}.h5ad").obs["source_row"].to_numpy()
        )
        assert np.array_equal(first_rows, second_rows)
        assert (
            second["seeds"][str(seed)]["row_digest_sha256"]
            == prepared_fixture.manifest["seeds"][str(seed)]["row_digest_sha256"]
        )


def test_fold_context_is_source_only_and_pbs_audited(
    prepared_fixture: SimpleNamespace,
) -> None:
    context = RUNNER.fit_fold_context(
        prepared_fixture.balanced_dir, "Donor1", allow_nonfrozen_fixture=True
    )
    assert context.source_donors == ["Donor2", "Donor3"]
    assert set(context.train_data.obs["donor"].astype(str)) == {"Donor2", "Donor3"}
    assert RUNNER.CELLFLOW_SPLIT_CELL_TYPE_KEY in context.train_data.obs
    assert np.array_equal(
        context.train_data.obs["cell_type"].astype(str).to_numpy(),
        context.train_data.obs[RUNNER.CELLFLOW_SPLIT_CELL_TYPE_KEY]
        .astype(str)
        .to_numpy(),
    )
    assert RUNNER.CELLFLOW_COVARIATE_ADAPTER["changes_values"] is False
    assert set(context.train_data.uns["donor_embeddings_b2"]) == {
        "Donor2",
        "Donor3",
    }
    assert context.pca.components_.shape == (4, N_GENES)
    assert context.pca_audit["target_rows_in_fit"] == 0
    assert context.pca_audit["fit_row_count"] == 2 * 2 * 4 * N_CELLS
    assert context.pca_audit["target_donor_embedding_row_count"] == 2 * N_CELLS
    assert (
        context.pca_audit["target_donor_embedding_row_digest_sha256"]
        == prepared_fixture.manifest["donor_pbs_row_digest_sha256"]["Donor1"]
    )
    transformed_mean = context.train_data.obsm["X_pca"].mean(axis=0)
    assert np.allclose(transformed_mean, 0.0, atol=2e-4)
    seed_data = ad.read_h5ad(prepared_fixture.balanced_dir / "seed7.h5ad")
    cloud_index = RUNNER.build_seed_cloud_index(seed_data, prepared_fixture.manifest)
    indexed = cloud_index.cloud("Donor1", "CT_A", "C1")
    direct_mask = (
        (seed_data.obs["donor"].astype(str).to_numpy() == "Donor1")
        & (seed_data.obs["cell_type"].astype(str).to_numpy() == "CT_A")
        & (seed_data.obs["cytokine"].astype(str).to_numpy() == "C1")
    )
    direct = RUNNER.to_dense(seed_data.X[direct_mask])
    direct_rows = seed_data.obs.loc[direct_mask, "source_row"].to_numpy()
    direct = direct[np.argsort(direct_rows)]
    assert np.array_equal(indexed, direct)
    assert np.allclose(
        cloud_index.centroid("Donor1", "CT_A", "C1"), direct.mean(axis=0)
    )


def test_resource_probe_provenance_requires_frozen_order(tmp_path: Path) -> None:
    path = tmp_path / "resource_probe.json"
    payload = {
        "target_donor": "Donor1",
        "selected_batch_size": 512,
        "attempts": [
            {"batch_size": 1024, "returncode": 86},
            {"batch_size": 512, "returncode": 0},
        ],
    }
    path.write_text(json.dumps(payload) + "\n")
    provenance = RUNNER.load_resource_probe_provenance(path, 512, False)
    assert provenance["selected_batch_size"] == 512
    assert len(provenance["sha256"]) == 64
    with pytest.raises(ValueError, match="authorize"):
        RUNNER.load_resource_probe_provenance(path, 256, False)
    payload["attempts"] = [
        {"batch_size": 512, "returncode": 0},
    ]
    path.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="frozen order"):
        RUNNER.load_resource_probe_provenance(path, 512, False)


def independent_distribution_metrics(
    pred: np.ndarray, target: np.ndarray
) -> dict[str, np.ndarray]:
    def sqdist(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.maximum(
            np.sum(left**2, axis=-1, keepdims=True)
            + np.swapaxes(np.sum(right**2, axis=-1, keepdims=True), 1, 2)
            - 2 * left @ np.swapaxes(right, 1, 2),
            0,
        )

    dpp, dtt, dpt = sqdist(pred, pred), sqdist(target, target), sqdist(pred, target)
    energy = (
        2 * np.sqrt(dpt + 1e-12).mean((1, 2))
        - np.sqrt(dpp + 1e-12).mean((1, 2))
        - np.sqrt(dtt + 1e-12).mean((1, 2))
    )
    mmd = np.zeros(len(pred))
    for bandwidth in (0.1, 1.0, 10.0):
        scale = 2 * bandwidth**2
        mmd += (
            np.exp(-dpp / scale).mean((1, 2))
            + np.exp(-dtt / scale).mean((1, 2))
            - 2 * np.exp(-dpt / scale).mean((1, 2))
        )
    mmd /= 3
    pred_centered = pred - pred.mean(axis=1, keepdims=True)
    target_centered = target - target.mean(axis=1, keepdims=True)
    dppc = sqdist(pred_centered, pred_centered)
    dttc = sqdist(target_centered, target_centered)
    dptc = sqdist(pred_centered, target_centered)
    centered_energy = (
        2 * np.sqrt(dptc + 1e-12).mean((1, 2))
        - np.sqrt(dppc + 1e-12).mean((1, 2))
        - np.sqrt(dttc + 1e-12).mean((1, 2))
    )

    def moments(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mu = values.mean(axis=1)
        centered = values - mu[:, None]
        variance = (centered**2).mean(axis=1)
        std = np.sqrt(variance)
        skew = np.where(
            std > 1e-12,
            (centered**3).mean(axis=1) / np.maximum(std**3, 1e-12),
            0,
        )
        return mu, variance, skew

    pred_moments = moments(pred)
    target_moments = moments(target)
    moment = sum(
        np.abs(left - right).mean(axis=1)
        for left, right in zip(pred_moments, target_moments, strict=True)
    )
    return {
        "mmd": mmd,
        "energy": energy,
        "centered_energy": centered_energy,
        "moment": moment,
    }


def test_metrics_match_independent_b1_definitions() -> None:
    rng = np.random.default_rng(11)
    true_clouds = rng.normal(size=(3, 7, 6)).astype(np.float32)
    true_clouds += np.arange(3, dtype=np.float32)[:, None, None] * 0.2
    pred_clouds = true_clouds + rng.normal(scale=0.15, size=true_clouds.shape)
    pred_clouds = pred_clouds.astype(np.float32)
    control_mean = rng.normal(size=6).astype(np.float32)
    observed = RUNNER.evaluate_cloud_batch(pred_clouds, true_clouds, control_mean)
    distribution = independent_distribution_metrics(pred_clouds, true_clouds)
    pred_centroids = pred_clouds.mean(axis=1)
    true_centroids = true_clouds.mean(axis=1)
    perturbation_reference = true_centroids.mean(axis=0)
    for index, row in enumerate(observed):
        expected_ctrl = RUNNER.safe_pearson(
            pred_centroids[index] - control_mean,
            true_centroids[index] - control_mean,
        )
        expected_pert = RUNNER.safe_pearson(
            pred_centroids[index] - perturbation_reference,
            true_centroids[index] - perturbation_reference,
        )
        distances = np.linalg.norm(true_centroids - pred_centroids[index][None], axis=1)
        expected_cacc = np.sum(distances > distances[index]) / 2
        assert row["pearson_ctrl"] == pytest.approx(expected_ctrl)
        assert row["pearson_pert"] == pytest.approx(expected_pert)
        assert row["centroid_accuracy"] == pytest.approx(expected_cacc)
        assert row["mse"] == pytest.approx(
            np.mean((pred_centroids[index] - true_centroids[index]) ** 2)
        )
        for metric in ("mmd", "energy", "centered_energy", "moment"):
            assert row[metric] == pytest.approx(
                float(distribution[metric][index]), abs=2e-6
            )
        assert all(math.isfinite(value) for value in row.values())
    assert any(
        abs(row["pearson_ctrl"] - row["pearson_pert"]) > 1e-4 for row in observed
    )


def make_metric_rows(
    manifest: dict[str, Any], target_donor: str, donor_offset: float = 0.0
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    method_offsets = {
        "target_control_copy": 0.10,
        "shifted_control": 0.20,
        "cellflow": 0.30,
    }
    for method in RUNNER.FROZEN_METHODS:
        for seed in manifest["sample_seeds"]:
            for cell_index, cell_type in enumerate(manifest["cell_types"]):
                for cytokine_index, cytokine in enumerate(manifest["cytokines"]):
                    value = (
                        method_offsets[method]
                        + donor_offset
                        + int(seed) * 1e-6
                        + cell_index * 1e-3
                        + cytokine_index * 1e-4
                    )
                    rows.append(
                        {
                            "method": method,
                            "sample_seed": int(seed),
                            "target_donor": target_donor,
                            "cell_type": cell_type,
                            "cytokine": cytokine,
                            "n_cells": int(manifest["n_cells_per_group"]),
                            "n_source_donors": len(manifest["donors"]) - 1,
                            "condition_id": (
                                f"seed{int(seed)}|{target_donor}|{cell_type}|{cytokine}"
                            ),
                            **{metric: value for metric in RUNNER.METRICS},
                        }
                    )
    return rows


def test_validate_fold_rows_rejects_wrong_keys_duplicates_and_nonfinite(
    prepared_fixture: SimpleNamespace,
) -> None:
    rows = make_metric_rows(prepared_fixture.manifest, "Donor1")
    RUNNER.validate_fold_rows(rows, prepared_fixture.manifest, "Donor1")

    wrong_seed = [dict(row) for row in rows]
    wrong_seed[0]["sample_seed"] = 99
    wrong_seed[0][
        "condition_id"
    ] = f"seed99|Donor1|{wrong_seed[0]['cell_type']}|{wrong_seed[0]['cytokine']}"
    with pytest.raises(ValueError, match="incomplete or duplicate"):
        RUNNER.validate_fold_rows(wrong_seed, prepared_fixture.manifest, "Donor1")

    duplicate = [dict(row) for row in rows]
    duplicate[1] = dict(duplicate[0])
    with pytest.raises(ValueError, match="incomplete or duplicate"):
        RUNNER.validate_fold_rows(duplicate, prepared_fixture.manifest, "Donor1")

    nonfinite = [dict(row) for row in rows]
    nonfinite[0]["energy"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        RUNNER.validate_fold_rows(nonfinite, prepared_fixture.manifest, "Donor1")


class FakeModel:
    def __init__(self, payload: bytes = b"synthetic-cellflow-checkpoint") -> None:
        self.payload = payload

    def save(self, directory: Path, overwrite: bool = False) -> None:
        path = Path(directory) / "CellFlow.pkl"
        if path.exists() and not overwrite:
            raise RuntimeError("refusing overwrite")
        path.write_bytes(self.payload)


def training_values() -> tuple[dict[str, Any], list[float]]:
    losses = np.linspace(2.0, 1.0, RUNNER.FROZEN_ITERATIONS).tolist()
    window = len(losses) // 10
    summary = {
        "iterations": RUNNER.FROZEN_ITERATIONS,
        "batch_size": 128,
        "model_seed": RUNNER.FROZEN_MODEL_SEED,
        "wall_seconds": 1.0,
        "first_10pct_mean_loss": float(np.mean(losses[:window])),
        "last_10pct_mean_loss": float(np.mean(losses[-window:])),
        "loss_decreased": True,
        "all_losses_finite": True,
    }
    return summary, losses


def write_fold_artifacts(
    root: Path,
    prepared: SimpleNamespace,
    donor: str,
    donor_offset: float,
    include_evaluation: bool = True,
) -> tuple[Path, list[dict[str, Any]], Any]:
    fold_dir = root / donor
    context = RUNNER.fit_fold_context(
        prepared.balanced_dir, donor, allow_nonfrozen_fixture=True
    )
    training_summary, losses = training_values()
    training_audit = RUNNER.save_training_artifacts(
        fold_dir,
        FakeModel(payload=f"checkpoint-{donor}".encode()),
        context,
        donor,
        128,
        training_summary,
        losses,
        {"devices": ["cuda:0"]},
        {
            "path": "synthetic fixture",
            "sha256": "0" * 64,
            "target_donor": "Donor1",
            "selected_batch_size": 128,
        },
    )
    rows = make_metric_rows(prepared.manifest, donor, donor_offset)
    fold_audit = {
        **training_audit,
        "evaluation_complete": include_evaluation,
        "balanced_manifest_sha256": RUNNER.hash_file(
            prepared.balanced_dir / "manifest.json"
        )["sha256"],
        "cache_dir": f"/tmp/{donor}",
    }
    if include_evaluation:
        RUNNER.save_evaluation_artifacts(
            fold_dir / "evaluation", rows, donor, prepared.manifest
        )
        fold_audit.update(
            {
                "evaluation_wall_seconds": 1.0,
                "metrics_sha256": RUNNER.hash_file(
                    fold_dir / "evaluation" / "per_condition_metrics.csv"
                )["sha256"],
                "evaluation_audit_sha256": RUNNER.hash_file(
                    fold_dir / "evaluation" / "evaluation_audit.json"
                )["sha256"],
                "evaluation_summary_sha256": RUNNER.hash_file(
                    fold_dir / "evaluation" / "summary.csv"
                )["sha256"],
            }
        )
    RUNNER.write_json(fold_dir / "fold_audit.json", fold_audit)
    return fold_dir, rows, context


def write_b1_controls(root: Path, fold_rows: dict[str, list[dict[str, Any]]]) -> None:
    fields = list(fold_rows[next(iter(fold_rows))][0].keys())
    for seed in SEEDS:
        seed_dir = root / f"seed{seed}"
        seed_dir.mkdir(parents=True)
        rows = [
            row
            for donor_rows in fold_rows.values()
            for row in donor_rows
            if row["method"] in SUMMARIZE.CONTROL_METHODS
            and int(row["sample_seed"]) == seed
        ]
        with (seed_dir / "per_condition_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def make_summary_args(
    result_root: Path,
    b1_root: Path,
    output_dir: Path,
    donors: list[str],
) -> argparse.Namespace:
    return argparse.Namespace(
        result_root=result_root,
        b1_controls_dir=b1_root,
        output_dir=output_dir,
        donors=donors,
        control_tolerance=5e-4,
        allow_nonfrozen_fixture=True,
        expected_cell_types=len(CELL_TYPES),
        expected_cytokines=len(CYTOKINES),
        expected_n_cells=N_CELLS,
        expected_seeds=list(SEEDS),
    )


def test_summarizer_audits_artifacts_and_aggregates_by_donor(
    prepared_fixture: SimpleNamespace, tmp_path: Path
) -> None:
    result_root = tmp_path / "folds"
    fold_rows: dict[str, list[dict[str, Any]]] = {}
    for index, donor in enumerate(DONORS):
        _, rows, _ = write_fold_artifacts(
            result_root, prepared_fixture, donor, donor_offset=index * 0.05
        )
        fold_rows[donor] = rows
    b1_root = tmp_path / "b1"
    write_b1_controls(b1_root, fold_rows)
    output_dir = tmp_path / "summary"
    args = make_summary_args(result_root, b1_root, output_dir, list(DONORS))
    audit = SUMMARIZE.summarize(args)
    assert audit["status"] == "AUDIT_PASS"
    assert audit["control_audit"]["n_control_keys"] == (
        len(DONORS) * 2 * len(SEEDS) * len(CELL_TYPES) * len(CYTOKINES)
    )
    assert (
        max(audit["control_audit"]["max_absolute_difference_by_metric"].values()) == 0
    )
    donor_rows = SUMMARIZE.read_csv(output_dir / "donor_summary.csv")
    global_rows = SUMMARIZE.read_csv(output_dir / "global_summary.csv")
    cellflow_donor = [
        float(row["energy"]) for row in donor_rows if row["method"] == "cellflow"
    ]
    cellflow_global = next(row for row in global_rows if row["method"] == "cellflow")
    assert float(cellflow_global["energy_mean"]) == pytest.approx(mean(cellflow_donor))
    assert int(cellflow_global["n_donors"]) == len(DONORS)
    paired = SUMMARIZE.read_csv(output_dir / "paired_donor_summary.csv")
    assert all(int(row["n_donors"]) == len(DONORS) for row in paired)
    with pytest.raises(FileExistsError):
        SUMMARIZE.summarize(args)


def test_summarizer_rejects_pca_leakage_missing_seed_and_nonfinite(
    prepared_fixture: SimpleNamespace, tmp_path: Path
) -> None:
    original_root = tmp_path / "original"
    fold_dir, _, _ = write_fold_artifacts(
        original_root, prepared_fixture, "Donor1", donor_offset=0.0
    )

    leaked_root = tmp_path / "leaked"
    shutil.copytree(original_root, leaked_root)
    leaked_audit_path = leaked_root / "Donor1" / "pca_audit.json"
    leaked_audit = json.loads(leaked_audit_path.read_text())
    leaked_audit["target_rows_in_fit"] = 1
    leaked_audit_path.write_text(json.dumps(leaked_audit, indent=2) + "\n")
    with pytest.raises(ValueError, match="PCA leakage"):
        SUMMARIZE.load_and_audit_fold(
            leaked_root / "Donor1",
            "Donor1",
            SEEDS,
            len(CELL_TYPES),
            len(CYTOKINES),
            N_CELLS,
            True,
        )

    missing_root = tmp_path / "missing"
    shutil.copytree(original_root, missing_root)
    metrics_path = missing_root / "Donor1" / "evaluation" / "per_condition_metrics.csv"
    rows = SUMMARIZE.read_csv(metrics_path)
    SUMMARIZE.write_csv(metrics_path, list(rows[0]), rows[1:])
    with pytest.raises(ValueError, match="rows, expected"):
        SUMMARIZE.load_and_audit_fold(
            missing_root / "Donor1",
            "Donor1",
            SEEDS,
            len(CELL_TYPES),
            len(CYTOKINES),
            N_CELLS,
            True,
        )

    nan_root = tmp_path / "nan"
    shutil.copytree(original_root, nan_root)
    metrics_path = nan_root / "Donor1" / "evaluation" / "per_condition_metrics.csv"
    rows = SUMMARIZE.read_csv(metrics_path)
    rows[0]["energy"] = "nan"
    SUMMARIZE.write_csv(metrics_path, list(rows[0]), rows)
    with pytest.raises(ValueError, match="non-finite"):
        SUMMARIZE.load_and_audit_fold(
            nan_root / "Donor1",
            "Donor1",
            SEEDS,
            len(CELL_TYPES),
            len(CYTOKINES),
            N_CELLS,
            True,
        )
    assert fold_dir.is_dir()


def test_eval_only_loads_checkpoint_and_writes_atomic_evaluation(
    prepared_fixture: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result_root = tmp_path / "train_only"
    fold_dir, rows, context = write_fold_artifacts(
        result_root,
        prepared_fixture,
        "Donor1",
        donor_offset=0.0,
        include_evaluation=False,
    )
    loaded: list[Path] = []

    class FakeCellFlow:
        @classmethod
        def load(cls, path: Path) -> object:
            loaded.append(Path(path))
            return object()

    monkeypatch.setattr(RUNNER, "configure_runtime", lambda _: None)
    monkeypatch.setattr(
        RUNNER,
        "import_cellflow_runtime",
        lambda: {"CellFlow": FakeCellFlow},
    )
    monkeypatch.setattr(RUNNER, "fit_fold_context", lambda *_args, **_kwargs: context)
    monkeypatch.setattr(RUNNER, "evaluate_fold", lambda *_args, **_kwargs: rows)
    args = argparse.Namespace(
        balanced_dir=prepared_fixture.balanced_dir,
        target_donor="Donor1",
        out_dir=fold_dir,
        cache_dir=tmp_path / "cache",
        batch_size=None,
        allow_nonfrozen_fixture=True,
    )
    RUNNER.run_eval_only(args)
    assert loaded == [fold_dir]
    assert (fold_dir / "evaluation" / "per_condition_metrics.csv").is_file()
    assert (fold_dir / "evaluation" / "summary.csv").is_file()
    fold_audit = json.loads((fold_dir / "fold_audit.json").read_text())
    assert fold_audit["evaluation_complete"] is True
    assert len(fold_audit["metrics_sha256"]) == 64
    with pytest.raises(FileExistsError):
        RUNNER.run_eval_only(args)


def test_resource_probe_preserves_nonresource_failure_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailedProcess:
        returncode = 2
        stdout = "candidate stdout"
        stderr = "API failure, not OOM"

    monkeypatch.setattr(
        RUNNER.subprocess, "run", lambda *_args, **_kwargs: FailedProcess()
    )
    out_dir = tmp_path / "probe"
    args = argparse.Namespace(
        out_dir=out_dir,
        balanced_dir=tmp_path / "balanced",
        target_donor="Donor1",
        allow_nonfrozen_fixture=True,
    )
    with pytest.raises(RuntimeError, match="non-resource"):
        RUNNER.run_resource_probe_parent(args)
    assert out_dir.is_dir()
    assert (out_dir / "batch1024.log").is_file()
    failure = json.loads((out_dir / "resource_probe_failure.json").read_text())
    assert failure["attempts"][0]["returncode"] == 2
    assert "API failure" in (out_dir / "batch1024.log").read_text()


@pytest.mark.skipif(
    os.environ.get("CELLFLOW_SCHOOL_TEST") != "1",
    reason="Requires the verified school CellFlow/JAX GPU environment",
)
def test_school_cellflow_unseen_donor_api_smoke() -> None:
    import optax
    from cellflow.model import CellFlow

    rng = np.random.default_rng(42)
    obs_rows: list[dict[str, Any]] = []
    values: list[np.ndarray] = []
    for donor in DONORS:
        for cell_type in CELL_TYPES:
            for cytokine in ("PBS", *CYTOKINES):
                for _ in range(8):
                    obs_rows.append(
                        {
                            "donor": donor,
                            "cell_type": cell_type,
                            "cytokine": cytokine,
                            "is_control": cytokine == "PBS",
                        }
                    )
                    values.append(rng.normal(size=10).astype(np.float32))
    data = ad.AnnData(X=np.stack(values), obs=pd.DataFrame(obs_rows))
    data.obsm["X_pca"] = np.asarray(data.X).copy()
    data.uns["esm2_embeddings"] = {
        cytokine: rng.normal(size=16).astype(np.float32) for cytokine in CYTOKINES
    }
    data.uns["donor_embeddings_b2"] = {
        donor: rng.normal(size=10).astype(np.float32) for donor in DONORS
    }
    RUNNER.verify_cellflow_covariate_adapter()
    RUNNER.attach_cellflow_split_alias(data.obs)
    train = data[data.obs["donor"] != "Donor1"].copy()
    train.uns["donor_embeddings_b2"] = {
        donor: data.uns["donor_embeddings_b2"][donor].copy()
        for donor in ("Donor2", "Donor3")
    }
    assert "Donor1" not in set(train.obs["donor"].astype(str))
    assert "Donor1" not in train.uns["donor_embeddings_b2"]
    model = CellFlow(train, solver="otfm")
    model.prepare_data(
        sample_rep="X_pca",
        control_key="is_control",
        perturbation_covariates={"cytokine_treatment": ("cytokine",)},
        perturbation_covariate_reps={"cytokine_treatment": "esm2_embeddings"},
        sample_covariates=["donor"],
        sample_covariate_reps={"donor": "donor_embeddings_b2"},
        split_covariates=["donor", RUNNER.CELLFLOW_SPLIT_CELL_TYPE_KEY],
        max_combination_length=1,
        null_value=0.0,
    )
    model.prepare_model(
        pooling="mean",
        layers_before_pool={
            "cytokine_treatment": {"layer_type": "mlp", "dims": [16]},
            "donor": {"layer_type": "mlp", "dims": [16]},
        },
        layers_after_pool={"layer_type": "mlp", "dims": [16]},
        condition_embedding_dim=8,
        cond_output_dropout=0.0,
        time_freqs=16,
        time_encoder_dims=(16,),
        hidden_dims=(16,),
        decoder_dims=(16,),
        probability_path={"constant_noise": 0.1},
        optimizer=optax.adam(1e-3),
        seed=42,
    )
    model.train(num_iterations=3, batch_size=16, valid_freq=10)
    control = data[
        (data.obs["donor"] == "Donor1")
        & (data.obs["cell_type"] == "CT_A")
        & data.obs["is_control"]
    ].copy()
    control.uns["donor_embeddings_b2"] = {
        "Donor1": data.uns["donor_embeddings_b2"]["Donor1"].copy()
    }
    covariates = pd.DataFrame(
        [
            {
                "donor": "Donor1",
                "cell_type": "CT_A",
                "cytokine": "C1",
                "is_control": False,
                "condition_id": "unseen_donor_C1",
            }
        ]
    )
    RUNNER.attach_cellflow_split_alias(covariates)
    assert covariates[RUNNER.CELLFLOW_SPLIT_CELL_TYPE_KEY].tolist() == ["CT_A"]
    predictions = model.predict(
        adata=control,
        covariate_data=covariates,
        sample_rep="X_pca",
        condition_id_key="condition_id",
        throw=True,
    )
    assert isinstance(predictions, dict)
    predicted = np.asarray(predictions["unseen_donor_C1"])
    assert predicted.shape == (control.n_obs, 10)
    assert np.isfinite(predicted).all()
