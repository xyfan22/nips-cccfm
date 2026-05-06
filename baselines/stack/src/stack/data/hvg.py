"""Highly variable gene computation utilities."""
from __future__ import annotations

import gc
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import scipy.sparse as sp

from .gene_processing import filter_gene_names, get_gene_names_from_h5, safe_decode_array

log = logging.getLogger(__name__)


__all__ = [
    "HVGComputationConfig",
    "compute_analytic_pearson_residuals",
    "compute_hvg_union",
]


@dataclass(slots=True)
class HVGComputationConfig:
    dataset_paths: List[str]
    gene_name_col: Optional[str] = None
    filter_organism: bool = True


def compute_analytic_pearson_residuals(X: np.ndarray, theta: float = 100.0) -> np.ndarray:
    if X.shape[0] == 0 or X.shape[1] == 0:
        return np.empty_like(X, dtype=np.float32)

    X = X.astype(np.float32)
    sequencing_depths = np.sum(X, axis=1, keepdims=True).clip(min=1e-6)
    total_counts_per_gene = np.sum(X, axis=0, keepdims=True)
    total_sequencing_depth = np.sum(sequencing_depths)
    gene_abundances = (total_counts_per_gene / total_sequencing_depth).clip(min=1e-6)
    expected_counts = sequencing_depths * gene_abundances
    variance = (expected_counts + (expected_counts ** 2) / theta).clip(min=1e-6)

    residuals = (X - expected_counts) / np.sqrt(variance)
    residuals = np.nan_to_num(residuals, nan=0.0, posinf=0.0, neginf=0.0)
    return residuals.astype(np.float32)


def compute_hvg_union(
    dataset_configs: List["DatasetConfig"],
    n_top_genes: int = 1000,
    theta: float = 100.0,
    max_cells_per_file: int = 10000,
    output_path: Optional[str] = None,
    random_state: int = 42,
    filter_gene_names_flag: bool = True,
) -> List[str]:
    from .datasets import DatasetConfig  # Circular import guard

    all_hvgs = set()
    rng = np.random.RandomState(random_state)
    processed_files = 0

    for config_idx, config in enumerate(dataset_configs):
        log.info("Processing dataset %s/%s: %s", config_idx + 1, len(dataset_configs), config.path)
        h5ad_files = sorted({*Path(config.path).glob("*.h5ad"), *Path(config.path).glob("*.h5")})

        for h5ad_file in h5ad_files:
            try:
                processed_files += 1
                log.info("Processing file %s: %s", processed_files, h5ad_file.name)

                with h5py.File(h5ad_file, "r") as handle:
                    if config.filter_organism:
                        obs = handle["obs"]
                        if "organism" not in obs:
                            log.warning("'organism' column not found in %s, skipping", h5ad_file.name)
                            continue

                        organism_ds = obs["organism"]
                        if "categories" in organism_ds:
                            organism_categories = safe_decode_array(organism_ds["categories"][:])
                            organism_codes = organism_ds["codes"][:]
                            organism_values = organism_categories[organism_codes]
                        else:
                            organism_values = safe_decode_array(organism_ds[:])

                        human_mask = organism_values == "Homo sapiens"
                        if not human_mask.any():
                            log.warning("No Homo sapiens cells found in %s, skipping", h5ad_file.name)
                            continue
                    else:
                        X = handle["X"]
                        n_cells = X.shape[0] if hasattr(X, "shape") else X.attrs["shape"][0]
                        human_mask = np.ones(n_cells, dtype=bool)

                    use_raw = "raw" in handle and "var" in handle["raw"]
                    gene_names = get_gene_names_from_h5(handle, config.gene_name_col, use_raw=use_raw)
                    if gene_names is None:
                        log.warning("Could not find gene names in %s, skipping", h5ad_file.name)
                        continue

                    all_valid_indices = np.where(human_mask)[0]
                    num_valid_cells = len(all_valid_indices)
                    if num_valid_cells == 0:
                        continue

                    if num_valid_cells > max_cells_per_file:
                        indices_to_load = rng.choice(all_valid_indices, size=max_cells_per_file, replace=False)
                        indices_to_load = np.sort(indices_to_load)
                    else:
                        indices_to_load = all_valid_indices

                    X_group = handle["raw"]["X"] if use_raw else handle["X"]
                    if hasattr(X_group, "shape"):
                        expr_data = X_group[indices_to_load, :]
                    else:
                        attrs = dict(X_group.attrs)
                        if attrs.get("encoding-type") == "csr_matrix":
                            indptrs = X_group["indptr"][:]
                            data_h5 = X_group["data"]
                            indices_h5 = X_group["indices"]

                            n_selected = len(indices_to_load)
                            n_genes = len(gene_names)
                            expr_data = np.zeros((n_selected, n_genes), dtype=np.float32)

                            for i, row_idx in enumerate(indices_to_load):
                                start_ptr = indptrs[row_idx]
                                end_ptr = indptrs[row_idx + 1]
                                if end_ptr > start_ptr:
                                    cell_indices = indices_h5[start_ptr:end_ptr]
                                    cell_values = data_h5[start_ptr:end_ptr].astype(np.float32)
                                    expr_data[i, cell_indices] = cell_values
                        else:
                            continue

                    if sp.issparse(expr_data):
                        expr_data = expr_data.toarray()
                    expr_data = expr_data.astype(np.float32)

                    if filter_gene_names_flag:
                        good_gene_names = filter_gene_names(gene_names.tolist())
                        good_gene_set = set(good_gene_names)
                        good_gene_indices = np.array([i for i, gene in enumerate(gene_names) if gene in good_gene_set])
                        if len(good_gene_indices) == 0:
                            continue

                        filtered_gene_names = gene_names[good_gene_indices]
                        filtered_expr_data = expr_data[:, good_gene_indices]
                    else:
                        filtered_gene_names = gene_names
                        filtered_expr_data = expr_data

                    if filtered_expr_data.shape[0] == 0 or filtered_expr_data.shape[1] == 0:
                        continue

                    pearson_residuals = compute_analytic_pearson_residuals(filtered_expr_data, theta=theta)
                    gene_variances = np.var(pearson_residuals, axis=0)
                    n_genes_to_select = min(n_top_genes, len(filtered_gene_names))
                    top_indices = np.argsort(gene_variances)[-n_genes_to_select:]
                    hvg_genes = [filtered_gene_names[i] for i in top_indices]

                    all_hvgs.update(hvg_genes)
                    del expr_data, filtered_expr_data, pearson_residuals
                    gc.collect()

                    if len(all_hvgs) > 15000:
                        break

            except Exception as error:  # pragma: no cover - logging
                log.exception("Error processing %s: %s", h5ad_file.name, error)
                continue

    union_genes = sorted(all_hvgs)
    log.info("Final HVG union contains %s genes from %s files", len(union_genes), processed_files)

    if output_path:
        with open(output_path, "wb") as handle:
            pickle.dump(union_genes, handle)
        log.info("Saved gene list to %s", output_path)

    return union_genes
