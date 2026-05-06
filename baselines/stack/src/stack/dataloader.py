"""Compatibility layer exposing dataset utilities from :mod:`stack.data`."""
from __future__ import annotations

from .data.datasets import (
    DatasetConfig,
    SimplifiedDatasetCache,
    SimplifiedMultiDataset,
    TestSamplerDataset,
    compute_analytic_pearson_residuals,
    compute_and_save_hvg_union,
    compute_hvg_union,
    create_datasets_from_gene_list,
    create_example_configs,
    create_train_val_test_datasets,
    load_gene_list,
    reset_h5_handle_pool,
    worker_init_fn,
)
from .data.h5_manager import _get_h5_handle, _reset_h5_handle_pool, _worker_init_fn

__all__ = [
    "DatasetConfig",
    "SimplifiedDatasetCache",
    "SimplifiedMultiDataset",
    "TestSamplerDataset",
    "compute_analytic_pearson_residuals",
    "compute_hvg_union",
    "compute_and_save_hvg_union",
    "create_datasets_from_gene_list",
    "create_example_configs",
    "create_train_val_test_datasets",
    "load_gene_list",
    "reset_h5_handle_pool",
    "worker_init_fn",
    "_worker_init_fn",
    "_get_h5_handle",
    "_reset_h5_handle_pool",
]
