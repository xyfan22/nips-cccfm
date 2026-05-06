"""Lightning data module backed by the StateICL dataset abstractions."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ..data.datasets import (
    DatasetConfig,
    SimplifiedMultiDataset,
    create_train_val_test_datasets,
    load_gene_list,
    worker_init_fn,
)

log = logging.getLogger(__name__)


class MultiDatasetDataModule(pl.LightningDataModule):
    """PyTorch Lightning wrapper that orchestrates multi-dataset sampling."""

    def __init__(
        self,
        dataset_configs: List[DatasetConfig],
        genelist_path: str,
        sample_size: int = 128,
        test_ratio: float = 0.2,
        val_ratio: float = 0.2,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        random_state: int = 42,
        cache_file: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.dataset_configs = dataset_configs
        self.genelist_path = genelist_path
        self.sample_size = sample_size
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.random_state = random_state
        self.cache_file = cache_file

        self.train_dataset: Optional[SimplifiedMultiDataset] = None
        self.val_dataset: Optional[SimplifiedMultiDataset] = None
        self.test_dataset: Optional[SimplifiedMultiDataset] = None

        self.target_genes = load_gene_list(genelist_path)
        self.n_genes = len(self.target_genes)
        log.info("Loaded %s genes from %s", self.n_genes, genelist_path)

    def setup(self, stage: Optional[str] = None) -> None:  # type: ignore[override]
        if stage in ("fit", None):
            log.info("Creating train/val/test datasets from %s configs", len(self.dataset_configs))
            self.train_dataset, self.val_dataset, self.test_dataset = create_train_val_test_datasets(
                dataset_configs=self.dataset_configs,
                target_genes=self.target_genes,
                sample_size=self.sample_size,
                test_ratio=self.test_ratio,
                val_ratio=self.val_ratio,
                random_state=self.random_state,
                cache_file=self.cache_file,
            )
            log.info("Train dataset: %s samples", len(self.train_dataset))
            log.info("Val dataset: %s samples", len(self.val_dataset))
            log.info("Test dataset: %s samples", len(self.test_dataset))
            if hasattr(self.train_dataset, "get_cache_stats"):
                log.info("Cache stats: %s", self.train_dataset.get_cache_stats())

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------
    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is not initialized; call setup('fit') first")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
            drop_last=True,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=min(2, self.num_workers),
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
            drop_last=False,
            worker_init_fn=worker_init_fn,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=min(2, self.num_workers),
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
            drop_last=False,
            worker_init_fn=worker_init_fn,
        )

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def get_split_info(self) -> Dict[str, List[str]]:
        if self.train_dataset is None:
            raise RuntimeError("Datasets have not been initialized. Call setup('fit') first.")
        return {
            "train_files": self.train_dataset.train_files,
            "validation_files": self.train_dataset.val_files,
            "test_files": self.train_dataset.test_files,
        }
