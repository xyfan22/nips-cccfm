"""Lightning data module for StateICL fine-tuning."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ..data.finetuning import datasets as finetuning_datasets

log = logging.getLogger(__name__)


class FinetuneDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module that orchestrates fine-tuning datasets."""

    def __init__(
        self,
        dataset_configs: List[finetuning_datasets.DatasetConfig],
        genelist_path: str,
        sample_size: int = 128,
        replacement_ratio: float = 0.25,
        intra_file_replacement_prob: float = 1.0,
        min_cells_per_group: int = 128,
        test_ratio: float = 0.2,
        val_ratio: float = 0.2,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        resample_each_epoch: bool = True,
        random_state: int = 42,
        cache_file: Optional[str] = None,
        max_memory_gb: Optional[float] = None,
        cache_ratio: float = 0.3,
        block_size: int = 1000,
    ) -> None:
        super().__init__()
        self.dataset_configs = dataset_configs
        self.genelist_path = genelist_path
        self.sample_size = sample_size
        self.replacement_ratio = replacement_ratio
        self.intra_file_replacement_prob = intra_file_replacement_prob
        self.min_cells_per_group = min_cells_per_group
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.resample_each_epoch = resample_each_epoch
        self.random_state = random_state
        self.cache_file = cache_file
        self.max_memory_gb = max_memory_gb
        self.cache_ratio = cache_ratio
        self.block_size = block_size

        self.train_dataset: Optional[finetuning_datasets.MultiDatasetSplittableDataset] = None
        self.val_dataset: Optional[finetuning_datasets.MultiDatasetSplittableDataset] = None
        self.test_dataset: Optional[finetuning_datasets.MultiDatasetSplittableDataset] = None

        self.target_genes = finetuning_datasets.load_gene_list(genelist_path)
        self.n_genes = len(self.target_genes)
        log.info("Loaded %s genes for fine-tuning from %s", self.n_genes, genelist_path)

    def setup(self, stage: Optional[str] = None) -> None:  # type: ignore[override]
        if stage in ("fit", None):
            log.info("Preparing fine-tuning datasets from %s configs", len(self.dataset_configs))
            (
                self.train_dataset,
                self.val_dataset,
                self.test_dataset,
            ) = finetuning_datasets.create_train_val_test_datasets(
                dataset_configs=self.dataset_configs,
                target_genes=self.target_genes,
                genelist_path=self.genelist_path,
                sample_size=self.sample_size,
                replacement_ratio=self.replacement_ratio,
                intra_file_replacement_prob=self.intra_file_replacement_prob,
                min_cells_per_group=self.min_cells_per_group,
                test_ratio=self.test_ratio,
                val_ratio=self.val_ratio,
                random_state=self.random_state,
                cache_file=self.cache_file,
                max_memory_gb=self.max_memory_gb,
                cache_ratio=self.cache_ratio,
                block_size=self.block_size,
            )
            log.info("Fine-tuning splits prepared: train=%s val=%s test=%s", len(self.train_dataset), len(self.val_dataset), len(self.test_dataset))

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------
    def train_dataloader(self) -> DataLoader:  # type: ignore[override]
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
            worker_init_fn=finetuning_datasets.worker_init_fn,
        )

    def val_dataloader(self) -> Optional[DataLoader]:  # type: ignore[override]
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
            worker_init_fn=finetuning_datasets.worker_init_fn,
        )

    def test_dataloader(self) -> Optional[DataLoader]:  # type: ignore[override]
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
            worker_init_fn=finetuning_datasets.worker_init_fn,
        )

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def get_split_info(self) -> Dict[str, List[str]]:
        if self.train_dataset is None:
            raise RuntimeError("Datasets have not been initialized. Call setup('fit') first.")
        return {
            "train_groups": getattr(self.train_dataset, "train_groups", []),
            "validation_groups": getattr(self.train_dataset, "val_groups", []),
            "test_groups": getattr(self.train_dataset, "test_groups", []),
        }


# Backwards compatibility -------------------------------------------------------------------------
class MultiDatasetDataModule(FinetuneDataModule):
    """Legacy alias retained for existing fine-tuning scripts."""

    pass
