"""Shared helpers for the StateICL training pipeline."""
from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch

from ..data.datasets import DatasetConfig

log = logging.getLogger(__name__)


@dataclass
class LocalizationContext:
    """Tracks temporary paths created during dataset localization."""

    dataset_configs: List[str]
    cache_file: Optional[str]
    local_job_dir: Optional[Path]
    job_id: Optional[str]
    enabled: bool

    def stage_out(self, save_dir: str, job_id: str) -> None:
        """Copy checkpoints back and remove temporary directories."""
        if not self.enabled or self.local_job_dir is None:
            return

        log.info("=" * 80)
        log.info("JOB FINISHED. Staging out results...")

        local_checkpoints = Path(save_dir)
        remote_save_dir = Path("../permanent_storage/checkpoints/run_" + str(job_id))
        remote_save_dir.mkdir(parents=True, exist_ok=True)

        if local_checkpoints.exists():
            log.info("Copying checkpoints from %s to %s", local_checkpoints, remote_save_dir)
            shutil.copytree(local_checkpoints, remote_save_dir, dirs_exist_ok=True)

        log.info("Cleaning up local job directory...")
        shutil.rmtree(self.local_job_dir, ignore_errors=True)
        log.info("Cleanup complete.")


def parse_dataset_configs(config_strings: Iterable[str]) -> List[DatasetConfig]:
    """Parse CLI dataset configuration strings into :class:`DatasetConfig` objects."""
    configs: List[DatasetConfig] = []

    for config_str in config_strings:
        parts = config_str.split(":")
        if not parts or not parts[0]:
            raise ValueError(f"Invalid config string: {config_str}")

        path = parts[0]
        filter_organism = True
        gene_name_col: Optional[str] = None

        if len(parts) > 1 and parts[1]:
            filter_organism = parts[1].lower() == "true"
        if len(parts) > 2 and parts[2]:
            gene_name_col = parts[2]

        config = DatasetConfig(path=path, filter_organism=filter_organism, gene_name_col=gene_name_col)
        log.info(
            "Added dataset config: path=%s, filter_organism=%s, gene_col=%s",
            path,
            filter_organism,
            gene_name_col,
        )
        configs.append(config)

    return configs


def localize_datasets(
    dataset_configs: List[str],
    cache_file: Optional[str],
    local_temp_dir: str,
    disable_localization: bool,
) -> LocalizationContext:
    """Optionally copy datasets to a local scratch directory for faster I/O."""
    if disable_localization:
        log.info("Data localization disabled. Using remote paths directly.")
        return LocalizationContext(dataset_configs, cache_file, None, None, enabled=False)

    job_id = str(os.environ.get("SLURM_JOB_ID", os.getpid()))
    local_job_dir = Path(local_temp_dir) / f"scshift_job_{job_id}"

    log.info("=" * 80)
    log.info("DATA LOCALIZATION: Enabled. Using local directory: %s", local_job_dir)
    log.info("=" * 80)

    localized_configs: List[str] = []

    try:
        local_job_dir.mkdir(parents=True, exist_ok=True)

        for config_str in dataset_configs:
            parts = config_str.split(":")
            remote_path = Path(parts[0])
            local_path = local_job_dir / "data" / remote_path.name

            if not local_path.exists():
                log.info("Copying data from %s to %s...", remote_path, local_path)
                shutil.copytree(remote_path, local_path, dirs_exist_ok=True)
                log.info("Data copy complete.")
            else:
                log.info("Data already exists at %s. Skipping copy.", local_path)

            parts[0] = str(local_path)
            localized_configs.append(":".join(parts))

        localized_cache = None
        if cache_file:
            remote_cache_file = Path(cache_file)
            localized_cache = local_job_dir / "cache" / remote_cache_file.name
            localized_cache.parent.mkdir(exist_ok=True)
            log.info("Using localized metadata cache file: %s", localized_cache)
        return LocalizationContext(
            localized_configs,
            str(localized_cache) if cache_file else cache_file,
            local_job_dir,
            job_id,
            True,
        )
    except Exception as exc:
        log.error("Error during data localization: %s", exc)
        shutil.rmtree(local_job_dir, ignore_errors=True)
        raise


def build_scheduler_config(args: Dict[str, any]) -> Dict[str, any]:
    """Extract scheduler configuration from parsed CLI arguments."""
    scheduler_type = args.get("scheduler")
    if not scheduler_type:
        return {}

    config: Dict[str, any] = {"type": scheduler_type}
    if scheduler_type == "cosine":
        config.update(
            {
                "T_max": args.get("scheduler_T_max", 100),
                "warmup_epochs": args.get("scheduler_warmup_epochs", 0),
                "eta_min": args.get("scheduler_eta_min", 1e-6),
            }
        )
    elif scheduler_type == "cosine_restarts":
        config.update(
            {
                "T_0": args.get("scheduler_T_0", 10),
                "T_mult": args.get("scheduler_T_mult", 2),
                "eta_min": args.get("scheduler_eta_min", 1e-6),
            }
        )
    elif scheduler_type == "step":
        config.update(
            {
                "step_size": args.get("scheduler_step_size", 30),
                "gamma": args.get("scheduler_gamma", 0.1),
            }
        )
    elif scheduler_type == "reduce_on_plateau":
        config.update(
            {
                "patience": args.get("scheduler_patience", 10),
                "factor": args.get("scheduler_factor", 0.5),
                "eta_min": args.get("scheduler_eta_min", 1e-6),
            }
        )
    return config


def configure_scheduler(optimizer: torch.optim.Optimizer, scheduler_config: Dict[str, any]) -> Optional[Dict[str, any]]:
    """Create a PyTorch scheduler configuration dictionary for Lightning."""
    if not scheduler_config:
        return None

    scheduler_type = scheduler_config.get("type", "cosine")

    if scheduler_type == "cosine":
        warmup_epochs = scheduler_config.get("warmup_epochs", 0)
        T_max = scheduler_config.get("T_max", 100)
        eta_min = scheduler_config.get("eta_min", 1e-6)

        if warmup_epochs > 0:
            from torch.optim.lr_scheduler import LinearLR, SequentialLR

            warmup_scheduler = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_epochs)
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max - warmup_epochs, eta_min=eta_min)
            scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

        return {"scheduler": scheduler, "interval": "epoch", "frequency": 1}

    if scheduler_type == "cosine_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_config.get("T_0", 10),
            T_mult=scheduler_config.get("T_mult", 2),
            eta_min=scheduler_config.get("eta_min", 1e-6),
        )
        return {"scheduler": scheduler, "interval": "epoch", "frequency": 1}

    if scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get("step_size", 30),
            gamma=scheduler_config.get("gamma", 0.1),
        )
        return {"scheduler": scheduler, "interval": "epoch", "frequency": 1}

    if scheduler_type == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_config.get("factor", 0.5),
            patience=scheduler_config.get("patience", 10),
            verbose=True,
        )
        return {
            "scheduler": scheduler,
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }

    raise ValueError(f"Unknown scheduler type: {scheduler_type}")
