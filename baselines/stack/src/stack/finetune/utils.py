"""Utility helpers for the StateICL fine-tuning CLI."""
from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional

import torch
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from ..data.finetuning.datasets import DatasetConfig, load_gene_list

log = logging.getLogger(__name__)


def parse_dataset_configs(config_strings: Iterable[str]) -> List[DatasetConfig]:
    """Parse CLI dataset configuration strings into :class:`DatasetConfig` objects."""
    configs: List[DatasetConfig] = []

    for config_str in config_strings:
        parts = config_str.split(":")
        if len(parts) < 4:
            raise ValueError(f"Invalid config string, not enough parts: {config_str}")

        dataset_type = parts[0]
        path = parts[1]
        filter_organism = True
        gene_name_col: Optional[str] = None

        if dataset_type == "human":
            donor_col = parts[2]
            cell_type_col = parts[3]
            if len(parts) > 4 and parts[4]:
                filter_organism = parts[4].lower() == "true"
            if len(parts) > 5 and parts[5]:
                gene_name_col = parts[5]
            config = DatasetConfig(
                path=path,
                type="human",
                donor_col=donor_col,
                cell_type_col=cell_type_col,
                filter_organism=filter_organism,
                gene_name_col=gene_name_col,
            )
        elif dataset_type == "drug":
            if len(parts) < 5:
                raise ValueError(f"Drug dataset config requires at least 5 parts: {config_str}")
            condition_col = parts[2]
            cell_line_col = parts[3]
            control_condition = parts[4]
            if len(parts) > 5 and parts[5]:
                filter_organism = parts[5].lower() == "true"
            if len(parts) > 6 and parts[6]:
                gene_name_col = parts[6]
            config = DatasetConfig(
                path=path,
                type="drug",
                condition_col=condition_col,
                cell_line_col=cell_line_col,
                control_condition=control_condition,
                filter_organism=filter_organism,
                gene_name_col=gene_name_col,
            )
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        configs.append(config)
        log.info("Added %s dataset config: path=%s gene_col=%s", dataset_type, path, gene_name_col)

    return configs


def build_scheduler_config(args: Dict[str, any]) -> Optional[Dict[str, any]]:
    """Extract scheduler configuration from parsed CLI arguments."""
    scheduler_type = args.get("scheduler")
    if not scheduler_type:
        return None

    config: Dict[str, any] = {"type": scheduler_type}
    if scheduler_type == "cosine":
        config.update(
            {
                "T_max": args.get("scheduler_T_max", 20),
                "warmup_epochs": args.get("scheduler_warmup_epochs", 0),
                "eta_min": args.get("scheduler_eta_min", 1e-6),
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
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    return config


def configure_logger(logger_type: str, project_name: str, run_name: Optional[str], save_dir: str):
    """Return a configured W&B or TensorBoard logger based on CLI arguments."""
    if logger_type == "wandb":
        return WandbLogger(project=project_name, name=run_name, save_dir=save_dir)
    return TensorBoardLogger(save_dir=save_dir, name=project_name, version=run_name)


def override_model_config_n_cells(ckpt_path: str, new_n_cells: int) -> Dict[str, any]:
    """Load a checkpoint and override the stored ``n_cells`` hyperparameter."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hyper_params = ckpt.get("hyper_parameters", ckpt.get("hparams", {}))
    model_config = dict(hyper_params.get("model_config", {}))
    old_value = model_config.get("n_cells")
    model_config["n_cells"] = int(new_n_cells)
    log.info("Override model_config.n_cells: %s -> %s", old_value, model_config["n_cells"])
    return model_config


def build_model_config(args: Dict[str, any], n_genes: int) -> Dict[str, any]:
    """Construct a model configuration dictionary for fresh fine-tuning runs."""
    return {
        "n_cells": args["sample_size"],
        "n_genes": n_genes,
        "n_hidden": args["n_hidden"],
        "token_dim": args["token_dim"],
        "n_layers": args["n_layers"],
        "n_heads": args["n_heads"],
        "dropout": args["dropout"],
        "mask_rate_min": args["mask_rate_min"],
        "mask_rate_max": args["mask_rate_max"],
        "sw_weight": args["sw_weight"],
    }


__all__ = [
    "build_model_config",
    "build_scheduler_config",
    "configure_logger",
    "override_model_config_n_cells",
    "parse_dataset_configs",
    "load_gene_list",
]
