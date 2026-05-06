"""Loss and evaluation utilities for the StateICL model."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from scvi.distributions import NegativeBinomial


class LossComputationMixin:
    """Provides loss and metric helpers for ``StateICLModel``."""

    def _compute_reconstruction_loss(
        self,
        nb_mean: torch.Tensor,
        nb_dispersion: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        nb_dist = NegativeBinomial(mu=nb_mean, theta=nb_dispersion)
        recon_loss_all = -nb_dist.log_prob(targets)
        mask_f = mask.float()
        masked_count = mask_f.sum()

        if masked_count > 0:
            recon_loss = (recon_loss_all * mask_f).sum() / masked_count
        else:
            recon_loss = torch.tensor(0.0, device=targets.device, dtype=recon_loss_all.dtype)

        return recon_loss, recon_loss_all

    def _compute_sw_loss(
        self,
        final_cell_embeddings: torch.Tensor,
        *,
        subsample_size: Optional[int] = None,
        min_size: int = 32,
        max_size: int = 128,
        n_proj: Optional[int] = None,
    ) -> torch.Tensor:
        n_cells = final_cell_embeddings.shape[1]
        device = final_cell_embeddings.device

        if subsample_size is None:
            upper = min(max_size, n_cells)
            lower = min(min_size, max(upper, 1))
            if lower == 0:
                return torch.tensor(0.0, device=device, dtype=final_cell_embeddings.dtype)
            if lower == upper:
                k = lower
            else:
                k = torch.randint(lower, upper + 1, (), device=device).item()
        else:
            k = max(1, min(subsample_size, n_cells))

        idx = torch.randperm(n_cells, device=device)[:k]
        final_cell_embeddings_subsampled = final_cell_embeddings[:, idx]
        prior_samples = torch.randn_like(final_cell_embeddings_subsampled)
        centered_embeddings = final_cell_embeddings_subsampled - final_cell_embeddings_subsampled.mean(
            dim=1, keepdim=True
        )

        if n_proj is None:
            return self.sw_distance(centered_embeddings, prior_samples)
        return self.sw_distance(centered_embeddings, prior_samples, n_proj=n_proj)

    def _compute_eval_metrics(
        self,
        nb_mean: torch.Tensor,
        original_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        eps = 1e-8
        mask_f = mask.float()

        diff = (nb_mean - original_features).abs()
        masked_mae = (diff * mask_f).sum() / mask_f.sum()

        masked_count = mask_f.sum(dim=-1, keepdim=True).clamp_min(1.0)

        pred_masked = nb_mean * mask_f
        true_masked = original_features * mask_f

        pred_mean = pred_masked.sum(dim=-1, keepdim=True) / masked_count
        true_mean = true_masked.sum(dim=-1, keepdim=True) / masked_count

        pred_centered = (pred_masked - pred_mean) * mask_f
        true_centered = (true_masked - true_mean) * mask_f

        cov = (pred_centered * true_centered).sum(dim=-1)
        pred_var = (pred_centered**2).sum(dim=-1) + eps
        true_var = (true_centered**2).sum(dim=-1) + eps

        corr_cell = cov / (pred_var.sqrt() * true_var.sqrt())
        masked_corr = corr_cell[mask_f.sum(dim=-1) > 1].mean()

        mask_rate = mask_f.mean()

        return {
            "masked_mae": masked_mae,
            "masked_corr": masked_corr,
            "mask_rate": mask_rate,
        }


__all__ = ["LossComputationMixin"]
