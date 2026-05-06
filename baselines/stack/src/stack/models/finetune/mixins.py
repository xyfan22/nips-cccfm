"""Reusable mixins for the fine-tuning model."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from geomloss import SamplesLoss

from ..utils import ReparamNBLogSampler
from ...modules import SlicedWassersteinDistance


class FinetuneInitializationMixin:
    """Provide fine-tuning specific modules and parameters."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.mmd_loss = SamplesLoss(loss="energy")

        self.query_pos_embedding = nn.Parameter(
            torch.empty(self.n_hidden, self.token_dim)
        )
        nn.init.normal_(self.query_pos_embedding, mean=0.0, std=0.02)

        def scale_pos_grad_hook(grad: torch.Tensor) -> torch.Tensor:
            return grad * 10.0

        self.query_pos_embedding.register_hook(scale_pos_grad_hook)

        self.nblog_sampler = ReparamNBLogSampler()
        self.cls = nn.Sequential(
            nn.Linear(self.n_hidden * self.token_dim * 2, self.n_hidden),
            nn.GELU(),
            nn.Linear(self.n_hidden, 1),
        )
        for parameter in self.cls.parameters():
            parameter.register_hook(scale_pos_grad_hook)

        self.sw_distance = SlicedWassersteinDistance(n_proj=self.n_proj)


class FinetuneLossMixin:
    """Mixin containing masking utilities and fine-tuning losses."""

    def apply_finetune_mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rectangular masking to an input tensor."""

        batch_size, n_cells, n_genes = x.shape
        device = x.device

        mask_rate = torch.empty(1, device=device).uniform_(0.1, 0.3).item()
        n_genes_to_mask = int(n_genes * mask_rate)
        mask_indices = torch.randperm(n_genes, device=device)[:n_genes_to_mask]

        mask = torch.zeros(batch_size, n_cells, n_genes, dtype=torch.bool, device=device)
        mask[:, :, mask_indices] = True

        masked_x = x.clone()
        masked_x[mask] = 0.0
        return masked_x, mask

    def _compute_cls_loss(
        self,
        mean_kept_cell: torch.Tensor,
        mid_feat: torch.Tensor,
        tail_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute classification loss/accuracy for mid & tail cells."""

        ctx_mid = mean_kept_cell.unsqueeze(1).expand(-1, mid_feat.size(1), -1)
        ctx_tail = mean_kept_cell.unsqueeze(1).expand(-1, tail_feat.size(1), -1)

        inp_mid = torch.cat([ctx_mid, mid_feat], dim=-1)
        inp_tail = torch.cat([ctx_tail, tail_feat], dim=-1)

        logit_mid = self.cls(inp_mid).squeeze(-1)
        logit_tail = self.cls(inp_tail).squeeze(-1)

        logits = torch.cat([logit_mid, logit_tail], dim=1)
        labels = torch.cat(
            [torch.zeros_like(logit_mid), torch.ones_like(logit_tail)],
            dim=1,
        )

        logits_flat = logits.reshape(-1)
        labels_flat = labels.reshape(-1)

        with torch.no_grad():
            pos = (labels_flat == 1).float().sum().clamp(min=1.0)
            neg = (labels_flat == 0).float().sum().clamp(min=1.0)
            pos_weight = neg / pos

        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        cls_loss = bce(logits_flat, labels_flat)
        cls_acc = ((logits_flat > 0).float() == labels_flat).float().mean()
        return cls_loss, cls_acc

    def _compute_mmd_loss(
        self,
        nb_mean: torch.Tensor,
        nb_dispersion: torch.Tensor,
        ground_truth_features: torch.Tensor,
        lib_size: torch.Tensor,
        final_cell_embeddings: torch.Tensor,
        t_cell_embeddings: Optional[torch.Tensor],
        position_mask: Optional[torch.Tensor],
        cell_type_ids: Optional[torch.Tensor],
        n_context_cell: int,
        n_cells: int,
        n_genes: int,
        time: float,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute MMD loss on replaced cells."""

        if n_context_cell >= n_cells:
            zero = torch.tensor(0.0, device=device)
            return zero, None, None

        mask_indices = torch.ones(n_genes, dtype=torch.bool, device=device)
        rep_lib_size = lib_size[:, n_context_cell:, :]

        if mask_indices.sum().item() > 1000:
            theta = 100
            expected_counts = (
                ground_truth_features[:, n_context_cell:, :].sum(dim=1, keepdim=True)
                * ground_truth_features[:, n_context_cell:, :].sum(dim=2, keepdim=True)
                / rep_lib_size.sum(dim=1, keepdim=True)
            )
            variance = (expected_counts + (expected_counts**2) / theta).clamp_min(1e-4)
            residuals = (
                ground_truth_features[:, n_context_cell:, :]
                - expected_counts
            ) / torch.sqrt(variance)
            residuals_var = residuals.var(dim=1).mean(dim=0)

            mask_indices &= torch.zeros_like(mask_indices).index_fill_(
                0,
                torch.topk(residuals_var, min(1000, residuals_var.numel())).indices,
                True,
            )

        pred_mean = nb_mean[:, n_context_cell:, :]
        pred_dispersion = (
            nb_dispersion[:, :n_context_cell, :].median(dim=1, keepdim=True).values.detach()
        )

        pred_dist_all = self.nblog_sampler(
            mu=pred_mean[:, :, mask_indices],
            theta=pred_dispersion[:, :, mask_indices],
            N=rep_lib_size / 1e4,
        )
        true_dist_all = torch.log1p(
            1e4 * ground_truth_features[:, n_context_cell:, mask_indices] / rep_lib_size
        )

        pred_embed_all = final_cell_embeddings[:, n_context_cell:, :]
        true_embed_all = None if t_cell_embeddings is None else t_cell_embeddings[:, n_context_cell:, :]

        if position_mask is not None:
            sample_losses = []
            sample_embed_losses = []
            if cell_type_ids is None:
                cell_type_ids = torch.ones_like(position_mask, dtype=torch.long)
            for i in range(nb_mean.shape[0]):
                type_ids = cell_type_ids[i]
                pos_mask = position_mask[i]
                type_ids = type_ids[n_context_cell:]
                pos_mask = pos_mask[n_context_cell:]

                pred_i = pred_dist_all[i]
                true_i = true_dist_all[i]

                pred_embed_i = pred_embed_all[i]
                true_embed_i = (
                    pred_embed_all[i] if true_embed_all is None else true_embed_all[i]
                )

                type_losses = []
                type_embed_losses = []
                for t in torch.unique(type_ids):
                    tm = (type_ids == t) & pos_mask
                    if tm.sum() > 1:
                        m_pred = pred_i[tm].unsqueeze(0)
                        m_true = true_i[tm].unsqueeze(0)
                        type_losses.append(self.mmd_loss(m_pred, m_true))

                        m_embed_pred = pred_embed_i[tm].unsqueeze(0)
                        m_embed_true = true_embed_i[tm].unsqueeze(0)
                        type_embed_losses.append(
                            self.mmd_loss(m_embed_pred, m_embed_true)
                        )

                if len(type_losses) > 0:
                    sample_losses.append(torch.stack(type_losses).mean())
                    sample_embed_losses.append(torch.stack(type_embed_losses).mean())

            mmd_loss = (
                torch.stack(sample_losses).mean()
                if len(sample_losses) > 0
                else torch.tensor(0.0, device=device)
            )
            mmd_embed_loss = (
                torch.stack(sample_embed_losses).mean()
                if len(sample_embed_losses) > 0
                else torch.tensor(0.0, device=device)
            )
        else:
            mmd_loss = self.mmd_loss(pred_dist_all, true_dist_all).mean()
            if true_embed_all is None:
                true_embed_all = pred_embed_all
            mmd_embed_loss = self.mmd_loss(pred_embed_all, true_embed_all).mean()

        mmd_loss = (0.5 * mmd_loss + 0.5 * mmd_embed_loss) / (1 - time)
        return mmd_loss, pred_dist_all, true_dist_all
