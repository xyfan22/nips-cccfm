"""Fine-tuning model built on top of the core StateICL architecture."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from ..core import StateICLModel
from .mixins import FinetuneInitializationMixin, FinetuneLossMixin


class ICL_FinetunedModel(
    FinetuneInitializationMixin,
    FinetuneLossMixin,
    StateICLModel,
):
    """Fine-tuning head on top of :class:`StateICLModel`."""

    def forward(
        self,
        observed_features: torch.Tensor,
        ground_truth_features: torch.Tensor,
        cell_type_ids: Optional[torch.Tensor] = None,
        position_mask: Optional[torch.Tensor] = None,
        t_cell_embeddings: Optional[torch.Tensor] = None,
        n_kept_cell: int = 0,
        mask_genes: bool = True,
        return_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        result: Dict[str, torch.Tensor] = {}

        batch_size, n_cells, n_genes = observed_features.shape

        features_log = torch.log1p(observed_features)
        gtfeatures_log = torch.log1p(ground_truth_features)
        time = torch.rand(1, device=observed_features.device).item() * 0.9375
        n_context_cell = n_kept_cell + int((n_cells - n_kept_cell) * time)
        features_log[:, n_kept_cell:n_context_cell] = gtfeatures_log[:, n_kept_cell:n_context_cell]
        mask = torch.zeros(
            batch_size,
            n_cells,
            n_genes,
            dtype=torch.bool,
            device=observed_features.device,
        )
        if mask_genes:
            masked_features, mask = self.apply_finetune_mask(features_log)
            tokens = self._reduce_and_tokenize(masked_features)
        else:
            tokens = self._reduce_and_tokenize(features_log)

        x = tokens

        mask_expanded = torch.zeros_like(x)
        if n_kept_cell < n_cells:
            mask_expanded[:, n_context_cell:, :, :] = 1.0
        query_emb = self.query_pos_embedding.unsqueeze(0).unsqueeze(0)
        x = x + query_emb * mask_expanded

        attn_mask = torch.zeros(
            n_cells,
            n_cells,
            dtype=torch.bool,
            device=observed_features.device,
        )
        if n_kept_cell < n_cells:
            attn_mask[:n_kept_cell, n_kept_cell:] = True
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

        x = self._run_attention_layers(x, gene_attn_mask=attn_mask)

        final_cell_embeddings = x.reshape(batch_size, n_cells, -1)
        lib_size = ground_truth_features.sum(dim=-1, keepdim=True).clamp(min=1.0)
        nb_mean, nb_dispersion, px_scale = self._compute_nb_parameters(
            final_cell_embeddings,
            lib_size,
        )

        result.update(
            {
                "px_scale": px_scale,
                "nb_mean": nb_mean,
                "nb_dispersion": nb_dispersion,
                "final_cell_embeddings": final_cell_embeddings,
            }
        )

        if not return_loss:
            return result

        mean_kept_cell = final_cell_embeddings[:, :n_kept_cell].mean(dim=1).detach()
        mid_feat = final_cell_embeddings[:, n_kept_cell:n_context_cell, :].detach()
        tail_feat = final_cell_embeddings[:, n_context_cell:, :].detach()
        cls_loss, cls_acc = self._compute_cls_loss(mean_kept_cell, mid_feat, tail_feat)

        if n_context_cell < n_cells and mask[:, :n_context_cell, :].float().sum() > 0:
            recon_loss, _ = self._compute_reconstruction_loss(
                nb_mean[:, :n_context_cell, :],
                nb_dispersion[:, :n_context_cell, :],
                ground_truth_features[:, :n_context_cell, :],
                mask[:, :n_context_cell, :],
            )
        else:
            recon_loss = torch.tensor(0.0, device=observed_features.device)

        mmd_loss, pred_dist_all, true_dist_all = self._compute_mmd_loss(
            nb_mean,
            nb_dispersion,
            ground_truth_features,
            lib_size,
            final_cell_embeddings,
            t_cell_embeddings,
            position_mask,
            cell_type_ids,
            n_context_cell,
            n_cells,
            n_genes,
            time,
            observed_features.device,
        )
        n_proj = getattr(self, "n_proj", None)
        subsample_size = max(32, min(128, n_cells))
        sw_loss = self._compute_sw_loss(
            final_cell_embeddings,
            subsample_size=subsample_size,
            min_size=subsample_size,
            max_size=subsample_size,
            n_proj=n_proj,
        )

        total_loss = recon_loss + mmd_loss + self.sw_weight * sw_loss + cls_loss
        result.update(
            {
                "loss": total_loss,
                "recon_loss": recon_loss,
                "mmd_loss": mmd_loss,
                "sw_loss": sw_loss,
                "cls_loss": cls_loss,
                "cls_acc": cls_acc,
            }
        )

        if not self.training:
            metrics = self._compute_eval_metrics(
                nb_mean[:, :n_context_cell, :],
                ground_truth_features[:, :n_context_cell, :],
                mask[:, :n_context_cell, :],
            )

            if pred_dist_all is not None and true_dist_all is not None:
                sw_predict = self.sw_distance(pred_dist_all, true_dist_all, n_proj=256)
            else:
                sw_predict = torch.tensor(0.0, device=observed_features.device)

            result.update({**metrics, "sw_predict": sw_predict})
        return result
