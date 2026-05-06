"""Core architecture components for the StateICL model."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modules import SlicedWassersteinDistance, TabularAttentionLayer


class StateICLModelBase(nn.Module):
    """Shared architecture and forward pass for the StateICL model."""

    def __init__(
        self,
        n_genes: int,
        n_hidden: int = 100,
        token_dim: int = 8,
        n_cells: int = 128,
        n_layers: int = 6,
        n_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        mask_rate_min: float = 0.2,
        mask_rate_max: float = 0.8,
        sw_weight: float = 1.0,
        n_proj: int = 64,
    ) -> None:
        super().__init__()

        self.n_genes = n_genes
        self.n_hidden = n_hidden
        self.token_dim = token_dim
        self.n_cells = n_cells
        self.n_layers = n_layers
        self.mask_rate_min = mask_rate_min
        self.mask_rate_max = mask_rate_max
        self.sw_weight = sw_weight
        self.n_proj = n_proj
        self.sw_distance = SlicedWassersteinDistance(n_proj=n_proj)

        self.gene_reduction = nn.Sequential(
            nn.Linear(n_genes, n_hidden * token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.gene_pos_embedding = nn.Parameter(torch.randn(n_hidden, token_dim))

        self.layers = nn.ModuleList(
            [
                TabularAttentionLayer(
                    token_dim=token_dim,
                    n_cells=n_cells,
                    n_hidden=n_hidden,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(n_hidden * token_dim, n_hidden * token_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden * token_dim * 2, n_genes * 2),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.2)

    def _reduce_and_tokenize(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, n_cells, _ = features.shape
        reduced = self.gene_reduction(features)
        return reduced.reshape(batch_size, n_cells, self.n_hidden, self.token_dim)

    def _run_attention_layers(
        self,
        tokens: torch.Tensor,
        gene_attn_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Sequence[torch.Tensor]]]:
        attn_maps: List[torch.Tensor] = []
        x = tokens
        for layer in self.layers:
            x, attn = layer(x, self.gene_pos_embedding, gene_attn_mask, return_attn)
            if return_attn:
                attn_maps.append(attn)
        if return_attn:
            return x, attn_maps
        return x

    def _compute_nb_parameters(
        self,
        final_cell_embeddings: torch.Tensor,
        observed_lib_size: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, n_cells, _ = final_cell_embeddings.shape
        flat_embeddings = final_cell_embeddings.reshape(batch_size * n_cells, -1)
        output = self.output_mlp(flat_embeddings)
        output = output.reshape(batch_size, n_cells, self.n_genes, 2)

        px_scale_logits = output[..., 0]
        nb_dispersion = F.softplus(output[..., 1])
        px_scale = F.softmax(px_scale_logits, dim=-1)
        nb_mean = px_scale * observed_lib_size
        return nb_mean, nb_dispersion, px_scale

    def apply_mask(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, n_cells, n_genes = features.shape
        device = features.device

        mask_rate = torch.empty(1, device=device).uniform_(
            self.mask_rate_min, self.mask_rate_max
        ).item()
        n_genes_to_mask = int(n_genes * mask_rate)

        mask_indices = torch.randperm(n_genes, device=device)[:n_genes_to_mask]

        mask = torch.zeros(batch_size, n_cells, n_genes, dtype=torch.bool, device=device)
        mask[:, :, mask_indices] = True

        masked_features = features.clone()
        masked_features[mask] = 0.0
        return masked_features, mask

    def forward(
        self,
        features: torch.Tensor,
        return_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        batch_size, n_cells, _ = features.shape
        device = features.device

        original_features = features.clone()
        observed_lib_size = original_features.sum(dim=-1, keepdim=True)

        features = torch.log1p(features)

        masked_features, mask = self.apply_mask(features)

        tokens = self._reduce_and_tokenize(masked_features)
        x = self._run_attention_layers(tokens)
        final_cell_embeddings = x.reshape(batch_size, n_cells, -1)

        nb_mean, nb_dispersion, px_scale = self._compute_nb_parameters(
            final_cell_embeddings, observed_lib_size
        )

        result = {
            "nb_mean": nb_mean,
            "nb_dispersion": nb_dispersion,
            "px_scale": px_scale,
            "observed_lib_size": observed_lib_size,
            "mask": mask,
            "cell_embeddings": final_cell_embeddings,
            "masked_features": masked_features,
            "original_features": original_features,
        }

        if return_loss:
            recon_loss, _ = self._compute_reconstruction_loss(
                nb_mean, nb_dispersion, original_features, mask
            )
            sw_loss = self._compute_sw_loss(final_cell_embeddings)
            total_loss = recon_loss + self.sw_weight * sw_loss

            result.update(
                {
                    "loss": total_loss,
                    "recon_loss": recon_loss,
                    "sw_loss": sw_loss,
                }
            )

            if not self.training:
                metrics = self._compute_eval_metrics(nb_mean, original_features, mask)
                result.update(metrics)
            else:
                zero = torch.tensor(0.0, device=device, dtype=nb_mean.dtype)
                result.update(
                    {
                        "masked_mae": zero,
                        "masked_corr": zero,
                        "mask_rate": zero,
                    }
                )

        return result


__all__ = ["StateICLModelBase"]
