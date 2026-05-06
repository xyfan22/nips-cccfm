"""Regularization components for StateICL models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SlicedWassersteinDistance:
    """Callable helper that computes the sliced Wasserstein distance."""

    n_proj: int = 64

    def __call__(self, x: torch.Tensor, y: torch.Tensor, n_proj: Optional[int] = None) -> torch.Tensor:
        if x.shape != y.shape:
            raise ValueError("Input tensors must have identical shapes")

        num_proj = n_proj or self.n_proj
        batch_size, _, latent_dim = x.shape
        projections = torch.randn(num_proj, latent_dim, device=x.device)
        projections = projections / projections.norm(dim=1, keepdim=True)

        x_proj = x @ projections.t()
        y_proj = y @ projections.t()

        x_sorted, _ = torch.sort(x_proj, dim=1)
        y_sorted, _ = torch.sort(y_proj, dim=1)
        return ((x_sorted - y_sorted) ** 2).mean()
