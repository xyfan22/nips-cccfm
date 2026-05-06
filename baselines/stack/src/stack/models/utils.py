"""Model-specific utility helpers for StateICL."""
from __future__ import annotations

from typing import Optional, Sequence, Union

import pickle

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import issparse

from ..modules import SlicedWassersteinDistance

__all__ = [
    "align_result_to_adata_numpy",
    "SlicedWassersteinDistance",
    "batch_sliced_wasserstein_1d",
    "safe_logit",
    "ReparamNBLogSampler",
    "ReparamNBLog_Sampler",
]

_SLICED_WASSERSTEIN = SlicedWassersteinDistance()


def align_result_to_adata_numpy(
    result: np.ndarray,        # shape (n_cells, n_model_genes)
    test_adata,                # AnnData
    genelist_path: str,
    gene_name_col: Optional[str] = None,
    prefer_raw: bool = True,
    fill_missing: float = 0.0,  # value for test-only genes
    cell_indices_to_keep: Optional[Union[Sequence[int], np.ndarray]] = None,  # keep by cells (rows)
) -> np.ndarray:
    """Align model outputs to match the ordering in ``test_adata``."""

    # --- sanity check on cell dimension
    if result.shape[0] != test_adata.n_obs:
        raise ValueError(
            "`result` n_cells (%d) must equal test_adata.n_obs (%d)." %
            (result.shape[0], test_adata.n_obs)
        )

    # 1) test gene names (prefer raw.var)
    var_df = test_adata.raw.var if (prefer_raw and (test_adata.raw is not None)) else test_adata.var
    if gene_name_col is not None and gene_name_col in var_df.columns:
        test_genes = var_df[gene_name_col].astype(str).str.upper().to_numpy()
    else:
        test_genes = var_df.index.astype(str).str.upper().to_numpy()

    # also get the source matrix to "preserve" from
    test_mat = test_adata.raw.X if (prefer_raw and (test_adata.raw is not None)) else test_adata.X

    # 2) model gene order from pickle
    with open(genelist_path, "rb") as f:
        model_genes_raw = pickle.load(f)
    model_genes = np.asarray([str(g).upper() for g in model_genes_raw])

    # 3) map model columns -> test columns (−1 if missing in test)
    pos_in_test = {g: i for i, g in enumerate(test_genes)}
    dest = np.fromiter((pos_in_test.get(g, -1) for g in model_genes),
                       dtype=np.int64, count=model_genes.size)
    present_mask = dest != -1  # model genes that exist in test
    src_idx = np.nonzero(present_mask)[0]     # columns in result to copy
    dst_idx = dest[present_mask]              # corresponding columns in aligned

    # 4) build keep mask on cell axis
    n_cells = result.shape[0]
    if cell_indices_to_keep is None:
        keep_cell_mask = np.zeros(n_cells, dtype=bool)
        keep_rows = np.empty(0, dtype=int)
    else:
        if isinstance(cell_indices_to_keep, np.ndarray) and cell_indices_to_keep.dtype == bool:
            if cell_indices_to_keep.size != n_cells:
                raise ValueError("Boolean cell_indices_to_keep must have length equal to n_cells.")
            keep_cell_mask = cell_indices_to_keep.astype(bool, copy=False)
            keep_rows = np.nonzero(keep_cell_mask)[0]
        else:
            keep_rows = np.asarray(cell_indices_to_keep, dtype=int)
            if keep_rows.size and (keep_rows.min() < 0 or keep_rows.max() >= n_cells):
                raise ValueError("cell_indices_to_keep has out-of-range indices.")
            keep_cell_mask = np.zeros(n_cells, dtype=bool)
            keep_cell_mask[keep_rows] = True

    not_keep_rows = np.nonzero(~keep_cell_mask)[0]

    # 5) allocate aligned output with fill_missing everywhere
    aligned = np.full((n_cells, test_genes.size), fill_missing, dtype=result.dtype)

    # 6) fill overlapping genes for rows NOT kept -> use model result
    if src_idx.size > 0 and not_keep_rows.size > 0:
        aligned[np.ix_(not_keep_rows, dst_idx)] = result[np.ix_(not_keep_rows, src_idx)]

    # 7) fill overlapping genes for rows kept -> preserve from test_adata
    if src_idx.size > 0 and keep_rows.size > 0:
        if issparse(test_mat):
            test_block = test_mat[keep_rows[:, None], dst_idx].toarray()
        else:
            # make sure we have a numpy array view
            test_block = np.asarray(test_mat)[np.ix_(keep_rows, dst_idx)]
        aligned[np.ix_(keep_rows, dst_idx)] = test_block.astype(aligned.dtype, copy=False)

    return aligned


def batch_sliced_wasserstein_1d(x: torch.Tensor, y: torch.Tensor, n_proj: int = 64) -> torch.Tensor:
    """Compatibility wrapper around :class:`SlicedWassersteinDistance`."""
    return _SLICED_WASSERSTEIN(x, y, n_proj=n_proj)


def safe_logit(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Numerically stable logit transform."""
    q = q.clamp(eps, 1 - eps)
    return torch.log(q) - torch.log1p(-q)


class ReparamNBLogSampler(nn.Module):
    """Reparameterizable sampler for ``log(1 + X / N)`` with ``X ~ NB(mu, theta)``."""

    def forward(
        self,
        mu: torch.Tensor,
        theta: torch.Tensor,
        N: torch.Tensor,
        tau: float = 0.7,
        eps: float = 1e-6,
        logistic_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if N.dim() == 2:
            N = N.unsqueeze(-1)  # (B, C, 1)
        B, C, G = mu.shape
        device, dtype = mu.device, mu.dtype

        # NB p0
        v = mu + mu**2 / theta.clamp_min(eps)
        log_p0 = theta.clamp_min(eps).log() - (theta + mu.clamp_min(eps)).log()
        p0 = torch.exp(theta.clamp_min(eps) * log_p0)  # (B, C, G)
        q = (1.0 - p0).clamp(eps, 1 - eps)

        # Moments for U = X + N
        EU = N + mu
        EX2 = v + mu**2
        EU2 = N**2 + 2 * N * mu + EX2

        # Conditional on X>0
        EU_pos = N + mu / q
        EU2_pos = N**2 + (2 * N * mu + mu**2 + v) / q

        ratio = (EU2_pos / (EU_pos.clamp_min(eps) ** 2)).clamp_min(1.0 + eps)
        sigma2_plus = torch.log(ratio)
        sigma_plus = torch.sqrt(sigma2_plus)
        m_plus = torch.log(EU_pos.clamp_min(eps)) - 0.5 * sigma2_plus

        # Reparam sample on log1p(X/N)
        logN = N.clamp_min(eps).log()
        eps_norm = torch.randn(B, C, G, device=device, dtype=dtype)
        Z_pos = (m_plus - logN) + sigma_plus * eps_norm

        z0 = torch.zeros(B, C, 1, device=device, dtype=dtype)  # broadcast

        # Concrete gate
        if logistic_noise is None:
            u = torch.rand(B, C, G, device=device, dtype=dtype).clamp(eps, 1 - eps)
            logistic_noise = torch.log(u) - torch.log1p(-u)
        logit_pre = safe_logit(q, eps) + logistic_noise
        S = torch.sigmoid(logit_pre / tau)

        Z = S * Z_pos + (1.0 - S) * z0
        return Z


# Backwards compatibility with the previous camel_case name
class ReparamNBLog_Sampler(ReparamNBLogSampler):
    pass
