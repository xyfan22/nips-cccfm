"""Low-rank basis for R3-LowRank-MP CFM (advisor 2026-04-27).

The hypothesis: useful residual cell-state signal lives in a low-rank centered
subspace. Per-gene Moment degradation in R3.1 came from the flow's freedom to
inject arbitrary high-frequency gene-level distortion. Restricting the flow to
a rank-D PCA basis (D << G) should preserve W_2 gain while preventing Moment
degradation.

`LowRankBasis.fit(adata, source_lines, target_line=None, mode='both', rank=64)`:
  - 'control'   — PCA on centered source-line ctrl cells (per-line ctrl-mean removed)
  - 'perturbed' — PCA on centered source-line pert cells (per-(line,pert) anchor-profile removed)
  - 'both'      — concat the above (default; covers both endpoints' variability)

After fit, exposes:
  - `B` (np.ndarray [G, rank]) — orthonormal columns, np.float32
  - `project(X)`   — `(X - profile) @ B`  with optional per-row profile
  - `unproject(Z)` — `Z @ B.T`
  - `save(path)`   — npz: B + metadata
  - `LowRankBasis.load(path)` — class method
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np


class LowRankBasis:
    def __init__(self):
        self.B: Optional[np.ndarray] = None      # [G, rank]
        self.rank: Optional[int] = None
        self.mode: Optional[str] = None
        self.gene_dim: Optional[int] = None
        self.explained_variance_ratio: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #
    def fit(
        self,
        adata,
        source_lines: Sequence[str],
        target_line: Optional[str] = None,
        anchor_delta_dict: Optional[dict] = None,
        line_ctrl_means: Optional[dict] = None,
        mode: str = "both",
        rank: int = 64,
        max_cells_per_group: int = 2000,
        min_cells_per_group: int = 50,
        seed: int = 42,
    ) -> "LowRankBasis":
        """Fit a rank-D PCA basis on centered residuals from source-line cells.

        Centering rules:
          - control cells: subtract per-line ctrl mean (per-line because
            different lines have different basal expression).
          - perturbed cells: subtract per-(line, pert) anchor profile
            = line_ctrl_means[line] + anchor_delta_dict[pert].

        target_line cells are EXCLUDED to avoid leakage.
        """
        assert mode in ("control", "perturbed", "both"), f"unknown mode {mode}"
        if mode in ("perturbed", "both"):
            assert anchor_delta_dict is not None and line_ctrl_means is not None, (
                f"mode='{mode}' requires anchor_delta_dict + line_ctrl_means"
            )

        from sklearn.decomposition import PCA

        obs = adata.obs
        rng = np.random.default_rng(seed)

        # Leakage check
        if target_line is not None:
            target_overlap = (obs["cell_line"].isin(list(source_lines)).values
                              & (obs["cell_line"].values == target_line))
            assert not target_overlap.any(), (
                f"LowRankBasis.fit: target_line '{target_line}' is among source_lines"
            )

        residual_chunks: list[np.ndarray] = []

        if mode in ("control", "both"):
            # Per-line centered control residuals
            for cl in source_lines:
                line_ctrl_mask = (obs["cell_line"].values == cl) & (obs["pert"].values == "control")
                idxs = np.flatnonzero(line_ctrl_mask)
                if len(idxs) < min_cells_per_group:
                    continue
                if len(idxs) > max_cells_per_group:
                    idxs = rng.choice(idxs, size=max_cells_per_group, replace=False)
                X = adata[idxs].X
                if hasattr(X, "toarray"):
                    X = X.toarray()
                X = np.asarray(X, dtype=np.float64)
                X_centered = X - X.mean(0, keepdims=True)
                residual_chunks.append(X_centered)

        if mode in ("perturbed", "both"):
            # Per-(line, pert) centered residuals = x_pert - anchor_profile(line, pert)
            # Note: x_pert.mean(0) ≈ source_pert_mean ≈ source_line_ctrl_mean + delta
            # so subtracting anchor_profile is approximately the same as subtracting the
            # group's own mean, but uses the predicted anchor (consistent with training).
            for cl in source_lines:
                line_mask = (obs["cell_line"].values == cl)
                ctrl_mean = line_ctrl_means[cl]
                line_perts = sorted(set(obs[line_mask]["pert"].astype(str).unique()) - {"control"})
                for pert in line_perts:
                    if pert not in anchor_delta_dict:
                        continue
                    pmask = line_mask & (obs["pert"].values == pert)
                    idxs = np.flatnonzero(pmask)
                    if len(idxs) < min_cells_per_group:
                        continue
                    if len(idxs) > max_cells_per_group:
                        idxs = rng.choice(idxs, size=max_cells_per_group, replace=False)
                    X = adata[idxs].X
                    if hasattr(X, "toarray"):
                        X = X.toarray()
                    X = np.asarray(X, dtype=np.float64)
                    anchor_profile = ctrl_mean + anchor_delta_dict[pert]
                    X_centered = X - anchor_profile[None, :]
                    residual_chunks.append(X_centered)

        if not residual_chunks:
            raise RuntimeError(f"LowRankBasis.fit: no residuals for mode='{mode}'")

        X_all = np.concatenate(residual_chunks, axis=0).astype(np.float32)
        del residual_chunks
        print(f"[LowRankBasis] fitting PCA(rank={rank}) on {X_all.shape[0]} centered "
              f"cells × {X_all.shape[1]} genes (mode={mode}, sources={list(source_lines)}) ...")
        pca = PCA(n_components=rank, copy=False, random_state=seed)
        pca.fit(X_all)
        self.B = pca.components_.T.astype(np.float32)              # [G, rank]
        self.explained_variance_ratio = pca.explained_variance_ratio_.astype(np.float32)
        self.rank = int(rank)
        self.mode = mode
        self.gene_dim = X_all.shape[1]
        cum = float(self.explained_variance_ratio.sum())
        print(f"[LowRankBasis] fit done: rank={rank}  cum_var_explained={cum:.3f}  "
              f"top1={float(self.explained_variance_ratio[0]):.3f}")
        return self

    # ------------------------------------------------------------------ #
    # Project / unproject
    # ------------------------------------------------------------------ #
    def project(self, X: np.ndarray, profile: Optional[np.ndarray] = None) -> np.ndarray:
        """X [N, G]  →  Z [N, rank]. Optionally subtract per-row profile first."""
        assert self.B is not None
        X = np.asarray(X, dtype=np.float32)
        if profile is not None:
            profile = np.asarray(profile, dtype=np.float32)
            if profile.ndim == 1:
                X = X - profile[None, :]
            else:
                X = X - profile
        return X @ self.B

    def unproject(self, Z: np.ndarray) -> np.ndarray:
        """Z [N, rank]  →  X [N, G]. Output is in the basis's centered subspace."""
        assert self.B is not None
        Z = np.asarray(Z, dtype=np.float32)
        return Z @ self.B.T

    # ------------------------------------------------------------------ #
    # Persist
    # ------------------------------------------------------------------ #
    def save(self, path: Path) -> None:
        assert self.B is not None
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            Path(path),
            B=self.B,
            explained_variance_ratio=self.explained_variance_ratio,
            rank=np.array(self.rank),
            mode=np.array(self.mode),
            gene_dim=np.array(self.gene_dim),
        )

    @classmethod
    def load(cls, path: Path) -> "LowRankBasis":
        npz = np.load(Path(path), allow_pickle=True)
        obj = cls()
        obj.B = npz["B"].astype(np.float32)
        obj.explained_variance_ratio = npz["explained_variance_ratio"].astype(np.float32)
        obj.rank = int(npz["rank"])
        obj.mode = str(npz["mode"])
        obj.gene_dim = int(npz["gene_dim"])
        return obj
