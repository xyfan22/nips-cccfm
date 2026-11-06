"""Anchor providers for the residual ccCFM (Phase R1).

Three concrete anchors share an abstract `AnchorDeltaProvider` interface:

    LinearAnchor  — pooled source-line per-pert delta (Ahlmann-Eltze 2025 form).
                    delta(g) = mean(X | source ∧ pert==g) − mean(X | source ∧ ctrl).

    ExactAnchor   — per-line delta then averaged across source lines.
                    delta(g) = mean_cl[ mean(X | line==cl ∧ pert==g) − mean(X | line==cl ∧ ctrl) ]
                    Reference / upper-bound row only.

    RidgeAnchor   — wraps PooledGeneEmbeddingRidgeAnchor with the new interface.
                    delta(g) = ridge( primary_gene_embedding[g] ).

All three:
  - fit(adata, source_lines, target_line=None, exclude_perts=None) — assert no target_line cells.
  - delta(gene_symbol) -> [G] or None if unsupported.
  - delta_batch(perts) -> [B, G] (zero-fills + warns on missing).

R1.2 helper `apply_anchor(x_control, context_ctrl_mean, delta_g)` builds the
mean-centred residual source state used by both training and evaluation.

The original `PooledGeneEmbeddingRidgeAnchor` is preserved below for the
existing reference baseline scripts (`scripts/11_pooled_ridge_anchor.py`,
`scripts/12_exact_gene_retrieval.py`); new code should use the wrappers above.
"""

from __future__ import annotations

import abc
import warnings
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np


# ============================================================================ #
# R1.2 helper — mean-centred anchored source state
# ============================================================================ #

def apply_anchor(
    x_control: np.ndarray,
    context_ctrl_mean: np.ndarray,
    delta_g: np.ndarray,
) -> np.ndarray:
    """Build `x0_anchor` cells per R1.2 formula.

    x_control:         [N, G]  sampled control cells.
                                During training: source-line ctrl cells.
                                During eval:     target-line ctrl cells.
    context_ctrl_mean: [G]     global ctrl mean of the **same line** that
                                x_control was drawn from. Hardcoding this to
                                target_ctrl_mean during training would bake
                                the source→target shift into the residual loss
                                (advisor 2026-04-26).
    delta_g:           [G]     anchor's predicted delta for this pert.

    Returns [N, G] satisfying  `mean(out, 0) == context_ctrl_mean + delta_g` exactly.
    """
    x_control = np.asarray(x_control, dtype=np.float32)
    ctx_mean = np.asarray(context_ctrl_mean, dtype=np.float32)
    delta_g = np.asarray(delta_g, dtype=np.float32)
    return (x_control - x_control.mean(axis=0, keepdims=True)) + ctx_mean + delta_g


def apply_anchor_torch(x_control, context_ctrl_mean, delta_g):
    """Torch equivalent of apply_anchor — used in the training loop to avoid
    numpy roundtrips on every batch.

    All three inputs must be torch tensors on the SAME device. Returns a torch
    tensor with the same dtype as x_control.

    Mean invariant: `out.mean(dim=0) == context_ctrl_mean + delta_g` (exact, up
    to float precision; matches the numpy version within ~1e-6).

    Tested in `code/tests/test_smoke.py::test_apply_anchor_torch_matches_numpy`.
    """
    import torch
    if not isinstance(x_control, torch.Tensor):
        raise TypeError("apply_anchor_torch expects torch.Tensor inputs; "
                        "use apply_anchor for numpy.")
    return (
        x_control - x_control.mean(dim=0, keepdim=True)
    ) + context_ctrl_mean + delta_g


# ============================================================================ #
# Abstract interface
# ============================================================================ #

class AnchorDeltaProvider(abc.ABC):
    """Maps `gene_symbol` -> predicted source-line delta `[G]`.

    All concrete anchors must:
      - reject target_line cells at fit time;
      - return None for unsupported perts (don't silently invent zeros);
      - expose `gene_dim` and `supported_perts` so callers can pre-filter.
    """

    @abc.abstractmethod
    def fit(
        self,
        adata,
        source_lines: Sequence[str],
        target_line: Optional[str] = None,
        exclude_perts: Optional[Iterable[str]] = None,
        min_cells: int = 50,
    ) -> "AnchorDeltaProvider":
        ...

    @abc.abstractmethod
    def delta(self, gene_symbol: str) -> Optional[np.ndarray]:
        ...

    @property
    @abc.abstractmethod
    def gene_dim(self) -> int: ...

    @property
    @abc.abstractmethod
    def supported_perts(self) -> set: ...

    def delta_batch(
        self,
        gene_symbols: Sequence[str],
        zero_fill_missing: bool = True,
    ) -> np.ndarray:
        """Vectorised lookup; warns on missing.

        zero_fill_missing=True : missing perts get all-zero delta + warning (default).
        zero_fill_missing=False: raises KeyError on first missing pert.
        """
        out = []
        missing = []
        for g in gene_symbols:
            d = self.delta(g)
            if d is None:
                missing.append(g)
                if zero_fill_missing:
                    out.append(np.zeros(self.gene_dim, dtype=np.float32))
                else:
                    raise KeyError(f"{type(self).__name__} has no delta for pert '{g}'")
            else:
                out.append(d)
        if missing:
            warnings.warn(
                f"{type(self).__name__}.delta_batch: {len(missing)}/{len(gene_symbols)} "
                f"perts missing, zero-filled. Examples: {missing[:5]}",
                RuntimeWarning, stacklevel=2,
            )
        return np.stack(out)


# ============================================================================ #
# LinearAnchor
# ============================================================================ #

class LinearAnchor(AnchorDeltaProvider):
    """Ahlmann-Eltze 2025 formulation.

        delta(g) = mean(X | source ∧ pert==g) − mean(X | source ∧ ctrl)

    Pools cells across source lines BEFORE taking the mean.
    """

    def __init__(self):
        self._delta_cache: dict[str, np.ndarray] = {}
        self._gene_dim: Optional[int] = None
        self._source_ctrl_mean: Optional[np.ndarray] = None

    def fit(self, adata, source_lines, target_line=None, exclude_perts=None, min_cells=50):
        obs = adata.obs
        src_mask = obs["cell_line"].isin(list(source_lines)).values
        # R1.4 leakage assertion
        if target_line is not None:
            n_tgt_in_src = int((src_mask & (obs["cell_line"].values == target_line)).sum())
            assert n_tgt_in_src == 0, (
                f"LinearAnchor.fit: source_lines={list(source_lines)} overlaps with "
                f"target_line='{target_line}' on {n_tgt_in_src} cells. Refusing to fit."
            )
        exclude = set(exclude_perts or [])

        def _pb(mask):
            X = adata[mask].X
            if hasattr(X, "toarray"):
                X = X.toarray()
            return np.asarray(X, dtype=np.float64).mean(axis=0)

        src_ctrl_mask = src_mask & (obs["pert"].values == "control")
        self._source_ctrl_mean = _pb(src_ctrl_mask).astype(np.float32)
        self._gene_dim = self._source_ctrl_mean.shape[0]

        for pert in sorted(set(obs["pert"].astype(str).unique()) - {"control"}):
            if pert in exclude:
                continue
            pmask = src_mask & (obs["pert"].values == pert)
            if int(pmask.sum()) < min_cells:
                continue
            self._delta_cache[pert] = (_pb(pmask) - self._source_ctrl_mean).astype(np.float32)
        return self

    def delta(self, gene_symbol: str) -> Optional[np.ndarray]:
        return self._delta_cache.get(gene_symbol)

    @property
    def gene_dim(self) -> int: return self._gene_dim

    @property
    def supported_perts(self) -> set: return set(self._delta_cache.keys())


# ============================================================================ #
# ExactAnchor
# ============================================================================ #

class ExactAnchor(AnchorDeltaProvider):
    """Per-line delta then averaged across source lines.

        delta(g) = mean_cl in source_lines [ mean(X | line==cl ∧ pert==g)
                                              − mean(X | line==cl ∧ ctrl) ]

    Differs from LinearAnchor: per-line mean BEFORE pooling, not pool BEFORE
    mean. Treated as an upper-bound / reference row in the paper, NOT the
    primary cross-line generalisation method.
    """

    def __init__(self):
        self._delta_cache: dict[str, np.ndarray] = {}
        self._gene_dim: Optional[int] = None

    def fit(self, adata, source_lines, target_line=None, exclude_perts=None, min_cells=50):
        obs = adata.obs
        if target_line is not None:
            for cl in source_lines:
                assert cl != target_line, (
                    f"ExactAnchor.fit: source_line '{cl}' equals target_line; refusing."
                )
        exclude = set(exclude_perts or [])

        def _pb(mask):
            X = adata[mask].X
            if hasattr(X, "toarray"):
                X = X.toarray()
            return np.asarray(X, dtype=np.float64).mean(axis=0)

        per_line_ctrl = {
            cl: _pb((obs["cell_line"].values == cl) & (obs["pert"].values == "control"))
            for cl in source_lines
        }
        self._gene_dim = next(iter(per_line_ctrl.values())).shape[0]

        delta_per_line: dict[str, list[np.ndarray]] = {}
        for cl in source_lines:
            lmask = (obs["cell_line"].values == cl)
            line_perts = sorted(set(obs[lmask]["pert"].astype(str).unique()) - {"control"})
            for pert in line_perts:
                if pert in exclude:
                    continue
                pmask = lmask & (obs["pert"].values == pert)
                if int(pmask.sum()) < min_cells:
                    continue
                delta_per_line.setdefault(pert, []).append(_pb(pmask) - per_line_ctrl[cl])

        for pert, deltas in delta_per_line.items():
            self._delta_cache[pert] = np.stack(deltas).mean(axis=0).astype(np.float32)
        return self

    def delta(self, gene_symbol: str) -> Optional[np.ndarray]:
        return self._delta_cache.get(gene_symbol)

    @property
    def gene_dim(self) -> int: return self._gene_dim

    @property
    def supported_perts(self) -> set: return set(self._delta_cache.keys())


# ============================================================================ #
# RidgeAnchor — wraps the legacy PooledGeneEmbeddingRidgeAnchor below
# ============================================================================ #

class RidgeAnchor(AnchorDeltaProvider):
    """Predicts per-pert delta from a primary gene embedding via ridge regression.

    delta(g) = ridge( primary_gene_embedding[g] )

    Only supports perts that appear in `primary_embedding_path`.
    Currently 6,641 / 7,500-ish genes (88 perts of the 577 test set are OOV).
    """

    def __init__(self, primary_embedding_path: Path):
        import torch
        emb = torch.load(primary_embedding_path, map_location="cpu", weights_only=False)
        assert isinstance(emb, dict), "primary embedding must be dict[symbol -> tensor]"
        self._embeddings = {g: v.cpu().numpy().astype(np.float32) for g, v in emb.items()}
        self._inner = PooledGeneEmbeddingRidgeAnchor()
        self._gene_dim: Optional[int] = None

    def fit(
        self,
        adata,
        source_lines,
        target_line=None,
        exclude_perts=None,
        min_cells=50,
        val_frac: float = 0.1,
        seed: int = 42,
        alpha_grid: Optional[Sequence[float]] = None,
    ):
        # PooledGeneEmbeddingRidgeAnchor is defined later in this file; resolve
        # the default here, not as a class-level default expression.
        if alpha_grid is None:
            alpha_grid = PooledGeneEmbeddingRidgeAnchor.DEFAULT_ALPHA_GRID
        obs = adata.obs
        src_mask = obs["cell_line"].isin(list(source_lines)).values
        if target_line is not None:
            n_tgt_in_src = int((src_mask & (obs["cell_line"].values == target_line)).sum())
            assert n_tgt_in_src == 0, (
                f"RidgeAnchor.fit: source_lines={list(source_lines)} overlaps with "
                f"target_line='{target_line}' on {n_tgt_in_src} cells. Refusing to fit."
            )
        exclude = set(exclude_perts or [])

        def _pb(mask):
            X = adata[mask].X
            if hasattr(X, "toarray"):
                X = X.toarray()
            return np.asarray(X, dtype=np.float64).mean(axis=0)

        X_list, y_list, meta = [], [], []
        for cl in source_lines:
            lmask = (obs["cell_line"].values == cl)
            ctrl_mean = _pb(lmask & (obs["pert"].values == "control"))
            for pert in sorted(set(obs[lmask]["pert"].astype(str).unique()) - {"control"}):
                if pert in exclude or pert not in self._embeddings:
                    continue
                pmask = lmask & (obs["pert"].values == pert)
                if int(pmask.sum()) < min_cells:
                    continue
                X_list.append(self._embeddings[pert])
                y_list.append((_pb(pmask) - ctrl_mean).astype(np.float32))
                meta.append((cl, pert))

        if not X_list:
            raise RuntimeError("RidgeAnchor.fit: no (gene_emb, delta) pairs survived filtering.")

        X_all = np.stack(X_list)
        y_all = np.stack(y_list)
        self._gene_dim = y_all.shape[1]

        # pert-level val split
        unique_perts = sorted({p for _, p in meta})
        rng = np.random.default_rng(seed)
        rng.shuffle(unique_perts)
        n_val = max(1, int(val_frac * len(unique_perts)))
        val_perts = set(unique_perts[:n_val])
        trn_idx = [i for i, (_, p) in enumerate(meta) if p not in val_perts]
        val_idx = [i for i, (_, p) in enumerate(meta) if p in val_perts]

        self._inner.fit(
            X_all[trn_idx], y_all[trn_idx],
            X_all[val_idx] if val_idx else None,
            y_all[val_idx] if val_idx else None,
            alpha_grid=alpha_grid,
        )
        return self

    def delta(self, gene_symbol: str) -> Optional[np.ndarray]:
        emb = self._embeddings.get(gene_symbol)
        if emb is None:
            return None
        return self._inner.predict(emb.reshape(1, -1)).flatten().astype(np.float32)

    @property
    def gene_dim(self) -> int: return self._gene_dim

    @property
    def supported_perts(self) -> set: return set(self._embeddings.keys())


# ============================================================================ #
# Legacy class — kept for the existing reference-baseline scripts.
# New code should use RidgeAnchor (the wrapper above) instead.
# ============================================================================ #


class PooledGeneEmbeddingRidgeAnchor:
    """Pooled gene-embedding ridge anchor.

    中文:
        - fit(X_train, y_train, X_val=None, y_val=None, alpha_grid=...):
          有 val 则 grid search; 无则用 alpha_grid 末端 (最保守) 单独 fit.
        - fit_single(X, y, alpha): 单 alpha 直接 fit.
        - predict(X): 返回 [B, G].
        - save/load: joblib pickled payload.
    """

    DEFAULT_ALPHA_GRID: tuple = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)

    def __init__(self):
        self.scaler = None  # sklearn.preprocessing.StandardScaler
        self.ridge = None   # sklearn.linear_model.Ridge
        self.alpha: Optional[float] = None
        self.val_score_: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        alpha_grid: Sequence[float] = DEFAULT_ALPHA_GRID,
    ) -> "PooledGeneEmbeddingRidgeAnchor":
        """Fit with alpha selection on validation Pearson.

        中文:
            X_train: [N, D_embed] gene embedding.
            y_train: [N, G] pseudobulk delta.
            X_val/y_val: 用于 alpha 选择; 可为 None.
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge

        self.scaler = StandardScaler().fit(X_train)
        Xs = self.scaler.transform(X_train)

        if X_val is None or y_val is None:
            self.alpha = float(alpha_grid[-1])
            self.ridge = Ridge(alpha=self.alpha, fit_intercept=True).fit(Xs, y_train)
            return self

        Xv = self.scaler.transform(X_val)
        best_score, best_alpha, best_ridge = -np.inf, None, None
        for a in alpha_grid:
            m = Ridge(alpha=float(a), fit_intercept=True).fit(Xs, y_train)
            y_pred = m.predict(Xv)
            score = _mean_row_pearson(y_pred, y_val)
            if score > best_score:
                best_score, best_alpha, best_ridge = score, float(a), m
        self.alpha = best_alpha
        self.ridge = best_ridge
        self.val_score_ = float(best_score)
        return self

    def fit_single(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float = 1.0,
    ) -> "PooledGeneEmbeddingRidgeAnchor":
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.alpha = float(alpha)
        self.ridge = Ridge(alpha=self.alpha, fit_intercept=True).fit(Xs, y)
        return self

    # ------------------------------------------------------------------ #
    # Predict / serialization
    # ------------------------------------------------------------------ #
    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.ridge is not None and self.scaler is not None, "call fit(_single) first"
        return self.ridge.predict(self.scaler.transform(X))

    def save(self, path: Path) -> None:
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "scaler": self.scaler,
                "ridge": self.ridge,
                "alpha": self.alpha,
                "val_score": self.val_score_,
            },
            Path(path),
        )

    def load(self, path: Path) -> "PooledGeneEmbeddingRidgeAnchor":
        import joblib
        payload = joblib.load(Path(path))
        self.scaler = payload["scaler"]
        self.ridge = payload["ridge"]
        self.alpha = payload.get("alpha")
        self.val_score_ = payload.get("val_score")
        return self


# ---------------------------------------------------------------------------- #
# utils
# ---------------------------------------------------------------------------- #

def _mean_row_pearson(A: np.ndarray, B: np.ndarray) -> float:
    """Row-wise Pearson 均值. A, B: [N, G]."""
    from scipy.stats import pearsonr
    scores = []
    for i in range(A.shape[0]):
        if np.std(A[i]) < 1e-12 or np.std(B[i]) < 1e-12:
            continue
        r, _ = pearsonr(A[i], B[i])
        if np.isfinite(r):
            scores.append(r)
    return float(np.mean(scores)) if scores else 0.0
