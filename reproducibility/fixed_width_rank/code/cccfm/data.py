"""ccCFM 数据加载.

中文:
    - CCCFMDataset: 训练用, 一个 item = 一个 source-line 的 perturbed cell, 配上
      同 line 随机采样的 control cell 作为 x_0. h 来自 x_0 (control) 的 encoder
      embedding, 与 x_1 无关 (cross-line 时 target 也只能见到 control).
    - build_train_dataloader / build_eval_dataloader: 主脚本入口.
    - TargetControls: eval 时载入 target-line control cells + 对应 h,
      用于 ODE 积分生成预测 (见 scripts/31_eval_cccfm.py).

v1 简化:
    - pair_by_line=True: 同 line 配对保证 conditioning 干净.
    - 无 OT 重排 (flow_module 的 use_ot_pairing=False).
    - 无 per-(line, pert) batching (D1 时再加).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


def _as_dense_row(x, dtype=np.float32) -> np.ndarray:
    """AnnData .X single-row access -> (G,) float32 ndarray."""
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=dtype).ravel()


def apply_source_alignment(
    adata,
    encoder_key: str,
    source_lines,
    mode: str = "zscore",
    eps: float = 1e-6,
):
    """Cross-line h alignment.

    Two modes (advisor 2026-04-26):

    'zscore' — *weak*. Single source-side normalizer:
        h_aligned = (h - source_ctrl_mean) / source_ctrl_std
        Same shift+scale applied to every cell. Source-ctrl ends up at N(0, 1);
        target-ctrl may still be OOD in this standardized space if the encoder
        has a real cross-line shift.

    'zscore_per_line' — *strong*. Per-cell-line domain normalization:
        for each cell_line cl in obs.cell_line.unique():
            mean_cl, std_cl = h on (cell_line==cl, pert==control)
            h[cell_line==cl] ← (h[cell_line==cl] - mean_cl) / std_cl
        Each line's controls become N(0, 1) in their own reference frame.
        Cross-line shift removed entirely; what remains is per-cell deviation
        from the line's basal state — what we actually want the velocity field
        to condition on.

    Returns the stats needed to replay the same alignment at eval time:
        mode='zscore'        →  (mean: [D], std: [D])
        mode='zscore_per_line' →  (cell_lines: list[str], means: [n_lines, D], stds: [n_lines, D])

    Mutates `adata.obsm[f'h_{encoder_key}']` in place.
    """
    if mode == "none":
        H = np.asarray(adata.obsm[f"h_{encoder_key}"], dtype=np.float32)
        D = H.shape[1]
        return np.zeros(D, dtype=np.float32), np.ones(D, dtype=np.float32)

    if mode == "zscore":
        obs = adata.obs
        src_ctrl_mask = (
            obs["cell_line"].isin(list(source_lines)).values
            & (obs["pert"].values == "control")
        )
        H = np.asarray(adata.obsm[f"h_{encoder_key}"], dtype=np.float32)
        src_h = H[src_ctrl_mask]
        if len(src_h) < 100:
            raise RuntimeError(
                f"only {len(src_h)} source-ctrl cells found for alignment; "
                f"check source_lines={list(source_lines)} and obsm key 'h_{encoder_key}'"
            )
        mean = src_h.mean(0).astype(np.float32)
        std = src_h.std(0).astype(np.float32).clip(min=eps)
        adata.obsm[f"h_{encoder_key}"] = ((H - mean) / std).astype(np.float32)
        return mean, std

    if mode == "zscore_per_line":
        obs = adata.obs
        H = np.asarray(adata.obsm[f"h_{encoder_key}"], dtype=np.float32)
        D = H.shape[1]
        H_out = H.copy()
        cell_lines = sorted(obs["cell_line"].astype(str).unique().tolist())
        means = np.zeros((len(cell_lines), D), dtype=np.float32)
        stds = np.ones((len(cell_lines), D), dtype=np.float32)
        for i, cl in enumerate(cell_lines):
            line_mask = (obs["cell_line"].values == cl)
            ctrl_mask = line_mask & (obs["pert"].values == "control")
            n_ctrl = int(ctrl_mask.sum())
            if n_ctrl >= 100:
                m = H[ctrl_mask].mean(0).astype(np.float32)
                s = H[ctrl_mask].std(0).astype(np.float32).clip(min=eps)
            else:
                # fall back to all cells of this line if controls are sparse
                if int(line_mask.sum()) == 0:
                    continue
                m = H[line_mask].mean(0).astype(np.float32)
                s = H[line_mask].std(0).astype(np.float32).clip(min=eps)
            means[i] = m
            stds[i] = s
            H_out[line_mask] = (H[line_mask] - m) / s
        adata.obsm[f"h_{encoder_key}"] = H_out.astype(np.float32)
        return cell_lines, means, stds

    raise ValueError(f"unknown alignment mode '{mode}'")


def _as_dense_matrix(x, dtype=np.float32) -> np.ndarray:
    """AnnData .X multi-row slice -> (N, G) float32 ndarray."""
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=dtype)


class CCCFMDataset(Dataset):
    """训练用: 一个 item = 一个 perturbed cell + 同 line 随机 control.

    __getitem__(idx) -> dict:
        x_0:  [G]   control cell (同 source line)
        x_1:  [G]   perturbed cell
        h:    [D]   encoder(x_0) 存在 obsm[f"h_{encoder_key}"] 里
        pert: str   perturbation gene symbol (≠ 'control')
        line: str   source cell_line (debug/loss 分组用)
    """

    def __init__(
        self,
        adata,
        source_lines: Sequence[str],
        pert_set: Sequence[str],
        encoder_key: str,
        seed: int = 42,
    ):
        obs = adata.obs
        src_mask = obs["cell_line"].isin(list(source_lines)).values
        pert_mask = (obs["pert"].values != "control") & obs["pert"].isin(list(pert_set)).values

        self.adata = adata
        self.X = adata.X
        self.H = np.asarray(adata.obsm[f"h_{encoder_key}"], dtype=np.float32)
        self.line_arr = obs["cell_line"].astype(str).values
        self.pert_arr = obs["pert"].astype(str).values

        self.perturbed_idx = np.flatnonzero(src_mask & pert_mask)

        self.line_ctrl_idx: dict[str, np.ndarray] = {}
        for line in source_lines:
            cmask = (obs["cell_line"].values == line) & (obs["pert"].values == "control")
            self.line_ctrl_idx[str(line)] = np.flatnonzero(cmask)
            if len(self.line_ctrl_idx[str(line)]) == 0:
                raise RuntimeError(f"No control cells found for source line '{line}'.")

        self.seed = seed

    def __len__(self) -> int:
        return len(self.perturbed_idx)

    def __getitem__(self, idx: int) -> dict:
        # Per-sample RNG seeded by (epoch-implicit index + self.seed); torch's
        # DataLoader worker_init_fn re-seeds np.random per worker.
        i = int(self.perturbed_idx[idx])
        line_i = self.line_arr[i]
        ctrl_pool = self.line_ctrl_idx[line_i]
        j = int(np.random.randint(0, len(ctrl_pool)))
        ctrl_i = int(ctrl_pool[j])

        x_1 = _as_dense_row(self.X[i])
        x_0 = _as_dense_row(self.X[ctrl_i])
        h = self.H[ctrl_i]
        return {
            "x_0": x_0,
            "x_1": x_1,
            "h": h,
            "pert": str(self.pert_arr[i]),
            "line": str(line_i),
        }


def cccfm_collate(batch: List[dict]) -> dict:
    return {
        "x_0": torch.from_numpy(np.stack([b["x_0"] for b in batch])),
        "x_1": torch.from_numpy(np.stack([b["x_1"] for b in batch])),
        "h": torch.from_numpy(np.stack([b["h"] for b in batch])),
        "perts": [b["pert"] for b in batch],
        "lines": [b["line"] for b in batch],
    }


def _worker_init_fn(worker_id: int) -> None:
    # Make DataLoader workers have distinct np.random state.
    info = torch.utils.data.get_worker_info()
    base = (info.seed if info is not None else 0) + worker_id
    np.random.seed(base % (2**31 - 1))


def build_train_dataloader(
    adata,
    source_lines: Sequence[str],
    pert_set: Sequence[str],
    encoder_key: str,
    batch_size: int = 256,
    num_workers: int = 4,
    val_frac: float = 0.05,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, CCCFMDataset]:
    """Build (train_loader, val_loader, underlying dataset).

    中文:
        - 在 perturbed-cell 层面做 95/5 split (不保留 pert holdout);
          真正的 cross-line 评估在 eval 脚本里独立做.
    """
    ds = CCCFMDataset(adata, source_lines, pert_set, encoder_key, seed=seed)
    n = len(ds)
    val_n = max(1, int(round(n * val_frac)))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    val_idx = perm[:val_n]
    train_idx = perm[val_n:]

    train_loader = DataLoader(
        Subset(ds, list(train_idx)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=cccfm_collate,
        worker_init_fn=_worker_init_fn,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        Subset(ds, list(val_idx)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=True,
        drop_last=False,
        collate_fn=cccfm_collate,
        worker_init_fn=_worker_init_fn,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader, ds


# --------------------------------------------------------------------------- #
# R2: group-safe BatchSampler + per-pert val split + residual loader         #
# --------------------------------------------------------------------------- #

class GroupBatchSampler(torch.utils.data.Sampler):
    """Batches contain perturbed cells from the SAME (cell_line, pert) group.

    Why: residual CFM uses `apply_anchor_torch(x_control, ctx_mean, delta_g)`
    which subtracts the BATCH mean of x_control. If a batch mixes lines or perts,
    the per-pert anchor cloud no longer matches per-pert shifted-control —
    correctness condition, not optimisation (advisor 2026-04-26 round 3).

    Iteration: for each (line, pert) group, yield ceil(group_size / batch_size)
    batches; the final batch in a group is padded with replacement so every
    yielded batch has length exactly `batch_size`. Order of groups + within-group
    sampling is reshuffled each epoch (call `set_epoch(epoch)` to vary seed).
    """

    def __init__(
        self,
        dataset,                 # CCCFMDataset or torch.utils.data.Subset of one
        batch_size: int,
        seed: int = 42,
    ):
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        # Resolve Subset → base dataset + local-to-global mapping
        if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
            self._base = dataset.dataset
            self._local_indices = list(dataset.indices)
        else:
            self._base = dataset
            self._local_indices = list(range(len(dataset)))

        # Build groups: (line, pert) -> list of LOCAL indices into the dataset
        groups: dict[tuple[str, str], list[int]] = {}
        for li in self._local_indices:
            gi = int(self._base.perturbed_idx[li])
            line = str(self._base.line_arr[gi])
            pert = str(self._base.pert_arr[gi])
            groups.setdefault((line, pert), []).append(li)
        self.groups = {k: np.asarray(v, dtype=np.int64) for k, v in groups.items()}
        self.group_keys = sorted(groups.keys())
        self._n_batches = sum(
            max(1, (len(v) + self.batch_size - 1) // self.batch_size)
            for v in self.groups.values()
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._n_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        all_batches: list[list[int]] = []
        for k in self.group_keys:
            idxs = self.groups[k]
            n = len(idxs)
            n_batches = max(1, (n + self.batch_size - 1) // self.batch_size)
            for _ in range(n_batches):
                if n >= self.batch_size:
                    pick = rng.choice(idxs, self.batch_size, replace=False)
                else:
                    pick = rng.choice(idxs, self.batch_size, replace=True)
                all_batches.append(pick.tolist())
        rng.shuffle(all_batches)
        for b in all_batches:
            yield b


def pert_holdout_split(
    adata,
    source_lines,
    n_holdout_perts: int = 80,
    seed: int = 42,
    exclude_perts: Optional[Iterable[str]] = None,
    min_cells_per_pert: int = 50,
) -> tuple[list[str], list[str]]:
    """Pick `n_holdout_perts` source-line perts as a per-pert val set, disjoint
    from `exclude_perts` (typically the 577 Jurkat shared test perts).

    Returns (train_perts, val_perts), both sorted, disjoint from each other and
    from exclude_perts.

    Caveat (advisor R2.5): with non-parametric anchors (Linear / Exact), the
    val perts' anchor delta is computed from val perts' source cells, so val is
    not strictly gene-OOD on those anchors. With parametric anchor (Ridge / GO),
    refit the anchor with `exclude_perts=val_perts` to get true gene-OOD val.
    """
    obs = adata.obs
    src_mask = obs["cell_line"].isin(list(source_lines)).values
    perts, counts = np.unique(obs["pert"].values[src_mask], return_counts=True)
    excl = set(exclude_perts or [])
    candidates = sorted(
        p for p, c in zip(perts, counts)
        if p != "control" and p not in excl and c >= min_cells_per_pert
    )
    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)
    n = min(n_holdout_perts, max(0, len(candidates) - 1))
    val_perts = sorted(candidates[:n])
    train_perts = sorted(set(candidates[n:]) | (set(perts) - excl - set(val_perts) - {"control"}))
    train_perts = [p for p in train_perts if p in set(perts) and p != "control" and p not in val_perts and p not in excl]
    return sorted(train_perts), val_perts


def compute_per_line_ctrl_means(adata, lines: Sequence[str]) -> dict[str, np.ndarray]:
    """Per-line control-cell mean (used as `context_ctrl_mean` in apply_anchor)."""
    obs = adata.obs
    out = {}
    for cl in lines:
        mask = (obs["cell_line"].values == cl) & (obs["pert"].values == "control")
        if not mask.any():
            raise RuntimeError(f"compute_per_line_ctrl_means: no controls for line '{cl}'")
        X = adata[mask].X
        if hasattr(X, "toarray"):
            X = X.toarray()
        out[cl] = np.asarray(X, dtype=np.float64).mean(axis=0).astype(np.float32)
    return out


def build_residual_dataloaders(
    adata,
    source_lines: Sequence[str],
    train_perts: Sequence[str],
    val_perts: Sequence[str],
    encoder_key: str,
    batch_size: int = 256,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, "CCCFMDataset", "CCCFMDataset"]:
    """Build (train_loader, val_loader, train_dataset, val_dataset) with:
    - GroupBatchSampler (same-(line, pert) batches),
    - per-pert holdout val,
    - both share the same encoder context obsm[h_<encoder_key>].

    Anchor + line_ctrl_means are constructed separately by the trainer.
    """
    train_ds = CCCFMDataset(
        adata, source_lines, list(train_perts), encoder_key, seed=seed,
    )
    val_ds = CCCFMDataset(
        adata, source_lines, list(val_perts), encoder_key, seed=seed + 1,
    )

    train_sampler = GroupBatchSampler(train_ds, batch_size=batch_size, seed=seed)
    val_sampler = GroupBatchSampler(val_ds, batch_size=batch_size, seed=seed + 1)

    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler, num_workers=num_workers,
        pin_memory=True, collate_fn=cccfm_collate,
        worker_init_fn=_worker_init_fn,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_sampler=val_sampler, num_workers=max(1, num_workers // 2),
        pin_memory=True, collate_fn=cccfm_collate,
        worker_init_fn=_worker_init_fn,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader, train_ds, val_ds


# --------------------------------------------------------------------------- #
# Inference-time helper                                                        #
# --------------------------------------------------------------------------- #

class TargetControls:
    """Target-line control cells + their per-cell context embedding, kept
    in CPU memory for repeated per-pert sampling during evaluation.
    """

    def __init__(self, adata, target_line: str, encoder_key: str):
        obs = adata.obs
        mask = (obs["cell_line"].values == target_line) & (obs["pert"].values == "control")
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            raise RuntimeError(f"No control cells found for target_line '{target_line}'.")
        self.X = _as_dense_matrix(adata.X[idx])
        self.H = np.asarray(adata.obsm[f"h_{encoder_key}"], dtype=np.float32)[idx]
        self.n = len(idx)

    def sample_batches(
        self,
        n_samples: int,
        batch_size: int,
        rng: np.random.Generator,
    ) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        """Yield `(x_0, h)` batches totaling `n_samples` cells.

        n_samples == -1 表示枚举全部 control cells (不重复).
        否则按均匀 with-replacement 采样 n_samples 次.
        """
        if n_samples < 0:
            idxs = np.arange(self.n)
        else:
            idxs = rng.integers(0, self.n, size=n_samples)
        for start in range(0, len(idxs), batch_size):
            sel = idxs[start : start + batch_size]
            yield self.X[sel], self.H[sel]

    def sample_full(
        self,
        n_samples: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the FULL `(x_0, h)` cloud at once (no chunking).

        Required by residual eval: x0_anchor must be built with a single
        mean-centering over the entire sampled cloud — chunking inside the
        anchor step would introduce per-chunk drift that diverges from the
        shifted-control baseline (advisor 2026-04-26 round 4).
        """
        if n_samples < 0:
            idxs = np.arange(self.n)
        else:
            idxs = rng.integers(0, self.n, size=n_samples)
        return self.X[idxs], self.H[idxs]
