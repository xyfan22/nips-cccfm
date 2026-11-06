"""Evaluation metrics and standardized cross-line evaluator.

中文:
    本模块是 ccCFM 项目所有方法的**唯一标准评估入口**.
    从 2026-04-21 起,  所有 baseline / ccCFM / ablation 的结果都通过
    CrossLineEvaluator 产出,  保证 schema 一致、口径一致.

    方法论参考: Vinas et al.,  "Systema: A framework for evaluating genetic
    perturbation response prediction beyond systematic variation", 
    Nature Biotechnology 2025.

关键度量:

    Mean-level (对 pseudobulk profile [G]):
        pearson_ctrl   - PearsonΔ with reference = target-line control mean  (领域标准,  跟文献可比)
        pearson_pert   - PearsonΔ with reference = target-line perturbation centroid mean
                         (Systema 主张,  扣掉 systematic variation 后更 honest)
        mse            - MSE(pred_profile,  true_profile). reference-independent.
        pearson20_ctrl - PearsonΔ 上只取预先算好的 top-20 DE 基因索引 (source-line |Δ|)
        pearson20_pert - 同上但 reference = pert centroid
        deg_overlap20  - top-20 by |pred_delta_ctrl| ∩ top-20 by |gt_delta_ctrl| / 20
        deg_overlap50  - 同 k=50
        centroid_accuracy - Systema: 预测 profile 在所有 target-line pert centroids
                           中距离 rank 的 normalized 正确率 (1.0 = 最近;
                           0.5 = 中位; 随机 ≈ 0.5).

    Distribution-level (对 cell 样本集 [N,  G],  仅 ccCFM / 生成式方法产出):
        mmd            - multi-scale RBF MMD^2
        energy         - energy distance
        w2_sinkhorn    - Sinkhorn-W2 in PCA subspace
        moment         - per-gene mean/var/skew L1 距离
"""

from __future__ import annotations

from typing import Iterable,  Optional,  Sequence

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Low-level primitives (functional,  no state)
# --------------------------------------------------------------------------- #

def _safe_pearson(a: np.ndarray,  b: np.ndarray) -> float:
    """Pearson 相关,  常数向量或 NaN 时返回 0.0."""
    from scipy.stats import pearsonr
    if a.size < 2 or b.size < 2:
        return 0.0
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    r,  _ = pearsonr(a,  b)
    return float(r) if np.isfinite(r) else 0.0


def pearson_delta(
    pred_profile: np.ndarray, 
    true_profile: np.ndarray, 
    reference: np.ndarray, 
) -> float:
    """PearsonΔ = Pearson(pred - ref,  true - ref).

    中文: reference 的选择是关键:
        - reference = control mean    -> 标准 PearsonΔ (可能被 systematic variation 虚高)
        - reference = pert centroid   -> Systema 风格 (扣掉共享 drift,  更 honest)
    """
    return _safe_pearson(pred_profile - reference,  true_profile - reference)


def pearson_delta_topk(
    pred_profile: np.ndarray, 
    true_profile: np.ndarray, 
    reference: np.ndarray, 
    topk_idxs: Optional[np.ndarray], 
) -> Optional[float]:
    """PearsonΔ restricted to top-k DE gene indices."""
    if topk_idxs is None or len(topk_idxs) < 2:
        return None
    return pearson_delta(
        pred_profile[topk_idxs],  true_profile[topk_idxs],  reference[topk_idxs]
    )


def mse_profile(pred_profile: np.ndarray,  true_profile: np.ndarray) -> float:
    """MSE between pred and true profile. Reference-independent."""
    return float(np.mean((pred_profile - true_profile) ** 2))


def deg_overlap_topk(
    pred_delta: np.ndarray, 
    gt_delta: np.ndarray, 
    k: int = 20, 
) -> float:
    """Set-overlap DEG recall: |top-k by |pred|| ∩ |top-k by |gt|| / k."""
    if len(pred_delta) < k or len(gt_delta) < k:
        return 0.0
    gt_top = set(np.argsort(np.abs(gt_delta))[::-1][:k].tolist())
    pred_top = set(np.argsort(np.abs(pred_delta))[::-1][:k].tolist())
    return len(gt_top & pred_top) / k


def centroid_accuracy_score(
    pred_profile: np.ndarray, 
    centroids: np.ndarray,      # [N_perts,  G]
    true_idx: int, 
) -> float:
    """Systema-style centroid accuracy.

    中文: 计算 pred_profile 到 centroids 里每行的 L2 距离,  看正确 pert 的距离
          在所有 centroid 中的排名 (normalized):
              1.0 -> self 最近 (最好), 
              0.5 -> 中位, 
              0.0 -> self 最远.
    """
    if centroids is None or len(centroids) < 2:
        return 0.0
    dists = np.linalg.norm(centroids - pred_profile[None,  :],  axis=1)
    self_dist = dists[true_idx]
    n_others = len(dists) - 1
    n_further = int(np.sum(dists > self_dist))
    return n_further / n_others


# --------------------------------------------------------------------------- #
# Distributional metrics (cell-level)
# --------------------------------------------------------------------------- #

def _pairwise_sq_dist(A: torch.Tensor,  B: torch.Tensor) -> torch.Tensor:
    A2 = (A * A).sum(-1,  keepdim=True)
    B2 = (B * B).sum(-1,  keepdim=True)
    d2 = A2 + B2.T - 2.0 * A @ B.T
    return torch.clamp(d2,  min=0.0)


def mmd_rbf_multiscale(
    X: torch.Tensor, 
    Y: torch.Tensor, 
    bandwidths: Iterable[float] = (0.1,  1.0,  10.0), 
) -> float:
    """Multi-scale RBF MMD^2 (biased)."""
    X = X.float(); Y = Y.float()
    dxx = _pairwise_sq_dist(X,  X)
    dyy = _pairwise_sq_dist(Y,  Y)
    dxy = _pairwise_sq_dist(X,  Y)
    bws = list(bandwidths)
    mmd = 0.0
    for bw in bws:
        two_bw2 = 2.0 * bw * bw
        mmd += float(
            torch.exp(-dxx / two_bw2).mean()
            + torch.exp(-dyy / two_bw2).mean()
            - 2.0 * torch.exp(-dxy / two_bw2).mean()
        )
    return mmd / len(bws)


def energy_distance(X: torch.Tensor,  Y: torch.Tensor) -> float:
    X = X.float(); Y = Y.float()
    dxy = torch.sqrt(_pairwise_sq_dist(X,  Y) + 1e-12).mean()
    dxx = torch.sqrt(_pairwise_sq_dist(X,  X) + 1e-12).mean()
    dyy = torch.sqrt(_pairwise_sq_dist(Y,  Y) + 1e-12).mean()
    return float(2.0 * dxy - dxx - dyy)


def sinkhorn_w2(
    X: torch.Tensor, 
    Y: torch.Tensor, 
    pca_dim: int = 50, 
    reg: float = 0.05, 
) -> float:
    try:
        import ot
    except ImportError as e:
        raise RuntimeError("pip install pot for Sinkhorn W2") from e
    from sklearn.decomposition import PCA
    Xn = X.cpu().numpy().astype(np.float64)
    Yn = Y.cpu().numpy().astype(np.float64)
    allx = np.concatenate([Xn,  Yn],  axis=0)
    dim = min(pca_dim,  allx.shape[1],  allx.shape[0] - 1)
    pca = PCA(n_components=dim)
    red = pca.fit_transform(allx)
    Xp,  Yp = red[: len(Xn)],  red[len(Xn):]
    a = np.ones(len(Xp)) / len(Xp)
    b = np.ones(len(Yp)) / len(Yp)
    M = ot.dist(Xp,  Yp,  metric="sqeuclidean")
    M /= (M.max() + 1e-12)
    return float(ot.sinkhorn2(a,  b,  M,  reg))


def per_gene_moment_match(X: torch.Tensor,  Y: torch.Tensor) -> float:
    from scipy.stats import skew
    Xn = X.cpu().numpy(); Yn = Y.cpu().numpy()
    m = np.abs(Xn.mean(0) - Yn.mean(0)).mean()
    v = np.abs(Xn.var(0) - Yn.var(0)).mean()
    
    # 这个地方有个bug: STATE 的预测输出是 2000 维, 经过 invert_alignment 映射回我们的 6642 维后, 有 4642 个基因列全是 0（STATE没覆盖这些基因）. 这些零列的方差为 0, scipy.stats.skew() 对常数序列返回 NaN, 然后 .mean() 把 NaN 传播到整个结果. 
    # 改之前
    # s = np.abs(skew(Xn,  axis=0) - skew(Yn,  axis=0)).mean()

    # 改之后
    sk = np.abs(skew(Xn,  axis=0) - skew(Yn,  axis=0))
    sk = np.nan_to_num(sk,  nan=0.0) # 零方差列的skew差异视为0
    s = sk.mean()
    
    return float(m + v + s)

# --------------------------------------------------------------------------- #
# Standardized cross-line evaluator
# --------------------------------------------------------------------------- #

class CrossLineEvaluator:
    """Standardized evaluator for cross-cell-line perturbation prediction.

    中文:
        本类是 ccCFM 项目**所有方法**的统一评估入口 — baseline,  ccCFM,  ablation
        一律通过它产出 metrics. 保证 schema / reference / top-k DE 定义一致.

        __init__ 时预计算一次:
            - target_ctrl_reference   target-line control mean
            - target_pert_reference   target-line pert centroid mean (Systema)
            - target_pert_centroids   per test pert,  target-line pseudobulk (for centroid accuracy)
            - topk_de_idxs_by_pert    per test pert,  top-k DE gene indices
                                      (computed from source-line |Δ|,  no target leakage)

        评估方法:
            evaluate_pert_mean(pert,  pred_profile)
                -> dict. pred_profile 是 [G] 预测 pseudobulk. 对应 mean-only baseline.
            evaluate_pert_cell_level(pert,  pred_cells,  gt_cells=None)
                -> dict. pred_cells 是 [N,  G] 预测 cells. 对 ccCFM / 生成式方法.

    参考:
        Vinas et al.,  "Systema: A framework for evaluating genetic perturbation
        response prediction beyond systematic variation",  Nat Biotech 2025.
    """

    METRIC_COLUMNS = [
        "pearson_ctrl", 
        "pearson_pert", 
        "mse", 
        "pearson20_ctrl", 
        "pearson20_pert", 
        "deg_overlap20", 
        "deg_overlap50", 
        "centroid_accuracy", 
        "mmd", 
        "energy", 
        "w2_sinkhorn", 
        "moment", 
        "n_target_cells", 
    ]

    def __init__(
        self, 
        adata, 
        source_lines: Sequence[str], 
        target_line: str, 
        test_perts: Sequence[str], 
        min_cells: int = 50, 
        topk_de: int = 20, 
        verbose: bool = True, 
    ):
        self.adata = adata
        self.source_lines = list(source_lines)
        self.target_line = target_line
        self.test_perts = list(test_perts)
        self.min_cells = min_cells
        self.topk_de = topk_de
        self.verbose = verbose
        self._build_caches()

    # -------------------- cache construction -------------------- #
    def _pb(self,  mask) -> np.ndarray:
        """Pseudobulk mean over a cell mask."""
        X = self.adata[mask].X
        if hasattr(X,  "toarray"):
            X = X.toarray()
        return np.asarray(X,  dtype=np.float64).mean(axis=0)

    def _build_caches(self) -> None:
        obs = self.adata.obs
        target = self.target_line

        # target-line control mean
        self.target_ctrl_reference = self._pb(
            (obs["cell_line"] == target) & (obs["pert"] == "control")
        )

        # source-line control mean (for DE computation)
        self.source_ctrl = self._pb(
            obs["cell_line"].isin(self.source_lines) & (obs["pert"] == "control")
        )

        # per-pert target-line centroids (for centroid accuracy + pert reference)
        self.target_pert_centroids: dict[str,  np.ndarray] = {}
        self.target_pert_n_cells: dict[str,  int] = {}
        for pert in self.test_perts:
            mask = (obs["cell_line"] == target) & (obs["pert"] == pert)
            n = int(mask.sum())
            self.target_pert_n_cells[pert] = n
            if n >= self.min_cells:
                self.target_pert_centroids[pert] = self._pb(mask)
        if self.verbose:
            print(f"[Evaluator] target-line centroids cached: "
                  f"{len(self.target_pert_centroids)}/{len(self.test_perts)}")

        # pert-centroid reference = mean over all target-line pert centroids
        if self.target_pert_centroids:
            all_centroids = np.stack(list(self.target_pert_centroids.values()))
            self.target_pert_reference = all_centroids.mean(axis=0)
            self._centroids_perts = list(self.target_pert_centroids.keys())
            self._centroids_matrix = all_centroids
            self._centroids_idx = {p: i for i,  p in enumerate(self._centroids_perts)}
        else:
            raise RuntimeError("No target-line centroids met min_cells; cannot evaluate.")

        # top-k DE indices per pert from source-line data
        self.topk_de_idxs_by_pert: dict[str,  np.ndarray] = {}
        for pert in self.test_perts:
            mask = obs["cell_line"].isin(self.source_lines) & (obs["pert"] == pert)
            if int(mask.sum()) < self.min_cells:
                continue
            src_pert_mean = self._pb(mask)
            delta = np.abs(src_pert_mean - self.source_ctrl)
            self.topk_de_idxs_by_pert[pert] = np.argsort(delta)[::-1][: self.topk_de]
        if self.verbose:
            print(f"[Evaluator] top-{self.topk_de} DE indices cached from sources: "
                  f"{len(self.topk_de_idxs_by_pert)}/{len(self.test_perts)}")

    # -------------------- evaluation API -------------------- #
    def evaluate_pert_mean(
        self, 
        pert: str, 
        pred_profile: np.ndarray, 
    ) -> Optional[dict]:
        """Evaluate a pseudobulk-level prediction.

        中文:
            pert:         目标 pert gene symbol
            pred_profile: [G] 预测的 target-line × pert 的 pseudobulk 表达
        返回:
            dict with METRIC_COLUMNS 里的字段. mean-only method 的 mmd/energy/w2/moment 全为 None.
        """
        if pert not in self.target_pert_centroids:
            return None
        true_profile = self.target_pert_centroids[pert]
        topk = self.topk_de_idxs_by_pert.get(pert)

        return {
            "pearson_ctrl":     pearson_delta(pred_profile,  true_profile,  self.target_ctrl_reference), 
            "pearson_pert":     pearson_delta(pred_profile,  true_profile,  self.target_pert_reference), 
            "mse":              mse_profile(pred_profile,  true_profile), 
            "pearson20_ctrl":   pearson_delta_topk(pred_profile,  true_profile,  self.target_ctrl_reference,  topk), 
            "pearson20_pert":   pearson_delta_topk(pred_profile,  true_profile,  self.target_pert_reference,  topk), 
            "deg_overlap20":    deg_overlap_topk(
                pred_profile - self.target_ctrl_reference, 
                true_profile - self.target_ctrl_reference,  k=20), 
            "deg_overlap50":    deg_overlap_topk(
                pred_profile - self.target_ctrl_reference, 
                true_profile - self.target_ctrl_reference,  k=50), 
            "centroid_accuracy": centroid_accuracy_score(
                pred_profile,  self._centroids_matrix,  self._centroids_idx[pert]), 
            "mmd":              None, 
            "energy":           None, 
            "w2_sinkhorn":      None, 
            "moment":           None, 
            "n_target_cells":   self.target_pert_n_cells[pert], 
        }

    def evaluate_pert_cell_level(
        self, 
        pert: str, 
        pred_cells: np.ndarray, 
        compute_heavy: bool = False, 
    ) -> Optional[dict]:
        """Evaluate a cell-level prediction (for ccCFM / generative methods).

        中文:
            pred_cells: [N,  G] 预测的 target-line × pert 细胞集合.
            compute_heavy=True 时再算 w2_sinkhorn / moment (慢,  默认关).
        """
        if pert not in self.target_pert_centroids:
            return None
        # mean-level metrics on average of pred_cells
        pred_profile = pred_cells.mean(axis=0)
        out = self.evaluate_pert_mean(pert,  pred_profile)
        if out is None:
            return None

        # distributional: compare pred_cells vs target-line GT cells
        obs = self.adata.obs
        gt_mask = (obs["cell_line"] == self.target_line) & (obs["pert"] == pert)
        X_gt = self.adata[gt_mask].X
        if hasattr(X_gt,  "toarray"):
            X_gt = X_gt.toarray()
        X_gt = np.asarray(X_gt,  dtype=np.float32)
        X_pred = np.asarray(pred_cells,  dtype=np.float32)

        Xp = torch.from_numpy(X_pred)
        Xg = torch.from_numpy(X_gt)
        out["mmd"] = mmd_rbf_multiscale(Xp,  Xg)
        out["energy"] = energy_distance(Xp,  Xg)
        if compute_heavy:
            out["w2_sinkhorn"] = sinkhorn_w2(Xp,  Xg)
            out["moment"] = per_gene_moment_match(Xp,  Xg)
        return out


# --------------------------------------------------------------------------- #
# Backwards-compat deprecated shims (kept only so old scripts don't crash
# while we migrate; do NOT use in new code)
# --------------------------------------------------------------------------- #

def pearson_pseudobulk(pred_delta: np.ndarray,  gt_delta: np.ndarray) -> float:
    """DEPRECATED. 用 CrossLineEvaluator.evaluate_pert_mean 代替."""
    return _safe_pearson(pred_delta,  gt_delta)


def mse_pseudobulk(pred_delta: np.ndarray,  gt_delta: np.ndarray) -> float:
    """DEPRECATED. 用 CrossLineEvaluator 代替."""
    return float(np.mean((pred_delta - gt_delta) ** 2))


def deg_topk_recall(pred_delta: np.ndarray,  gt_delta: np.ndarray,  k: int = 20) -> float:
    """DEPRECATED alias of deg_overlap_topk."""
    return deg_overlap_topk(pred_delta,  gt_delta,  k)
