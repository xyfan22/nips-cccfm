"""Perturbation token encoding.

中文:
    - PertTokenizer 是 nn.Module, 参数 (投影层 / 特殊 token buffer / pretrained
      embedding matrix) 正确随 optimizer / state_dict / .to(device) 管理.
    - 两个模式:
        'gene_embedding'     —— 用主 gene embedding 直接映射到 out_dim.
        'concat_secondary'   —— 主 + 辅 gene embedding concat 再 MLP 压回.
    - 三类 token 严格区分, 避免 "未扰动" 与 "未知扰动" 语义坍缩:
        control_token  —— 明确的 no-perturbation 标识 (默认 zero).
        unk_token      —— 已知扰动但当前 gene embedding 缺失 (默认 mean 观测向量).
        genuine        —— 有 embedding 的基因, 走投影得到.

    - 采用 nn.Embedding.from_pretrained(frozen) 而不是 Python dict 查表:
      (a) .to(device) 自动迁移; (b) DDP 下 state_dict 一致; (c) 索引查询比
      逐样本 dict 查询快.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn


CONTROL_IDX = 0
UNK_IDX = 1


class PertTokenizer(nn.Module):
    """gene_symbol -> z_pert ∈ R^{out_dim}.

    中文参数:
        mode:
            'gene_embedding'     使用 primary embedding.
            'concat_secondary'   concat primary || secondary, 经 MLP.
        primary_embedding_path:
            torch.save 的 dict[gene_symbol -> 1D tensor].
        secondary_embedding_path:
            同上; mode='concat_secondary' 时必需.
        out_dim:
            输出 token 维度.
        freeze_embedding:
            True 时基因 embedding 不更新 (冻结先验). v1 默认 True.
    """

    def __init__(
        self,
        mode: str = "gene_embedding",
        primary_embedding_path: Optional[Path] = None,
        secondary_embedding_path: Optional[Path] = None,
        out_dim: int = 512,
        freeze_embedding: bool = True,
    ):
        super().__init__()
        assert mode in ("gene_embedding", "concat_secondary"), f"unknown mode {mode}"
        assert primary_embedding_path is not None, (
            "primary_embedding_path is required; pass data/processed/primary_gene_embedding.pt"
        )

        self.mode = mode
        self.out_dim = out_dim

        primary_dict = self._load_embed(primary_embedding_path)
        assert primary_dict is not None, f"Failed to load primary embedding from {primary_embedding_path}"

        gene_symbols = sorted(primary_dict.keys())
        # index layout: 0 = control, 1 = unk, 2.. = genes (alpha order)
        self.gene_to_idx: Dict[str, int] = {g: i + 2 for i, g in enumerate(gene_symbols)}
        self.control_symbol = "control"

        primary_matrix = torch.stack([primary_dict[g] for g in gene_symbols]).float()
        p_dim = primary_matrix.shape[1]
        # control row = zeros, unk row = mean (so unk != control in L2).
        primary_full = torch.cat([
            torch.zeros(1, p_dim),
            primary_matrix.mean(0, keepdim=True),
            primary_matrix,
        ], dim=0)  # [V+2, p_dim]
        self.primary_embed = nn.Embedding.from_pretrained(
            primary_full, freeze=freeze_embedding, padding_idx=None
        )

        if mode == "concat_secondary":
            secondary_dict = self._load_embed(secondary_embedding_path)
            assert secondary_dict is not None, (
                "mode='concat_secondary' requires secondary_embedding_path"
            )
            # Align secondary to same row order; fallback to mean for missing genes.
            s_vecs = []
            missing = []
            for g in gene_symbols:
                v = secondary_dict.get(g)
                if v is None:
                    missing.append(g)
                    s_vecs.append(None)
                else:
                    s_vecs.append(v.float())
            s_dim = next(v for v in s_vecs if v is not None).shape[-1]
            s_stack = torch.stack([
                v if v is not None else torch.zeros(s_dim) for v in s_vecs
            ]).float()
            # Fill missing with observed-mean.
            if missing:
                observed = torch.stack([v for v in s_vecs if v is not None])
                mean_s = observed.mean(0)
                for i, v in enumerate(s_vecs):
                    if v is None:
                        s_stack[i] = mean_s
            secondary_full = torch.cat([
                torch.zeros(1, s_dim),
                s_stack.mean(0, keepdim=True),
                s_stack,
            ], dim=0)
            self.secondary_embed = nn.Embedding.from_pretrained(
                secondary_full, freeze=freeze_embedding, padding_idx=None
            )
            self.proj = nn.Sequential(
                nn.Linear(p_dim + s_dim, out_dim),
                nn.GELU(),
                nn.Linear(out_dim, out_dim),
            )
        else:
            self.secondary_embed = None
            if p_dim != out_dim:
                self.proj = nn.Linear(p_dim, out_dim)
            else:
                self.proj = nn.Identity()

        # Special-token vectors (post-projection) for control / unk are wired via
        # index lookup: control row is zeros -> after Linear it's `Linear.bias`.
        # That's fine semantically (a learnable "no-pert" bias) but makes control
        # != 0 after training. We keep the pre-projection zero and rely on the
        # unk row being clearly separated from control in embedding space.

    # ------------------------------------------------------------------ #
    # Static helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_embed(path: Optional[Path]) -> Optional[Dict[str, torch.Tensor]]:
        if path is None:
            return None
        path = Path(path)
        if not path.exists():
            return None
        obj = torch.load(path, map_location="cpu")
        assert isinstance(obj, dict), f"{path} must be dict[gene_symbol -> tensor]"
        return obj

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def _symbols_to_idx(self, gene_symbols: List[str]) -> torch.Tensor:
        idxs = []
        for g in gene_symbols:
            if g == self.control_symbol:
                idxs.append(CONTROL_IDX)
            else:
                idxs.append(self.gene_to_idx.get(g, UNK_IDX))
        return torch.as_tensor(idxs, dtype=torch.long, device=self.primary_embed.weight.device)

    def forward(self, gene_symbols: List[str]) -> torch.Tensor:
        """gene_symbols: list of len B -> z_pert: [B, out_dim]."""
        idx = self._symbols_to_idx(gene_symbols)
        p = self.primary_embed(idx)
        if self.mode == "gene_embedding":
            return self.proj(p)
        s = self.secondary_embed(idx)
        return self.proj(torch.cat([p, s], dim=-1))
