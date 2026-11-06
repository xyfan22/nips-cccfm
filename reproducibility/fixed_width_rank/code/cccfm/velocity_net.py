"""ccCFM velocity 网络 v_θ(x_t, t, h, z_pert).

中文:
    - 输入:
        x_t    ∈ R^G        (t 时刻的 cell state, G=6836)
        t      ∈ [0, 1]
        h      ∈ R^{D_ctx}   (冻结 encoder 的 per-cell context)
        z_pert ∈ R^{D_pert}  (扰动 token)
    - 输出: velocity v ∈ R^G (同 x_t 空间)
    - 结构: 简单 residual MLP. 6836 维已经不低, 不急着上 transformer.
    - conditioning: 默认 concat (h, z_pert, t_emb) 投影到 hidden, 加到 x_t 的 projection 上.
      后续可加 FiLM ablation (见 config.model.cond_mode).

设计权衡:
    - 维持 gene 空间直接输入输出, 简单可靠.
    - 不强行上 transformer 的原因: gene-wise attention 对小数据容易过拟合.
    - 如果 STATE-SE 的 pretrained structure 有帮助, 它已经被 h 带进来了.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """t ∈ [0, 1] 的 sinusoidal embedding (常见 diffusion/FM 用法)."""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "time_dim must be even"
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] -> [B, dim]
        device = t.device
        half = self.dim // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        freqs = torch.exp(-scale * torch.arange(half, device=device, dtype=torch.float32))
        ang = t[:, None].float() * freqs[None, :]   # [B, half]
        return torch.cat([ang.sin(), ang.cos()], dim=-1)  # [B, dim]


class ResidualBlock(nn.Module):
    """pre-norm MLP residual block."""

    def __init__(self, dim: int, hidden_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        hidden = hidden_dim or dim
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class VelocityNet(nn.Module):
    """ccCFM velocity field 主网络.

    中文参数:
        gene_dim:    G, 默认 6836 (3-way HVG).
        context_dim: D_ctx, encoder 输出维度 (因 encoder 而异, 见 build_encoder).
        pert_dim:    D_pert, PertTokenizer 的 out_dim.
        hidden_dim:  内部宽度.
        n_blocks:    residual block 数.
        time_dim:    sinusoidal 时间 embedding 维度.
        dropout:     默认 0, 大模型过拟合时开.
    """

    def __init__(
        self,
        gene_dim: int = 6836,
        context_dim: int = 2048,
        pert_dim: int = 512,
        hidden_dim: int = 1024,
        n_blocks: int = 4,
        time_dim: int = 128,
        dropout: float = 0.0,
        zero_init_output: bool = False,
    ):
        super().__init__()
        self.gene_dim = gene_dim
        self.context_dim = context_dim
        self.pert_dim = pert_dim
        self.hidden_dim = hidden_dim

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

        cond_dim = context_dim + pert_dim + time_dim

        self.in_proj = nn.Linear(gene_dim, hidden_dim)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=dropout) for _ in range(n_blocks)
        ])

        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, gene_dim),
        )

        if zero_init_output:
            # R2.3: at residual-mode init, force v_theta(x_t, t, h, z_g) ≡ 0.
            # Then ODE integration from x0_anchor produces x0_anchor exactly,
            # i.e. the model starts at the shifted-control baseline. Training
            # then learns the residual on top, so anchor-level perturbation
            # metrics are preserved through the first gradient step.
            with torch.no_grad():
                self.out_proj[-1].weight.zero_()
                if self.out_proj[-1].bias is not None:
                    self.out_proj[-1].bias.zero_()

    def forward(
        self,
        x_t: torch.Tensor,     # [B, G]
        t: torch.Tensor,       # [B]
        h: torch.Tensor,       # [B, D_ctx]
        z_pert: torch.Tensor,  # [B, D_pert]
    ) -> torch.Tensor:
        # 时间 embedding
        t_emb = self.time_embed(t)                       # [B, time_dim]
        cond = torch.cat([h, z_pert, t_emb], dim=-1)     # [B, cond_dim]
        x_h = self.in_proj(x_t) + self.cond_proj(cond)   # [B, hidden]
        for block in self.blocks:
            x_h = block(x_h)
        return self.out_proj(x_h)                        # [B, G]
