"""ccCFM Lightning 训练模块.

中文:
    - 基于 torchcfm (v1.0.7, last commit 2026-04-20, 活跃维护) 做 CFM 路径采样.
    - OT pairing 我们手动做, 而不是直接用 torchcfm 的 ExactOptimalTransport*.
      原因: torchcfm 的 OT 会内部重排 (x_0, x_1), 但不会同步重排 h / z_pert,
      导致 conditioning 和 x_0 / x_1 失配.
    - 正确做法: 显式调 ot_sampler.sample_map 拿 (i, j) 索引, 对 (x_0, h) 用 i 重排,
      对 (x_1, z_pert) 用 j 重排, 然后用 vanilla ConditionalFlowMatcher 采路径.
    - 这等价于 Tong 2023 OT-CFM, 只是我们显式管理索引以保持 conditioning 对齐.

Batch 字段:
    x_0:    [B, G]     target 侧 control 细胞 (同 pert 对应 x_1 的同 line control)
    x_1:    [B, G]     perturbed 细胞 (同一 perturbation 跨 source lines 池化)
    h:      [B, D_ctx] 冻结 encoder 对 x_0 的 per-cell context
    z_pert: [B, D_pert] 扰动 token (与 x_1 的 perturbation label 对应)

参考:
    - torchcfm:  https://github.com/atong01/conditional-flow-matching
    - Tong et al. 2023: Improving and generalizing flow-based generative models
      with minibatch optimal transport.

注意:
    - 当前骨架假设 batch 内所有样本有相同/无关 perturbation label 对 OT 无影响.
      更严格的 per-pert-group OT (CellFlow 风格) 以后可加, 见 training_step 里 TODO.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl


class CCCFMLitModule(pl.LightningModule):
    """ccCFM Lightning module.

    中文参数:
        velocity_net:   VelocityNet 实例.
        use_ot_pairing: True 用 OT-CFM, False 用 vanilla CFM (独立配对).
        ot_method:      'exact' (精确 OT, 小 batch) 或 'sinkhorn' (近似, 大 batch).
        sigma:          路径 noise scale, 0.0 = 纯直线; 0.1 = 轻微扰动 (对噪声数据更鲁棒).
        lr / weight_decay: AdamW 参数.
        anchor:         可选 PooledAnchor, residual composition 推理时用, 训练不动.
    """

    def __init__(
        self,
        velocity_net: torch.nn.Module,
        pert_tokenizer: torch.nn.Module,
        use_ot_pairing: bool = True,
        ot_method: str = "exact",
        sigma: float = 0.0,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        anchor=None,
        # R2: residual mode -- training target = x_1 - (x_0 + delta_anchor(g))
        anchor_mode: str = "direct",
        anchor_delta_dict: dict[str, "np.ndarray"] | None = None,
        line_ctrl_means: dict[str, "np.ndarray"] | None = None,
        # R3.1: training target projection. 'raw' = match x_1 directly (R2 default).
        # 'mean_preserved' = project x_1 to anchor mean (advisor 2026-04-27 R3-MP):
        #   x1_mp = x_1 - x_1.mean(0) + ctx_mean + delta_anchor(g)
        # forces the flow to learn cell-state distribution, not mean-shift correction.
        anchor_target_mode: str = "raw",
        # R3-LowRank (advisor 2026-04-27 §2.1): 'gene' = full gene-space CFM (default).
        # 'lowrank' = run CFM in a rank-D PCA subspace defined by `lowrank_basis_B`:
        #   z0 = (x0_anchor - anchor_profile) @ B
        #   z1 = (x1_mp     - anchor_profile) @ B
        #   flow learns z0 -> z1 in rank-D space.
        # Inference unprojects + adds anchor profile + deterministic projection.
        # Requires anchor_mode='residual' + anchor_target_mode='mean_preserved'.
        flow_space: str = "gene",
        lowrank_basis_B: "np.ndarray | None" = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[
            "velocity_net", "pert_tokenizer", "anchor",
            "anchor_delta_dict", "line_ctrl_means",
        ])
        self.velocity_net = velocity_net
        self.pert_tokenizer = pert_tokenizer
        self.use_ot_pairing = use_ot_pairing
        assert anchor_mode in ("direct", "residual"), f"unknown anchor_mode {anchor_mode}"
        self.anchor_mode = anchor_mode
        assert anchor_target_mode in ("raw", "mean_preserved"), \
            f"unknown anchor_target_mode {anchor_target_mode}"
        self.anchor_target_mode = anchor_target_mode
        assert flow_space in ("gene", "lowrank"), f"unknown flow_space {flow_space}"
        self.flow_space = flow_space
        if flow_space == "lowrank":
            assert anchor_mode == "residual" and anchor_target_mode == "mean_preserved", (
                "flow_space='lowrank' requires residual + mean_preserved (advisor §2.1)"
            )
            assert lowrank_basis_B is not None, "flow_space='lowrank' needs lowrank_basis_B [G, rank]"

        # 延迟 import, 避免 torchcfm 未装时整个模块都 import 不了.
        from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
        # 始终用 vanilla 做 path 采样; OT 我们自己管.
        self.fm = ConditionalFlowMatcher(sigma=sigma)

        self.ot_sampler = None
        if use_ot_pairing:
            import warnings
            warnings.warn(
                "use_ot_pairing=True runs OT over the whole batch without any "
                "semantic constraint on cell line or perturbation grouping. "
                "For cross-line training, ensure the dataloader feeds "
                "per-line (or ideally per-(line, pert)) homogeneous batches; "
                "otherwise OT may silently pair a control cell with a "
                "perturbed cell from a different line/pert, corrupting "
                "training. See docs/problem_setup.md and data.py TODOs.",
                RuntimeWarning,
                stacklevel=2,
            )
            from torchcfm.optimal_transport import OTPlanSampler
            self.ot_sampler = OTPlanSampler(method=ot_method)

        self.anchor = anchor

        # Residual-mode buffers: stack delta(g) and line ctrl means as 2-D buffers
        # so they ride device transfers / DDP automatically. Lookup is by
        # python dict (anchor_pert_to_idx / line_to_idx).
        self.anchor_pert_to_idx: dict[str, int] = {}
        self.line_to_idx: dict[str, int] = {}
        if self.anchor_mode == "residual":
            assert anchor_delta_dict is not None and line_ctrl_means is not None, (
                "anchor_mode='residual' requires anchor_delta_dict + line_ctrl_means"
            )
            import numpy as np
            perts = sorted(anchor_delta_dict.keys())
            lines = sorted(line_ctrl_means.keys())
            self.anchor_pert_to_idx = {p: i for i, p in enumerate(perts)}
            self.line_to_idx = {l: i for i, l in enumerate(lines)}
            delta_mat = np.stack([anchor_delta_dict[p] for p in perts]).astype(np.float32)
            line_mat = np.stack([line_ctrl_means[l] for l in lines]).astype(np.float32)
            self.register_buffer("anchor_delta_matrix", torch.from_numpy(delta_mat))
            self.register_buffer("line_ctrl_mean_matrix", torch.from_numpy(line_mat))
        else:
            # placeholder buffers so .to(device) / state_dict don't break
            self.register_buffer("anchor_delta_matrix", torch.zeros(1, 1))
            self.register_buffer("line_ctrl_mean_matrix", torch.zeros(1, 1))

        # Low-rank basis buffer (R3-LowRank). Shape [G, rank] when flow_space='lowrank',
        # else placeholder. Storage with register_buffer ensures DDP/.to(device) work.
        if flow_space == "lowrank":
            B_arr = np.asarray(lowrank_basis_B, dtype=np.float32)
            assert B_arr.ndim == 2, f"lowrank_basis_B must be 2-D, got {B_arr.shape}"
            self.register_buffer("lowrank_B", torch.from_numpy(B_arr))
            self.lowrank_rank = int(B_arr.shape[1])
            self.lowrank_gene_dim = int(B_arr.shape[0])
        else:
            self.register_buffer("lowrank_B", torch.zeros(1, 1))
            self.lowrank_rank = None
            self.lowrank_gene_dim = None

    # ------------------------------------------------------------------ #
    # Batch post-processing: tokenize perts on-device
    # ------------------------------------------------------------------ #
    def on_after_batch_transfer(self, batch, dataloader_idx: int = 0):
        # DataLoader collate 返回 'perts': list[str]; 在这里把它查表成 [B, pert_dim]
        # 且 tokenizer 的 embedding 已随 module.to(device) 移动, DDP 下也正确.
        if not isinstance(batch, dict):
            return batch
        if "perts" in batch and "z_pert" not in batch:
            batch["z_pert"] = self.pert_tokenizer(batch["perts"])

        # R2 residual mode: build x0_anchor on device. Requires the BatchSampler
        # to have produced a (line, pert)-homogeneous batch (R2.2 correctness
        # condition); we assert this and then look up the canonical (line, pert).
        if self.anchor_mode == "residual" and "x0_anchor" not in batch:
            from cccfm.anchor import apply_anchor_torch
            perts = batch["perts"]; lines = batch["lines"]
            pert0, line0 = perts[0], lines[0]
            # cheap homogeneity check (helpful when GroupBatchSampler is wrong)
            if any(p != pert0 for p in perts) or any(l != line0 for l in lines):
                raise RuntimeError(
                    f"residual mode requires (line, pert)-homogeneous batches; "
                    f"got perts={set(perts)} lines={set(lines)}. "
                    f"Did you replace the DataLoader's batch_sampler with GroupBatchSampler?"
                )
            if pert0 not in self.anchor_pert_to_idx:
                raise KeyError(
                    f"pert '{pert0}' not in anchor_delta_dict (size "
                    f"{len(self.anchor_pert_to_idx)}). Filter the dataset to "
                    f"anchor.supported_perts before training."
                )
            if line0 not in self.line_to_idx:
                raise KeyError(f"line '{line0}' not in line_ctrl_means")
            delta_g = self.anchor_delta_matrix[self.anchor_pert_to_idx[pert0]]   # [G]
            ctx_mean = self.line_ctrl_mean_matrix[self.line_to_idx[line0]]       # [G]
            batch["x0_anchor"] = apply_anchor_torch(batch["x_0"], ctx_mean, delta_g)
            # R3.1 mean-preserved target: project x_1 to the same anchor mean,
            # forcing the flow to learn centered cell-state distribution rather
            # than mean shift. (advisor 2026-04-27)
            if self.anchor_target_mode == "mean_preserved":
                batch["x1_mp"] = apply_anchor_torch(batch["x_1"], ctx_mean, delta_g)
            # R3-LowRank: project endpoints into rank-D PCA subspace.
            # anchor_profile = ctx_mean + delta_g  (the per-(line,pert) anchor).
            # z0 = (x0_anchor - anchor_profile) @ B   has batch mean ≈ 0 (centered ctrl shape).
            # z1 = (x1_mp     - anchor_profile) @ B   has batch mean ≈ 0 (centered pert shape).
            if self.flow_space == "lowrank":
                anchor_profile = ctx_mean + delta_g     # [G]
                batch["anchor_profile"] = anchor_profile
                B = self.lowrank_B                       # [G, rank]
                batch["z0"] = (batch["x0_anchor"] - anchor_profile) @ B
                if "x1_mp" in batch:
                    batch["z1"] = (batch["x1_mp"] - anchor_profile) @ B
        return batch

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss = self._cfm_loss(batch)
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss = self._cfm_loss(batch)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        # 中文: 真实的 distribution-level val (MMD / energy) 交给 on_validation_epoch_end
        #       的 Callback 去做 (见 scripts/train.py), 否则 per-batch MMD 噪声太大.
        return loss

    def _cfm_loss(self, batch: dict) -> torch.Tensor:
        # R3-LowRank: CFM lives in z-space (rank-D); velocity_net is dimensioned
        # for rank input/output. Source = z0, target = z1 — both already in batch.
        if self.flow_space == "lowrank":
            source = batch["z0"]
            target = batch["z1"]
        elif self.anchor_mode == "residual":
            source = batch["x0_anchor"]
            # R3.1 mean-preserved target if set
            target = batch["x1_mp"] if self.anchor_target_mode == "mean_preserved" else batch["x_1"]
        else:
            source = batch["x_0"]
            target = batch["x_1"]

        h = batch["h"]
        z_pert = batch["z_pert"]

        if self.use_ot_pairing:
            source, target, h, z_pert = self._ot_reorder(source, target, h, z_pert)

        t, x_t, u_t = self.fm.sample_location_and_conditional_flow(source, target)
        v_pred = self.velocity_net(x_t, t, h, z_pert)
        return F.mse_loss(v_pred, u_t)

    def _ot_reorder(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        h: torch.Tensor,
        z_pert: torch.Tensor,
    ):
        """按 OT plan 重排 (x_0, h) 和 (x_1, z_pert), 保持各自的 conditioning 对齐.

        中文:
            - ot_sampler.get_map(x_0, x_1): 返回 [B, B] OT plan 矩阵.
            - ot_sampler.sample_map(pi, B): 从 plan 按概率抽样得到索引 (i, j),
              其中 x_0[i] 与 x_1[j] 构成 OT-matched pair.
            - 我们同步用 i 重排 h (因为 h 是 Encoder(x_0) 得到, 和 x_0 一一对应),
              用 j 重排 z_pert (因为 z_pert 是 x_1 的 perturbation token).
        """
        pi = self.ot_sampler.get_map(x_0, x_1)
        i, j = self.ot_sampler.sample_map(pi, x_0.shape[0], replace=True)
        return x_0[i], x_1[j], h[i], z_pert[j]

    # ------------------------------------------------------------------ #
    # Optimizer
    # ------------------------------------------------------------------ #
    def configure_optimizers(self):
        # 用 self.parameters() 确保 pert_tokenizer (及未来新加的 submodule) 的
        # trainable 参数都被 AdamW 管理; frozen embedding 因 requires_grad=False
        # 自动排除. 历史 bug (2026-04-23): 早期只传 velocity_net.parameters(),
        # 漏了 pert_tokenizer.proj 的 66k 参数, 等同 proj 冻结在随机初始化.
        opt = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=self.trainer.max_epochs if self.trainer else 50,
        )
        return {"optimizer": opt, "lr_scheduler": sched}

    def on_fit_start(self) -> None:
        """Regression guard: ensure optimizer covers every trainable param.

        Catches future "forgot a submodule" bugs at fit start, not 5 hours
        into eval. See bug.md (2026-04-23).
        """
        trainable = {id(p) for p in self.parameters() if p.requires_grad}
        in_opt = {id(p) for g in self.optimizers().param_groups for p in g["params"]}
        missing = trainable - in_opt
        if missing:
            missing_names = [n for n, p in self.named_parameters()
                             if p.requires_grad and id(p) in missing]
            raise RuntimeError(
                f"optimizer is missing {len(missing)} trainable param tensor(s): "
                f"{missing_names}. See configure_optimizers()."
            )

    # ------------------------------------------------------------------ #
    # Inference helper (residual composition)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_perturbed(
        self,
        x_0: torch.Tensor,
        h: torch.Tensor,
        z_pert: torch.Tensor,
        n_steps: int = 50,
        method: str = "heun",
        anchor_pb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """从 control cells 采样预测 perturbed cells.

        中文:
            - ODE 积分 x_0 -> x_1 (method ∈ {euler, heun, dopri5}).
            - anchor_pb: pseudobulk-delta anchor 预测 [G], 评估时用于 residual composition,
              不影响 cell-level samples 本身.
            - 返回 [N, G] 预测 perturbed cell samples.
        """
        from cccfm.sampling import sample_euler, sample_heun, sample_dopri5
        if method == "euler":
            return sample_euler(self.velocity_net, x_0, h, z_pert, n_steps=n_steps)
        elif method == "heun":
            return sample_heun(self.velocity_net, x_0, h, z_pert, n_steps=n_steps)
        elif method == "dopri5":
            return sample_dopri5(self.velocity_net, x_0, h, z_pert)
        else:
            raise ValueError(f"Unknown method: {method}")

    @torch.no_grad()
    def predict_for_pert(
        self,
        x_0: torch.Tensor,
        h: torch.Tensor,
        pert_symbol: str,
        n_steps: int = 50,
        method: str = "heun",
    ) -> torch.Tensor:
        """广播 pert_symbol 到 batch, tokenize, 再跑 ODE 积分.

        Direct mode: starts ODE at x_0 (raw target controls).
        Residual mode: starts ODE at x0_anchor (mean-centred + delta_g shift).
                       Requires `target_line` to know which ctrl mean to use —
                       use `predict_for_pert_anchored` instead.
        """
        if self.anchor_mode == "residual":
            raise RuntimeError(
                "residual mode requires `predict_for_pert_anchored(..., target_line=...)`. "
                "Bare predict_for_pert can't pick the right context_ctrl_mean."
            )
        B = x_0.shape[0]
        z_pert = self.pert_tokenizer([pert_symbol] * B)
        return self.predict_perturbed(x_0, h, z_pert, n_steps=n_steps, method=method)

    @torch.no_grad()
    def predict_for_pert_anchored(
        self,
        x_0: torch.Tensor,
        h: torch.Tensor,
        pert_symbol: str,
        target_line: str,
        n_steps: int = 50,
        method: str = "heun",
    ) -> torch.Tensor:
        """Residual-mode inference: ODE starts at x0_anchor, not raw x_0.

        x0_anchor = (x_0 - x_0.mean(0)) + ctx_mean[target_line] + delta_g
        Then run velocity field ODE x0_anchor -> predicted perturbed cells.

        For flow_space='lowrank' runs, the ODE is in z-space; we project z0,
        integrate, then unproject and add anchor profile, with the deterministic
        post-hoc mean projection (advisor §6.4) as a safety net.
        """
        from cccfm.anchor import apply_anchor_torch
        if self.anchor_mode != "residual":
            raise RuntimeError("predict_for_pert_anchored only valid for anchor_mode='residual'")
        if pert_symbol not in self.anchor_pert_to_idx:
            raise KeyError(f"anchor has no delta for pert '{pert_symbol}'")
        if target_line not in self.line_to_idx:
            raise KeyError(f"no line_ctrl_mean for '{target_line}'")
        delta_g = self.anchor_delta_matrix[self.anchor_pert_to_idx[pert_symbol]]
        ctx_mean = self.line_ctrl_mean_matrix[self.line_to_idx[target_line]]
        x0_anchor = apply_anchor_torch(x_0, ctx_mean, delta_g)
        Bsz = x_0.shape[0]
        z_pert = self.pert_tokenizer([pert_symbol] * Bsz)

        if self.flow_space == "lowrank":
            anchor_profile = ctx_mean + delta_g                 # [G]
            z0 = (x0_anchor - anchor_profile) @ self.lowrank_B  # [B, rank]
            z_pred = self.predict_perturbed(z0, h, z_pert, n_steps=n_steps, method=method)
            # Off-basis passthrough (interpretation of advisor §2.1):
            # pred = x0_anchor + (z_pred - z0) @ B.T
            # = "flow only edits the low-rank component; off-basis natural cell variance
            #    flows through from the input control cell unchanged"
            # At zero-init z_pred=z0 → pred=x0_anchor (matches anchor exactly).
            # Strict §2.1 equation (pred = z_pred @ B.T + anchor) gives a rank-D-thin
            # distribution that lacks natural per-gene variance, exploding Energy/Moment.
            delta_z = z_pred - z0                               # [B, rank]
            pred = x0_anchor + delta_z @ self.lowrank_B.T       # [B, G]
            # Safety mean projection (§6.4): in practice near-identity since the
            # passthrough already preserves x0_anchor's mean = anchor_profile.
            pred = pred - pred.mean(dim=0, keepdim=True) + anchor_profile
            return pred

        return self.predict_perturbed(x0_anchor, h, z_pert, n_steps=n_steps, method=method)
