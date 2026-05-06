"""Smoke test: 确保 VelocityNet forward 形状正确, CFM loss 能计算.

中文:
    - 不跑真实数据, 用 random tensor 验 shape + 无报错.
    - 装完 torchcfm 后 `pytest tests/` 应 1 秒内通过.
"""

import torch


def test_velocity_net_forward_shape():
    from cccfm.velocity_net import VelocityNet

    B, G, D_ctx, D_pert = 8, 256, 64, 32
    net = VelocityNet(
        gene_dim=G,
        context_dim=D_ctx,
        pert_dim=D_pert,
        hidden_dim=128,
        n_blocks=2,
        time_dim=32,
    )
    x_t = torch.randn(B, G)
    t = torch.rand(B)
    h = torch.randn(B, D_ctx)
    z = torch.randn(B, D_pert)
    v = net(x_t, t, h, z)
    assert v.shape == (B, G)


def test_cfm_loss_runs():
    """确认 torchcfm 装了且 flow_module 能跑一个 training step."""
    try:
        from torchcfm.conditional_flow_matching import (
            ExactOptimalTransportConditionalFlowMatcher,
        )
    except ImportError:
        import pytest
        pytest.skip("torchcfm not installed")

    import tempfile

    from cccfm.flow_module import CCCFMLitModule
    from cccfm.pert_tokens import PertTokenizer
    from cccfm.velocity_net import VelocityNet

    B, G, D_ctx, D_pert = 8, 256, 64, 32
    net = VelocityNet(
        gene_dim=G, context_dim=D_ctx, pert_dim=D_pert,
        hidden_dim=128, n_blocks=2, time_dim=32,
    )
    # PertTokenizer needs a primary embedding file; synthesize a tiny one for the smoke.
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save({"GENE_A": torch.randn(D_pert), "GENE_B": torch.randn(D_pert)}, f.name)
        tok = PertTokenizer(
            mode="gene_embedding",
            primary_embedding_path=f.name,
            out_dim=D_pert,
            freeze_embedding=True,
        )

    module = CCCFMLitModule(
        velocity_net=net, pert_tokenizer=tok,
        use_ot_pairing=False, sigma=0.0,
    )
    batch = {
        "x_0": torch.randn(B, G),
        "x_1": torch.randn(B, G),
        "h": torch.randn(B, D_ctx),
        "z_pert": torch.randn(B, D_pert),
    }
    loss = module._cfm_loss(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_apply_anchor_torch_matches_numpy():
    """R2: apply_anchor_torch must agree with numpy version.

    Mean invariant:  out.mean(0) == context_ctrl_mean + delta_g (up to float prec).
    Numerical agreement: torch and numpy outputs within ~1e-6 elementwise.
    """
    import numpy as np
    import torch
    from cccfm.anchor import apply_anchor, apply_anchor_torch

    rng = np.random.default_rng(0)
    G, N = 200, 64
    x_np = rng.standard_normal((N, G)).astype(np.float32)
    ctx_np = rng.standard_normal(G).astype(np.float32)
    delta_np = rng.standard_normal(G).astype(np.float32) * 0.1

    out_np = apply_anchor(x_np, ctx_np, delta_np)

    x_t = torch.from_numpy(x_np)
    ctx_t = torch.from_numpy(ctx_np)
    delta_t = torch.from_numpy(delta_np)
    out_t = apply_anchor_torch(x_t, ctx_t, delta_t).numpy()

    assert out_np.shape == (N, G) == out_t.shape

    # mean invariant on both
    target_mean = ctx_np + delta_np
    assert np.allclose(out_np.mean(0), target_mean, atol=1e-5), \
        f"numpy mean invariant: max|diff|={np.abs(out_np.mean(0) - target_mean).max():.2e}"
    assert np.allclose(out_t.mean(0), target_mean, atol=1e-5), \
        f"torch mean invariant: max|diff|={np.abs(out_t.mean(0) - target_mean).max():.2e}"

    # torch vs numpy agreement (per-cell)
    assert np.allclose(out_np, out_t, atol=1e-6), \
        f"torch≠numpy: max|diff|={np.abs(out_np - out_t).max():.2e}"

    # type-check rejection of numpy input to torch helper
    try:
        apply_anchor_torch(x_np, ctx_t, delta_t)
        assert False, "should have raised TypeError on numpy input"
    except TypeError:
        pass


def test_optimizer_covers_all_trainable_params():
    """Regression guard: configure_optimizers must include pert_tokenizer.proj.

    See bug.md (2026-04-23) — early version only passed `velocity_net.parameters()`,
    silently freezing 66k pert_tokenizer params at random init.
    """
    import tempfile

    from cccfm.flow_module import CCCFMLitModule
    from cccfm.pert_tokens import PertTokenizer
    from cccfm.velocity_net import VelocityNet

    B, G, D_ctx, D_pert = 8, 256, 64, 32
    net = VelocityNet(
        gene_dim=G, context_dim=D_ctx, pert_dim=D_pert,
        hidden_dim=128, n_blocks=2, time_dim=32,
    )
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save({"GENE_A": torch.randn(D_pert)}, f.name)
        tok = PertTokenizer(mode="gene_embedding", primary_embedding_path=f.name,
                            out_dim=D_pert, freeze_embedding=True)
    lit = CCCFMLitModule(velocity_net=net, pert_tokenizer=tok)

    params = [p for p in lit.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4)
    n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    n_trainable = sum(p.numel() for p in lit.parameters() if p.requires_grad)
    assert n_opt == n_trainable, (
        f"optimizer covers {n_opt} params but module has {n_trainable} trainable; "
        "configure_optimizers is missing some submodules."
    )
