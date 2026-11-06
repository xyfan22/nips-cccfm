"""ODE 积分推理.

中文:
    - 给定 control cells, 预测 perturbed cells 分布.
    - Euler: 最快, 默认用.
    - Heun (RK2): 略准, 计算约 2x.
    - DOPRI-5: 自适应, 需要 torchdiffeq, 更准但慢.
    - 用途: 输出 cell-level samples, 喂给 eval_metrics.full_evaluate.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def sample_euler(
    velocity_net,
    x_0: torch.Tensor,       # [N, G]
    h: torch.Tensor,         # [N, D_ctx]
    z_pert: torch.Tensor,    # [N, D_pert]
    n_steps: int = 50,
) -> torch.Tensor:
    """Euler 积分 t: 0 -> 1."""
    velocity_net.eval()
    x = x_0.clone()
    dt = 1.0 / n_steps
    for step in range(n_steps):
        t = torch.full((x.shape[0],), step * dt, device=x.device, dtype=x.dtype)
        v = velocity_net(x, t, h, z_pert)
        x = x + dt * v
    return x


@torch.no_grad()
def sample_heun(
    velocity_net,
    x_0: torch.Tensor,
    h: torch.Tensor,
    z_pert: torch.Tensor,
    n_steps: int = 50,
) -> torch.Tensor:
    """Heun (RK2)."""
    velocity_net.eval()
    x = x_0.clone()
    dt = 1.0 / n_steps
    for step in range(n_steps):
        t = torch.full((x.shape[0],), step * dt, device=x.device, dtype=x.dtype)
        t_next = torch.full((x.shape[0],), (step + 1) * dt, device=x.device, dtype=x.dtype)
        v1 = velocity_net(x, t, h, z_pert)
        x_pred = x + dt * v1
        v2 = velocity_net(x_pred, t_next, h, z_pert)
        x = x + 0.5 * dt * (v1 + v2)
    return x


@torch.no_grad()
def sample_dopri5(
    velocity_net,
    x_0: torch.Tensor,
    h: torch.Tensor,
    z_pert: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> torch.Tensor:
    """DOPRI-5 自适应. 需 torchdiffeq."""
    try:
        from torchdiffeq import odeint
    except ImportError as e:
        raise RuntimeError("pip install torchdiffeq for DOPRI-5") from e

    velocity_net.eval()

    def ode_fn(t, x):
        t_batch = t.expand(x.shape[0])
        return velocity_net(x, t_batch, h, z_pert)

    t_span = torch.tensor([0.0, 1.0], device=x_0.device, dtype=x_0.dtype)
    traj = odeint(ode_fn, x_0, t_span, rtol=rtol, atol=atol, method="dopri5")
    return traj[-1]
