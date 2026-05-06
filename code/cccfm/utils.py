"""通用工具: seed, io.

中文:
    - 目前只放 set_seed 和一些轻量 io.
    - 后续如果需要 distributed-aware 的 all_gather 等再扩.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """固定随机种子.
    中文: 覆盖 python / numpy / torch / cuda. Lightning 也会额外调一次.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_tensor(obj: Any, path: Path) -> None:
    """torch.save with parent dir 自动创建."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


def load_tensor(path: Path) -> Any:
    return torch.load(Path(path), map_location="cpu")


def is_rank_zero() -> bool:
    """DDP 下只在 rank 0 做某些副作用 (打印 / 写 csv)."""
    import torch.distributed as dist
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0
