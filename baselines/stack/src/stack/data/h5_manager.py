"""Utilities for managing shared HDF5 file handles and worker initialization."""
from __future__ import annotations

import atexit
import random
from dataclasses import dataclass
from typing import Dict

import h5py
import numpy as np
import torch


@dataclass(slots=True)
class H5HandleManager:
    """Central registry that keeps reusable read-only :mod:`h5py` handles."""

    _handles: Dict[str, h5py.File] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._handles is None:
            self._handles = {}

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def get(self, path: str) -> h5py.File:
        """Return a cached handle for *path* creating it when missing."""
        handle = self._handles.get(path)
        if handle is None or not handle.id.valid:
            handle = h5py.File(
                path,
                "r",
                libver="latest",
                rdcc_nbytes=1024 * 1024 * 1024,
                rdcc_nslots=1_000_003,
                rdcc_w0=0.75,
            )
            self._handles[path] = handle
        return handle

    def close_all(self) -> None:
        """Close every cached handle and clear the registry."""
        for handle in list(self._handles.values()):
            try:
                handle.close()
            finally:
                pass
        self._handles.clear()

    # ------------------------------------------------------------------
    # PyTorch DataLoader integration
    # ------------------------------------------------------------------
    def worker_init(self, worker_id: int) -> None:
        """Reset state and seed RNGs when a new worker process starts."""
        self.close_all()
        np.random.seed(worker_id)
        random.seed(worker_id)
        torch.manual_seed(worker_id)


# Singleton exposed for module level functions
_HANDLE_MANAGER = H5HandleManager()


def get_h5_handle(path: str) -> h5py.File:
    """Public helper mirroring the legacy :func:`_get_h5_handle`."""
    return _HANDLE_MANAGER.get(path)


def reset_h5_handle_pool() -> None:
    """Explicitly close and clear cached handles."""
    _HANDLE_MANAGER.close_all()


def worker_init_fn(worker_id: int) -> None:
    """Entry point that matches the signature expected by :class:`DataLoader`."""
    _HANDLE_MANAGER.worker_init(worker_id)


def _worker_init_fn(worker_id: int) -> None:
    """Legacy alias preserved for backwards compatibility."""
    worker_init_fn(worker_id)


# Backwards compatibility aliases ---------------------------------------------------------------
_get_h5_handle = get_h5_handle
_reset_h5_handle_pool = reset_h5_handle_pool


@atexit.register
def _close_handles_on_exit() -> None:
    _HANDLE_MANAGER.close_all()
