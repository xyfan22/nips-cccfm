"""Backward compatible entry-point for the core StateICL model."""

from __future__ import annotations

from .models.core import StateICLModel, scShiftAttentionModel
from .model_loading import load_model_from_checkpoint

__all__ = ["StateICLModel", "scShiftAttentionModel", "load_model_from_checkpoint"]
