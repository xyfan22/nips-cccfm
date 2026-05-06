"""Core StateICL model and the reusable building blocks it is composed of."""

from .base import StateICLModelBase
from .inference import InferenceMixin
from .losses import LossComputationMixin


class StateICLModel(InferenceMixin, LossComputationMixin, StateICLModelBase):
    """Full StateICL model combining architecture, inference, and loss logic."""


scShiftAttentionModel = StateICLModel

__all__ = [
    "StateICLModelBase",
    "InferenceMixin",
    "LossComputationMixin",
    "StateICLModel",
    "scShiftAttentionModel",
]
