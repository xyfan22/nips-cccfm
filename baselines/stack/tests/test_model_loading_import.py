"""Ensure the public checkpoint loader can be imported from multiple entry-points."""

import importlib

import pytest


pytest.importorskip("torch", reason="torch is required for checkpoint loading")


def test_model_loading_importable_from_package() -> None:
    module = importlib.import_module("stack.model")
    assert hasattr(module, "load_model_from_checkpoint")


def test_model_loading_importable_from_public_module() -> None:
    module = importlib.import_module("stack.model_loading")
    assert hasattr(module, "load_model_from_checkpoint")


