import argparse
import json
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from stack.cli_utils import (
    ConfigFileError,
    apply_config,
    filter_unused_arguments,
    load_config_file,
)


def test_load_config_file_json(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"learning_rate": 1e-4, "batch_size": 64}))

    data = load_config_file(config_path)

    assert data["learning_rate"] == pytest.approx(1e-4)
    assert data["batch_size"] == 64


def test_apply_config_updates_defaults():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=int, default=5)

    recognised, unused = apply_config(parser, {"alpha": 0.25, "gamma": "ignored"})
    args = parser.parse_args([])

    assert recognised == {"alpha": 0.25}
    assert unused == ("gamma",)
    assert args.alpha == pytest.approx(0.25)
    assert args.beta == 5


def test_filter_unused_arguments_detects_missing_required():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=str)
    args = parser.parse_args([])

    with pytest.raises(SystemExit):
        filter_unused_arguments(args, ("alpha",), parser)


def test_load_config_file_rejects_unknown_extension(tmp_path):
    config_path = tmp_path / "config.txt"
    config_path.write_text("alpha=1")

    with pytest.raises(ConfigFileError):
        load_config_file(config_path)


def test_load_config_file_missing(tmp_path):
    with pytest.raises(ConfigFileError):
        load_config_file(tmp_path / "missing.yaml")
