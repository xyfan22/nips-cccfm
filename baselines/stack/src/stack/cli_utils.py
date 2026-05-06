"""Utilities shared by the project command line entry points."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

logger = logging.getLogger(__name__)


class ConfigFileError(RuntimeError):
    """Raised when a configuration file cannot be parsed."""


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ConfigFileError(
            "PyYAML is required to parse YAML configuration files. "
            "Install it with `pip install pyyaml` or use a JSON config instead."
        ) from exc

    data = yaml.safe_load(path.read_text())
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigFileError(f"Expected a mapping at the top level of {path}, got {type(data)!r}.")
    return data


def load_config_file(path: str | Path) -> Dict[str, Any]:
    """Load a JSON or YAML configuration file and return its content."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise ConfigFileError(f"Configuration file '{config_path}' does not exist.")

    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return _load_yaml(config_path)
    if suffix == ".json":
        data = json.loads(config_path.read_text())
        if not isinstance(data, dict):
            raise ConfigFileError(
                f"Expected a mapping at the top level of {config_path}, got {type(data)!r}."
            )
        return data

    raise ConfigFileError(
        f"Unsupported configuration extension '{config_path.suffix}'. "
        "Only .json, .yaml and .yml files are supported."
    )


def apply_config(parser, config: Dict[str, Any]) -> Tuple[Dict[str, Any], Tuple[str, ...]]:
    """Apply defaults from ``config`` to ``parser`` and return recognised keys.

    Parameters
    ----------
    parser:
        The ``argparse.ArgumentParser`` whose defaults should be updated.
    config:
        A dictionary coming from :func:`load_config_file`.

    Returns
    -------
    recognised:
        The subset of key/value pairs that were applied to the parser.
    unused:
        Keys that were not recognised by the parser.
    """

    valid_dests = {
        action.dest
        for action in parser._actions  # type: ignore[attr-defined]
        if action.dest not in {"help", "--help", "==SUPPRESS=="}
    }

    recognised: Dict[str, Any] = {k: v for k, v in config.items() if k in valid_dests}
    if recognised:
        parser.set_defaults(**recognised)

    unused = tuple(sorted(k for k in config.keys() if k not in recognised))
    if unused:
        logger.warning("Unrecognised configuration keys: %s", ", ".join(unused))

    return recognised, unused


def apply_config_from_file(parser, path: str | Path | None) -> Tuple[Dict[str, Any], Tuple[str, ...]]:
    """Convenience wrapper that loads ``path`` and calls :func:`apply_config`."""

    if not path:
        return {}, tuple()

    config = load_config_file(path)
    return apply_config(parser, config)


def filter_unused_arguments(namespace, required: Iterable[str], parser) -> None:
    """Validate required fields after config/CLI merging."""

    missing = [name for name in required if getattr(namespace, name) in (None, [], "")]
    if missing:
        parser.error(
            "Missing required arguments: "
            + ", ".join(f"--{name.replace('_', '-')} (or set via config)" for name in missing)
        )
