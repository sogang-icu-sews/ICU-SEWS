"""Shared YAML configuration loader.

Owner: 공동
Work: Keep file paths and experiment settings outside business logic.
"""

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml_config(name: str) -> dict[str, Any]:
    """Load one YAML file from the project ``configs`` directory."""
    config_path = PROJECT_ROOT / "configs" / name
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    return config
