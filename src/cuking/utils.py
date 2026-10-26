"""Utility functions for reproducibility, I/O, and environment configuration."""

import json
import logging
import os
from pathlib import Path
import random
from typing import Any, Dict, Optional, Union
import numpy as np
import torch
import yaml

UnionPath = Union[str, Path]


def setup_logger(name: str = "cuking", level: int = logging.INFO) -> logging.Logger:
    """Create and configure standard application logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_hf_token() -> Optional[str]:
    """Retrieve Hugging Face token from environment variable."""
    return os.environ.get("HF_TOKEN")


def load_yaml(file_path: UnionPath) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: Dict[str, Any], file_path: UnionPath) -> None:
    """Save configuration dictionary to a YAML file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def load_json(file_path: UnionPath) -> Dict[str, Any]:
    """Load data from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: UnionPath, indent: int = 2) -> None:
    """Save data dictionary to a JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

