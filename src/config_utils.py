\
from __future__ import annotations

from pathlib import Path
import os
import random
import yaml
import numpy as np


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


def ensure_project_dirs() -> None:
    for p in ["data/raw", "data/processed", "models", "results"]:
        Path(p).mkdir(parents=True, exist_ok=True)
