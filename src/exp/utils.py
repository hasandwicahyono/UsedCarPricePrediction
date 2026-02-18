from __future__ import annotations

import hashlib
import os
import random
import re
from typing import Iterable

import numpy as np
from contextlib import contextmanager

_MILEAGE_RE = re.compile(r"mileage|odometer|km", re.I)
_YEAR_RE = re.compile(r"year", re.I)
_ENGINE_RE = re.compile(r"engine|displacement", re.I)


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


@contextmanager
def seed_context(seed: int):
    set_seed(seed)
    try:
        yield
    finally:
        pass


def model_label(name: str, residual_cfg: dict | None) -> str:
    if residual_cfg is None:
        return name
    return f"{name}+{residual_cfg['kind']}"


def build_monotone_constraints(feature_names: Iterable[str]) -> str:
    constraints = []
    for f in feature_names:
        if _MILEAGE_RE.search(f):
            constraints.append(-1)
        elif _YEAR_RE.search(f):
            constraints.append(+1)
        elif _ENGINE_RE.search(f):
            constraints.append(+1)
        else:
            constraints.append(0)
    return "(" + ",".join(map(str, constraints)) + ")"


def make_pre_cache_key(model_name: str, seed: int, indices: np.ndarray) -> tuple:
    digest = hashlib.blake2b(indices.tobytes(), digest_size=16).digest()
    return (model_name, seed, digest)
