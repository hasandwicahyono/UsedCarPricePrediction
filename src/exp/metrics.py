from __future__ import annotations
# /Users/macbook/Library/CloudStorage/GoogleDrive-nur.ichsan@gmail.com/My Drive/UNS/2023/Business Intelligence/Paper/Salomo/NewIJAI/src/exp/metrics.py
# NOTE: Metric classes below are registered and instantiated via the metric registry/factory.

from abc import abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Protocol
import numpy as np

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
)
from .registry import METRIC_REGISTRY

Direction = Literal["minimize", "maximize"]

def register_metric(name: str):
    def _wrap(cls):
        METRIC_REGISTRY.register(name, cls)
        return cls
    return _wrap


class MetricStrategy(Protocol):
    name: str
    direction: Direction

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float: ...
    def supports_pruning(self) -> bool: ...
    def is_better(self, new: float, best: float) -> bool: ...
    def as_loss(self, score: float) -> float: ...


@dataclass(frozen=True)
class BaseMetric:
    name: str
    direction: Direction

    def supports_pruning(self) -> bool:
        # pruning is only safe for monotone “loss-like” metrics
        return self.name.lower() in {"mae", "rmse", "mse"}

    def is_better(self, new: float, best: float) -> bool:
        if self.direction == "minimize":
            return new < best
        return new > best

    def as_loss(self, score: float) -> float:
        # loss-like value for pruning/reporting
        return score if self.direction == "minimize" else -score
    
    @abstractmethod
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float: ...


@register_metric("mae")
@dataclass(frozen=True)
class MAEMetric(BaseMetric):
    name: str = "MAE"
    direction: Direction = "minimize"

    def compute(self, y_true, y_pred) -> float:
        return float(mean_absolute_error(y_true, y_pred))


@register_metric("rmse")
@dataclass(frozen=True)
class RMSEMetric(BaseMetric):
    name: str = "RMSE"
    direction: Direction = "minimize"

    def compute(self, y_true, y_pred) -> float:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


@register_metric("mse")
@dataclass(frozen=True)
class MSEMetric(BaseMetric):
    name: str = "MSE"
    direction: Direction = "minimize"

    def compute(self, y_true, y_pred) -> float:
        return float(mean_squared_error(y_true, y_pred))


@register_metric("medae")
@dataclass(frozen=True)
class MedAEMetric(BaseMetric):
    name: str = "MedAE"
    direction: Direction = "minimize"

    def compute(self, y_true, y_pred) -> float:
        return float(median_absolute_error(y_true, y_pred))


@register_metric("r2")
@register_metric("r^2")
@dataclass(frozen=True)
class R2Metric(BaseMetric):
    name: str = "R2"
    direction: Direction = "maximize"

    def compute(self, y_true, y_pred) -> float:
        return float(r2_score(y_true, y_pred))


@register_metric("negmse")
@dataclass(frozen=True)
class NegMSEMetric(BaseMetric):
    name: str = "NegMSE"
    direction: Direction = "maximize"

    def compute(self, y_true, y_pred) -> float:
        return -float(mean_squared_error(y_true, y_pred))
    

@lru_cache(maxsize=None)
def _make_metric_cached(metric_key: str) -> MetricStrategy:
    cls = METRIC_REGISTRY.get(metric_key)
    if cls is None:
        raise ValueError(f"Unsupported metric_name: {metric_key}")
    return cls()


def make_metric(metric_name: str) -> MetricStrategy:
    return _make_metric_cached(metric_name.strip().lower())


class MetricFactory:
    @staticmethod
    def create(metric_name: str) -> MetricStrategy:
        return make_metric(metric_name)
