from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol
import numpy as np

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
)

Direction = Literal["minimize", "maximize"]

class MetricStrategy(Protocol):
    name: str
    direction: Direction

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float: ...
    def supports_pruning(self) -> bool: ...
    def is_better(self, new: float, best: float) -> bool: ...


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


@dataclass(frozen=True)
class MAEMetric(BaseMetric):
    name: str = "MAE"
    direction: Direction = "minimize"

    def compute(self, y_true, y_pred) -> float:
        return float(mean_absolute_error(y_true, y_pred))


@dataclass(frozen=True)
class RMSEMetric(BaseMetric):
    name: str = "RMSE"
    direction: Direction = "minimize"

    def compute(self, y_true, y_pred) -> float:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


@dataclass(frozen=True)
class MSEMetric(BaseMetric):
    name: str = "MSE"
    direction: Direction = "minimize"

    def compute(self, y_true, y_pred) -> float:
        return float(mean_squared_error(y_true, y_pred))


@dataclass(frozen=True)
class MedAEMetric(BaseMetric):
    name: str = "MedAE"
    direction: Direction = "minimize"

    def compute(self, y_true, y_pred) -> float:
        return float(median_absolute_error(y_true, y_pred))


@dataclass(frozen=True)
class R2Metric(BaseMetric):
    name: str = "R2"
    direction: Direction = "maximize"

    def compute(self, y_true, y_pred) -> float:
        return float(r2_score(y_true, y_pred))


@dataclass(frozen=True)
class NegMSEMetric(BaseMetric):
    name: str = "NegMSE"
    direction: Direction = "maximize"

    def compute(self, y_true, y_pred) -> float:
        return -float(mean_squared_error(y_true, y_pred))
    

def make_metric(metric_name: str) -> MetricStrategy:
    m = metric_name.lower()
    if m == "mae":
        return MAEMetric()
    if m == "rmse":
        return RMSEMetric()
    if m == "mse":
        return MSEMetric()
    if m == "medae":
        return MedAEMetric()
    if m in {"r2", "r^2"}:
        return R2Metric()
    if m == "negmse":
        return NegMSEMetric()
    raise ValueError(f"Unsupported metric_name: {metric_name}")