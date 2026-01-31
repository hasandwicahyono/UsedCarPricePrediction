from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Protocol, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RunContext:
    run_id: str
    outer_fold: Optional[int] = None
    inner_fold: Optional[int] = None
    model_name: Optional[str] = None
    trial_id: Optional[int] = None


class DataSource(Protocol):
    def read(self) -> pd.DataFrame: ...


class Evaluator(Protocol):
    def summary(self, df_results: pd.DataFrame) -> pd.DataFrame: ...
    def paired_tests(self, df_results: pd.DataFrame, metric: str, baseline: str) -> pd.DataFrame: ...
    def significance_matrix(self, df_results: pd.DataFrame, metric: str) -> pd.DataFrame: ...


class PlotStrategy(Protocol):
    def render(self, df_results: pd.DataFrame, metric: str = "MAE", ascending: bool = True): ...


class SplitStrategy(Protocol):
    def split(
        self,
        X: pd.DataFrame,
        y: Optional[np.ndarray],
        seed: int,
    ) -> Iterable[Tuple[np.ndarray, np.ndarray]]: ...


class TuningObserver(Protocol):
    def on_outer_fold_start(self, ctx: RunContext) -> None: ...
    def on_outer_fold_end(self, ctx: RunContext) -> None: ...
    def on_trial_start(self, ctx: RunContext) -> None: ...
    def on_trial_end(self, ctx: RunContext, score: Optional[float]) -> None: ...
