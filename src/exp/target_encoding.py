from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class LeakageSafeTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Leak-safe *by construction* when used in CV:
    - fit() uses only fold-train (X_train, y_train)
    - transform() applies learned mapping to X_valid/test
    """
    def __init__(
        self,
        cols: List[str],
        smoothing: float = 10.0,
        min_samples_leaf: int = 1,
        noise_std: float = 0.0,
        random_state: int = 42,
        output_dtype: str = "float64",
    ):
        self.cols = cols
        self.smoothing = float(smoothing)
        self.min_samples_leaf = int(min_samples_leaf)
        self.noise_std = float(noise_std)
        self.random_state = int(random_state)
        self.output_dtype = output_dtype

        self.global_mean_ = None
        self.mapping_: Dict[str, Dict[Any, float]] = {}

    def fit(self, X: pd.DataFrame, y):
        X = pd.DataFrame(X).copy()
        y = np.asarray(y).astype(float)
        self.global_mean_ = float(np.mean(y))

        self.mapping_.clear()
        for c in self.cols:
            s = X[c]

            # group stats
            df = pd.DataFrame({"cat": s, "y": y})
            stats = df.groupby("cat")["y"].agg(["count", "mean"])

            # smoothing:
            # enc = (count*mean + smoothing*global) / (count + smoothing)
            cnt = stats["count"].astype(float)
            mu = stats["mean"].astype(float)

            # optional: enforce min_samples_leaf by increasing shrinkage for tiny counts
            eff_cnt = np.maximum(cnt, self.min_samples_leaf)

            enc = (eff_cnt * mu + self.smoothing * self.global_mean_) / (eff_cnt + self.smoothing)
            self.mapping_[c] = enc.to_dict()

        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        for col in self.cols:
            enc = X[col].map(self.mapping_[col])
            # Ensure numeric dtype before fillna to avoid categorical setitem errors.
            enc = pd.Series(enc, index=X.index, dtype="float64")
            enc = enc.fillna(self.global_mean_)
            X[f"{col}__te"] = enc.astype(float)
            X.drop(columns=[col], inplace=True)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{col}__te" for col in self.cols], dtype=object)
