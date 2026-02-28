from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def _first_existing(columns: pd.Index, candidates: List[str]) -> str | None:
    for c in candidates:
        if c in columns:
            return c
    return None


def _build_interaction_features(
    X: pd.DataFrame,
    *,
    reference_year: int = 2025,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Add XGBoost-only interaction-heavy features from common automotive columns.
    Returns transformed dataframe plus new numeric/categorical column names.
    """
    Xo = X.copy()
    cols = Xo.columns
    new_num: List[str] = []
    new_cat: List[str] = []

    year_col = _first_existing(cols, ["year", "model_year", "manufacture_year"])
    mileage_col = _first_existing(cols, ["mileage", "odometer", "km", "kilometers", "kilometres"])
    brand_col = _first_existing(cols, ["brand", "make"])
    model_col = _first_existing(cols, ["model", "series"])

    mileage_num = None
    if mileage_col is not None:
        mileage_num = pd.to_numeric(Xo[mileage_col], errors="coerce")

    age_col = "vehicle_age"
    if year_col is not None:
        year_num = pd.to_numeric(Xo[year_col], errors="coerce")
        Xo[age_col] = (reference_year - year_num).clip(lower=0)
        new_num.append(age_col)

    bin_col = "mileage_bin"
    if mileage_num is not None:
        Xo[bin_col] = pd.cut(
            mileage_num,
            bins=[-np.inf, 10_000, 30_000, 60_000, 100_000, np.inf],
            labels=["very_low", "low", "mid", "high", "very_high"],
        ).astype("object")
        new_cat.append(bin_col)

    if mileage_num is not None and age_col in Xo.columns:
        inter_col = "age_x_mileage"
        ratio_col = "mileage_per_year"
        age = pd.to_numeric(Xo[age_col], errors="coerce").fillna(0.0)
        mileage_safe = mileage_num.fillna(0.0)
        Xo[inter_col] = age * mileage_safe
        Xo[ratio_col] = mileage_safe / np.maximum(age, 1.0)
        new_num.extend([inter_col, ratio_col])

    if brand_col is not None and model_col is not None:
        combo_col = "brand_model"
        Xo[combo_col] = Xo[brand_col].astype(str) + "__" + Xo[model_col].astype(str)
        new_cat.append(combo_col)

    return Xo, new_num, new_cat


def add_interaction_features(
    X: pd.DataFrame,
    interaction_policy: str,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    if interaction_policy == "interactions":
        return _build_interaction_features(X)
    return X, [], []
