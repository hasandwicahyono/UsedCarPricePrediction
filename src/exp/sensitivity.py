from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PARAM_PREFIX = "param__"


def trial_records_to_frame(records: Iterable[dict]) -> pd.DataFrame:
    return pd.DataFrame(list(records))


def _eta_squared(values: pd.Series, groups: pd.Series) -> float:
    overall = values.mean()
    total = float(((values - overall) ** 2).sum())
    if total <= 0:
        return 0.0
    between = 0.0
    for _, subset in values.groupby(groups):
        between += float(len(subset) * (subset.mean() - overall) ** 2)
    return between / total


def summarize_hyperparameter_sensitivity(
    trials_df: pd.DataFrame,
    *,
    min_trials: int = 3,
) -> pd.DataFrame:
    """
    Rank hyperparameters by association with validation score.

    Numeric parameters use absolute Spearman correlation with the Optuna score.
    Categorical parameters use eta-squared, the share of score variance explained
    by group means. Both scores are in [0, 1], where larger means more sensitive.
    """
    if trials_df is None or trials_df.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "base_model",
                "residual_kind",
                "metric",
                "direction",
                "parameter",
                "parameter_type",
                "n_trials",
                "n_unique",
                "sensitivity",
                "spearman_corr",
                "best_value",
                "worst_value",
                "score_mean",
                "score_std",
            ]
        )

    df = trials_df.copy()
    df = df[df.get("state", "COMPLETE").eq("COMPLETE")]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[np.isfinite(df["value"])]
    param_cols = [c for c in df.columns if c.startswith(PARAM_PREFIX)]

    rows = []
    group_cols = ["model", "base_model", "residual_kind", "metric", "direction"]
    for group_key, group in df.groupby(group_cols, dropna=False):
        direction = group_key[group_cols.index("direction")]
        for col in param_cols:
            subset = group[[col, "value"]].dropna()
            if len(subset) < min_trials:
                continue
            n_unique = subset[col].nunique(dropna=True)
            if n_unique < 2:
                continue

            numeric = pd.to_numeric(subset[col], errors="coerce")
            parameter_type = "numeric" if numeric.notna().all() else "categorical"
            spearman_corr = np.nan
            if parameter_type == "numeric":
                spearman_corr = float(numeric.corr(subset["value"], method="spearman"))
                sensitivity = abs(spearman_corr) if np.isfinite(spearman_corr) else 0.0
                level_means = subset.assign(_param=numeric).groupby("_param")["value"].mean()
            else:
                labels = subset[col].astype(str)
                sensitivity = float(_eta_squared(subset["value"], labels))
                level_means = subset.assign(_param=labels).groupby("_param")["value"].mean()

            if direction == "maximize":
                best_value = level_means.idxmax()
                worst_value = level_means.idxmin()
            else:
                best_value = level_means.idxmin()
                worst_value = level_means.idxmax()

            row = dict(zip(group_cols, group_key))
            row.update(
                {
                    "parameter": col[len(PARAM_PREFIX):],
                    "parameter_type": parameter_type,
                    "n_trials": int(len(subset)),
                    "n_unique": int(n_unique),
                    "sensitivity": float(sensitivity),
                    "spearman_corr": spearman_corr,
                    "best_value": best_value,
                    "worst_value": worst_value,
                    "score_mean": float(subset["value"].mean()),
                    "score_std": float(subset["value"].std(ddof=1)) if len(subset) > 1 else 0.0,
                }
            )
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["model", "sensitivity"], ascending=[True, False])
        .reset_index(drop=True)
    )


def plot_hyperparameter_sensitivity(
    sensitivity_df: pd.DataFrame,
    *,
    model: str | None = None,
    top_n: int = 20,
    title: str = "Hyperparameter Sensitivity",
):
    if sensitivity_df is None or sensitivity_df.empty:
        raise ValueError("No sensitivity rows available to plot.")

    import matplotlib.pyplot as plt

    df = sensitivity_df.copy()
    if model is not None:
        df = df[df["model"].eq(model) | df["base_model"].eq(model)]
    if df.empty:
        raise ValueError(f"No sensitivity rows available for model={model!r}.")

    df = df.sort_values("sensitivity", ascending=False).head(top_n)
    labels = df["model"].astype(str) + " | " + df["parameter"].astype(str)
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(9, max(3, 0.38 * len(df))))
    ax.barh(y, df["sensitivity"], color="#3b6ea8")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Sensitivity score")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    return fig


def save_hyperparameter_sensitivity(
    trials_df: pd.DataFrame,
    *,
    out_dir: str = "outputs/sensitivity",
    top_n: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    trials_df.to_csv(out_path / "optuna_trials.csv", index=False)
    summary = summarize_hyperparameter_sensitivity(trials_df)
    summary.to_csv(out_path / "hyperparameter_sensitivity.csv", index=False)

    fig_path = out_path / "hyperparameter_sensitivity_top.png"
    if not summary.empty:
        fig = plot_hyperparameter_sensitivity(summary, top_n=top_n)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    return trials_df, summary, str(fig_path)
