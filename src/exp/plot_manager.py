from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from .patterns import PlotStrategy

class PlotManager:
    """
    Centralized plot saving utility.
    Ensures consistent resolution, naming, and directory structure.
    """

    def __init__(
        self,
        base_dir: str = "outputs/figures/shap",
        dpi: int = 300,
        fmt: str = "png",
        tight_layout: bool = True,
    ):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.fmt = fmt
        self.tight_layout = tight_layout

    def save(self, filename: str):
        """
        Save current matplotlib figure.
        """
        path = self.base_dir / f"{filename}.{self.fmt}"
        if self.tight_layout:
            plt.tight_layout()
        plt.savefig(path, dpi=self.dpi, bbox_inches="tight")
        print(f"[saved] {path.resolve()}")

    def save_fig(self, fig, filename: str):
        """
        Save the provided matplotlib figure.
        """
        path = self.base_dir / f"{filename}.{self.fmt}"
        if self.tight_layout:
            fig.tight_layout()
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        print(f"[saved] {path.resolve()}")

    def plot_point_range(self, results_df: pd.DataFrame, metric: str = "MAE", ascending: bool = True):
        """
        Point-range plot for a metric with mean ± std across folds.
        Accepts positional or keyword args to stay backward-compatible.
        """
        agg = (
            results_df
            .groupby("model", observed=False)[metric]
            .agg(["mean", "std"])
            .sort_values("mean", ascending=ascending)
        )
        fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(agg))))
        x = agg["mean"].values
        y = np.arange(len(agg))
        err = agg["std"].values

        fig_name = f"Point-range plot by model: {metric}"
        ax.errorbar(x, y, xerr=err, fmt="o", color="#1f77b4", ecolor="#8fbce6", capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels(agg.index)
        ax.set_xlabel(f"{metric} (mean ± std)")
        ax.set_title(fig_name)
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        return fig

    def metric_summary(
        self,
        results_df: pd.DataFrame,
        metric: str = "MAE",
        ci: float = 0.95,
        ascending: bool = True,
    ) -> pd.DataFrame:
        rows = []
        alpha = 1.0 - ci
        for model, group in results_df.groupby("model", observed=False):
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy()
            n = len(values)
            mean = float(np.mean(values)) if n else np.nan
            std = float(np.std(values, ddof=1)) if n > 1 else 0.0
            sem = std / np.sqrt(n) if n > 1 else 0.0
            tcrit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)) if n > 1 else 0.0
            ci_half_width = sem * tcrit
            rows.append(
                {
                    "model": model,
                    "n": n,
                    "mean": mean,
                    "std": std,
                    "sem": sem,
                    "ci": ci,
                    "ci_low": mean - ci_half_width,
                    "ci_high": mean + ci_half_width,
                    "ci_half_width": ci_half_width,
                }
            )
        return pd.DataFrame(rows).sort_values("mean", ascending=ascending).reset_index(drop=True)

    def plot_metric_ci(
        self,
        results_df: pd.DataFrame,
        metric: str = "MAE",
        *,
        ci: float = 0.95,
        ascending: bool = True,
        baseline: str | None = None,
    ):
        """
        Mean metric plot with confidence intervals and optional baseline marker.
        """
        summary = self.metric_summary(results_df, metric=metric, ci=ci, ascending=ascending)
        fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * len(summary))))
        y = np.arange(len(summary))
        xerr = np.vstack([
            summary["mean"].to_numpy() - summary["ci_low"].to_numpy(),
            summary["ci_high"].to_numpy() - summary["mean"].to_numpy(),
        ])
        ax.errorbar(
            summary["mean"],
            y,
            xerr=xerr,
            fmt="o",
            color="#2f5f8f",
            ecolor="#9ab7d3",
            elinewidth=2,
            capsize=4,
        )
        if baseline is not None and baseline in set(summary["model"]):
            baseline_mean = float(summary.loc[summary["model"].eq(baseline), "mean"].iloc[0])
            ax.axvline(
                baseline_mean,
                color="#8a3b2f",
                linestyle="--",
                linewidth=1.5,
                label=f"Baseline: {baseline}",
            )
            ax.legend(loc="best")
        ax.set_yticks(y)
        ax.set_yticklabels(summary["model"])
        ax.set_xlabel(f"{metric} mean with {int(ci * 100)}% CI")
        ax.set_title(f"Model Comparison: {metric}")
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        return fig

    def plot_baseline_delta(
        self,
        results_df: pd.DataFrame,
        metric: str = "MAE",
        *,
        baseline: str = "RandomForest",
        ci: float = 0.95,
        lower_is_better: bool = True,
    ):
        """
        Fold-paired model delta against a baseline.
        Positive improvement means better than baseline.
        """
        pivot = results_df.pivot(index="outer_fold", columns="model", values=metric)
        if baseline not in pivot.columns:
            matches = [c for c in pivot.columns if c.startswith(f"{baseline}+")]
            if len(matches) == 1:
                baseline = matches[0]
            else:
                raise ValueError(f"Baseline '{baseline}' not found. Available: {list(pivot.columns)}")

        rows = []
        alpha = 1.0 - ci
        for model in pivot.columns:
            if model == baseline:
                continue
            paired = pivot[[baseline, model]].dropna()
            if paired.empty:
                continue
            if lower_is_better:
                delta = paired[baseline] - paired[model]
            else:
                delta = paired[model] - paired[baseline]
            values = delta.to_numpy()
            n = len(values)
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if n > 1 else 0.0
            sem = std / np.sqrt(n) if n > 1 else 0.0
            tcrit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)) if n > 1 else 0.0
            rows.append(
                {
                    "model": model,
                    "delta_mean": mean,
                    "ci_half_width": sem * tcrit,
                    "n": n,
                }
            )
        summary = pd.DataFrame(rows).sort_values("delta_mean", ascending=False)
        fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * len(summary))))
        colors = np.where(summary["delta_mean"].to_numpy() >= 0, "#2f7d5b", "#a64b45")
        y = np.arange(len(summary))
        ax.barh(y, summary["delta_mean"], xerr=summary["ci_half_width"], color=colors, alpha=0.9, capsize=4)
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(summary["model"])
        ax.invert_yaxis()
        ax.set_xlabel(f"Improvement over {baseline} ({metric})")
        ax.set_title(f"Baseline Delta: {metric}")
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        return fig

    def save_metric_comparison_plots(
        self,
        results_df: pd.DataFrame,
        metrics: list[str],
        *,
        baseline: str = "RandomForest",
        ci: float = 0.95,
        lower_is_better: dict[str, bool] | None = None,
    ) -> dict[str, str]:
        saved = {}
        lower_is_better = lower_is_better or {}
        for metric in metrics:
            ascending = lower_is_better.get(metric, metric.upper() != "R2")
            fig = self.plot_metric_ci(
                results_df,
                metric=metric,
                ci=ci,
                ascending=ascending,
                baseline=baseline,
            )
            stem = f"model_comparison_{metric.lower()}_ci"
            self.save_fig(fig, stem)
            saved[stem] = str(self.base_dir / f"{stem}.{self.fmt}")

            try:
                fig = self.plot_baseline_delta(
                    results_df,
                    metric=metric,
                    baseline=baseline,
                    ci=ci,
                    lower_is_better=ascending,
                )
                stem = f"baseline_delta_{metric.lower()}"
                self.save_fig(fig, stem)
                saved[stem] = str(self.base_dir / f"{stem}.{self.fmt}")
            except ValueError:
                continue
        return saved

    def plot_correlation_heatmap(
        self,
        df: pd.DataFrame,
        cols: list,
        title: str = "Correlation Heatmap (Numeric Features + Target)",
        figsize=(9, 7),
        annotate: bool = True,
    ):
        """
        Correlation heatmap for specified columns.
        """
        corr = df[cols].corr()
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax, label="Correlation")
        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right")
        ax.set_yticklabels(cols)

        if annotate:
            for i in range(len(cols)):
                for j in range(len(cols)):
                    val = corr.iat[i, j]
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="black")

        ax.set_title(title)
        return fig


class PointRangePlot(PlotStrategy):
    def __init__(self, plot_manager: PlotManager):
        self.plot_manager = plot_manager

    def render(self, df_results: pd.DataFrame, metric: str = "MAE", ascending: bool = True):
        return self.plot_manager.plot_point_range(df_results, metric=metric, ascending=ascending)
