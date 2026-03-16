from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
            .groupby("model")[metric]
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
