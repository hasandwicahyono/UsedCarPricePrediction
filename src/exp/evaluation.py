import pandas as pd
import pandas.errors as pd_errors
from scipy.stats import ttest_rel, wilcoxon
from .patterns import Evaluator

def _ensure_pandas4warning():
    # Work around mixed pandas versions where core code expects Pandas4Warning.
    if not hasattr(pd_errors, "Pandas4Warning"):
        class Pandas4Warning(Warning):
            pass
        pd_errors.Pandas4Warning = Pandas4Warning

def summarize_mean_std(
    df_results: pd.DataFrame,
    decimals: int = 4,
    format: bool = False,
) -> pd.DataFrame:
    s = (
        df_results
        .groupby("model")[["R2","MAE","MedAE","MSE","RMSE"]]
        .agg(["mean","std"])
        .reset_index()
    )
    s.columns = ["model"] + [f"{m}_{stat}" for (m, stat) in s.columns[1:]]
    num_cols = [c for c in s.columns if c != "model"]
    if format:
        fmt = f"{{:,.{decimals}f}}"
        def _format_cols(df: pd.DataFrame) -> pd.DataFrame:
            try:
                return df.applymap(lambda x: fmt.format(x))
            except AttributeError:
                # fallback for pandas builds without DataFrame.applymap
                return df.map(lambda x: fmt.format(x))

        s[num_cols] = _format_cols(s[num_cols])
    return s


def match_one(pivot: pd.DataFrame, name: str) -> str | None:
    if name in pivot.columns:
        return name
    matches = [c for c in pivot.columns if c.startswith(f"{name}+")]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous model name '{name}'. Matches: {matches}"
        )
    return None
    
def match_many(pivot: pd.DataFrame, names: list[str]) -> list[str]:
    out = []
    for n in names:
        if n in pivot.columns:
            out.append(n)
            continue
        matches = [c for c in pivot.columns if c.startswith(f"{n}+")]
        out.extend(matches)
    return out


def paired_tests(
    df_results: pd.DataFrame,
    metric: str = "MAE",
    baseline: str = "RandomForest",
    models: list[str] | None = None,
) -> pd.DataFrame:
    _ensure_pandas4warning()
    pivot = df_results.pivot(index="outer_fold", columns="model", values=metric)

    baseline_col = match_one(pivot, baseline)
    if baseline_col is None:
        raise ValueError(
            f"Baseline '{baseline}' not found. Available: {list(pivot.columns)}"
        )

    if models is not None:
        keep = match_many(pivot, models)
        if baseline_col not in keep:
            keep = [baseline_col] + keep
        pivot = pivot[keep]
    base = pivot[baseline_col]
    out = []
    for m in pivot.columns:
        if m == baseline_col:
            continue
        out.append({
            "metric": metric,
            "baseline": baseline_col,
            "model": m,
            "paired_t_p": float(ttest_rel(pivot[m], base).pvalue),
            "wilcoxon_p": float(wilcoxon(pivot[m], base).pvalue),
            "n_outer_folds": int(pivot.shape[0])
        })
    if not out:
        return pd.DataFrame(columns=["metric","baseline","model","paired_t_p","wilcoxon_p","n_outer_folds"])
    return pd.DataFrame(out).sort_values(["paired_t_p","wilcoxon_p"])


def significance_matrix(df_results: pd.DataFrame, metric: str = "MAE") -> pd.DataFrame:
    _ensure_pandas4warning()
    pivot = df_results.pivot(index="outer_fold", columns="model", values=metric)
    models = list(pivot.columns)
    out = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1, m2 = models[i], models[j]
            out.append({
                "metric": metric,
                "model_a": m1,
                "model_b": m2,
                "paired_t_p": float(ttest_rel(pivot[m1], pivot[m2]).pvalue),
                "wilcoxon_p": float(wilcoxon(pivot[m1], pivot[m2]).pvalue),
                "n_outer_folds": int(pivot.shape[0]),
            })
    if not out:
        return pd.DataFrame(columns=["metric","model_a","model_b","paired_t_p","wilcoxon_p","n_outer_folds"])
    return pd.DataFrame(out).sort_values(["paired_t_p","wilcoxon_p"])


class DefaultEvaluator(Evaluator):
    def summary(self, df_results: pd.DataFrame) -> pd.DataFrame:
        return summarize_mean_std(df_results)

    def paired_tests(
        self,
        df_results: pd.DataFrame,
        metric: str = "MAE",
        baseline: str = "RandomForest",
    ) -> pd.DataFrame:
        return paired_tests(df_results, metric=metric, baseline=baseline)

    def significance_matrix(self, df_results: pd.DataFrame, metric: str = "MAE") -> pd.DataFrame:
        return significance_matrix(df_results, metric=metric)
