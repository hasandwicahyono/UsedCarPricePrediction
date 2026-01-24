import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

def summarize_mean_std(df_results: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    s = (
        df_results
        .groupby("model")[["R2","MAE","MedAE","MSE","RMSE"]]
        .agg(["mean","std"])
        .reset_index()
    )
    s.columns = ["model"] + [f"{m}_{stat}" for (m, stat) in s.columns[1:]]
    num_cols = [c for c in s.columns if c != "model"]
    fmt = f"{{:,.{decimals}f}}"
    s[num_cols] = s[num_cols].applymap(lambda x: fmt.format(x))
    return s


def paired_tests(
    df_results: pd.DataFrame,
    metric: str = "MAE",
    baseline: str = "RandomForest",
    models: list[str] | None = None,
) -> pd.DataFrame:
    pivot = df_results.pivot(index="outer_fold", columns="model", values=metric)
    if models is not None:
        keep = [m for m in models if m in pivot.columns]
        if baseline not in keep and baseline in pivot.columns:
            keep = [baseline] + keep
        pivot = pivot[keep]
    base = pivot[baseline]
    out = []
    for m in pivot.columns:
        if m == baseline:
            continue
        out.append({
            "metric": metric,
            "baseline": baseline,
            "model": m,
            "paired_t_p": float(ttest_rel(pivot[m], base).pvalue),
            "wilcoxon_p": float(wilcoxon(pivot[m], base).pvalue),
            "n_outer_folds": int(pivot.shape[0])
        })
    if not out:
        return pd.DataFrame(columns=["metric","baseline","model","paired_t_p","wilcoxon_p","n_outer_folds"])
    return pd.DataFrame(out).sort_values(["paired_t_p","wilcoxon_p"])


def significance_matrix(df_results: pd.DataFrame, metric: str = "MAE") -> pd.DataFrame:
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
