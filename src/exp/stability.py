import numpy as np
import pandas as pd
from itertools import combinations

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def feature_selection_stability(records: list) -> pd.DataFrame:
    """
    records: list of dicts from runner.feature_stability_ filtered to one model
    """
    # build selected feature sets per fold
    fold_sets = {}
    for r in records:
        fold = r["outer_fold"]
        fn = r["feature_names"]
        mask = r["selection_mask"]

        if mask is None:
            # no selection => treat as "all"
            sel = set(fn)
        else:
            # mask corresponds to numeric pipeline output only; easiest is:
            # store also numeric_feature_names_out separately if you want perfect accounting.
            # practical shortcut: compute stability only on features after preprocessing
            # by recomputing selector support over full names when possible.
            # If you want exact, store numeric_feature_names_out in tuning.
            sel = set(np.array(fn)[mask]) if len(mask) == len(fn) else set(fn)

        fold_sets[fold] = sel

    # selection frequency
    all_feats = sorted(set().union(*fold_sets.values()))
    freq = {f: 0 for f in all_feats}
    for s in fold_sets.values():
        for f in s:
            freq[f] += 1

    freq_df = pd.DataFrame({
        "feature": list(freq.keys()),
        "selected_count": list(freq.values()),
    })
    freq_df["selected_frac"] = freq_df["selected_count"] / len(fold_sets)

    # pairwise Jaccard
    pairs = []
    folds = sorted(fold_sets.keys())
    for i, j in combinations(folds, 2):
        pairs.append({
            "fold_i": i,
            "fold_j": j,
            "jaccard": jaccard(fold_sets[i], fold_sets[j])
        })
    jac_df = pd.DataFrame(pairs)
    jac_summary = jac_df["jaccard"].agg(["mean","std"]).to_dict()

    return freq_df.sort_values("selected_frac", ascending=False), jac_df, jac_summary
