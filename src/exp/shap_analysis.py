from .shap_explainers import ShapExplainerFactory
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from collections import defaultdict
from .schema_utils import clean_feature_name
#from .factories import build_shap_explainer
from tqdm.auto import tqdm

class ShapAnalyzer:
    """
    Consumes shap_store_ produced by NestedCVRunner (outer-test only).
    Supports Tree/Linear/Kernel explainers.
    Kernel explainers (SVR/NN) are subsampled by default.
    """
    def __init__(
        self, 
        shap_store: list, 
        background_size: int = 100, 
        max_eval_samples: int = 300, 
        seed: int = 42, 
        plot_manager: any = None,
        models: list[str] | None = None,
    ):
        
        self.store = shap_store
        self.background_size = background_size
        self.max_eval_samples = max_eval_samples
        self.rng = np.random.default_rng(seed)
        self.grouped = self._group(models)
        self.plot_manager = plot_manager

    def _group(self, models: list[str] | None = None):
        g = defaultdict(list)
        for d in self.store:
            labels = [d["model_name"]]
            if "model_label" in d and d["model_label"] != d["model_name"]:
                labels.append(d["model_label"])

            if models is not None and not any(m in models for m in labels):
                continue

            for label in labels:
                g[label].append(d)
        return g

    def available_models(self):
        return list(self.grouped.keys())

    def compute(self, model_name: str):
        if model_name not in self.grouped:
            raise ValueError(f"No SHAP data for {model_name}")
        entries = self.grouped[model_name]
        feature_lists = [list(e["feature_names"]) for e in entries]
        common = set(feature_lists[0])
        for fns in feature_lists[1:]:
            common &= set(fns)
        if not common:
            raise ValueError(f"No common features across folds for {model_name}")
        common_ordered = [f for f in feature_lists[0] if f in common]

        X_all, sv_all = [], []
        kernel_policies = {"kernel"}

        for item in tqdm(entries, desc=f"Computing SHAP for {model_name}"):
            if "X_test" not in item or "model" not in item:
                raise KeyError(f"Missing X_test or model in SHAP store for {model_name}")   
            X = item["X_test"]
            model = item["model"]

            policy = item.get("explain_policy")
            # guard for legacy entries that stored a class instead of a string
            if hasattr(policy, "__name__"):
                policy = policy.policy if hasattr(policy, "policy") else policy.__name__.lower()
            if policy is None:
                raise KeyError(f"Missing explain_policy for {model_name}")

            # background
            bg_n = min(self.background_size, X.shape[0])
            bg_idx = self.rng.choice(X.shape[0], bg_n, replace=False)
            X_bg = X[bg_idx]

            explainer = ShapExplainerFactory.create(policy, model, X_bg) #build_shap_explainer(policy, model, X_bg)

            X_eval = X
            if policy in kernel_policies and X.shape[0] > self.max_eval_samples:
                idx = self.rng.choice(X.shape[0], self.max_eval_samples, replace=False)
                X_eval = X[idx]

            sv = np.asarray(explainer.explain(X_eval))

            fn_item = list(item["feature_names"])
            idx_lookup = {f: i for i, f in enumerate(fn_item)}
            idx_map = [idx_lookup[f] for f in common_ordered]

            X_all.append(X_eval[:, idx_map])
            sv_all.append(sv[:, idx_map])
 
        cleaned_names = [clean_feature_name(f) for f in common_ordered]
        return np.vstack(X_all), np.vstack(sv_all), cleaned_names

    def beeswarm(self, model_name: str, max_display: int = 20, figsize=(10, 6), save: bool = True):
        X, sv, fn = self.compute(model_name)
        plt.figure(figsize=figsize)
        shap.summary_plot(
            sv, 
            X, 
            feature_names=fn, 
            max_display=max_display, 
            show=False
        )
        plt.tight_layout()
        if save and self.plot_manager is not None:
            self.plot_manager.save(f"shap_beeswarm_{model_name.lower()}")
        plt.show()
