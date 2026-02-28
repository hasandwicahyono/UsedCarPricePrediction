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
        max_tree_eval_samples: int = 500,
        max_eval_samples: int = 300, 
        max_entries_per_model: int | None = None,
        fast_background_size: int = 50,
        fast_max_tree_eval_samples: int = 200,
        fast_max_eval_samples: int = 120,
        fast_max_entries_per_model: int = 2,
        seed: int = 42, 
        plot_manager: any = None,
        models: list[str] | None = None,
    ):
        
        self.store = shap_store
        self.background_size = background_size
        self.max_tree_eval_samples = max_tree_eval_samples
        self.max_eval_samples = max_eval_samples
        self.max_entries_per_model = max_entries_per_model
        self.fast_background_size = fast_background_size
        self.fast_max_tree_eval_samples = fast_max_tree_eval_samples
        self.fast_max_eval_samples = fast_max_eval_samples
        self.fast_max_entries_per_model = fast_max_entries_per_model
        self.rng = np.random.default_rng(seed)
        self.grouped = self._group(models)
        self._idx_map_cache = {}
        self.plot_manager = plot_manager

    def _group(self, models: list[str] | None = None):
        g = defaultdict(list)
        model_filter = set(models) if models is not None else None
        for d in self.store:
            labels = [d["model_name"]]
            if "model_label" in d and d["model_label"] != d["model_name"]:
                labels.append(d["model_label"])

            if model_filter is not None and not any(lbl in model_filter for lbl in labels):
                continue

            for label in labels:
                g[label].append(d)
        return g

    def available_models(self):
        return list(self.grouped.keys())

    @staticmethod
    def _is_random_forest_entry(item: dict, model_name: str) -> bool:
        model_type = str(item.get("model_type", ""))
        if model_type.lower() == "randomforest":
            return True
        return "randomforest" in str(model_name).lower()

    def compute(self, model_name: str):
        if model_name not in self.grouped:
            raise ValueError(f"No SHAP data for {model_name}")
        entries = self.grouped[model_name]

        is_rf = any(self._is_random_forest_entry(e, model_name) for e in entries)

        max_entries = self.max_entries_per_model
        if is_rf:
            if max_entries is None:
                max_entries = self.fast_max_entries_per_model
            else:
                max_entries = min(max_entries, self.fast_max_entries_per_model)

        if max_entries is not None and len(entries) > max_entries:
            keep_idx = self.rng.choice(len(entries), max_entries, replace=False)
            entries = [entries[i] for i in np.sort(keep_idx)]

        bg_size = self.background_size
        max_eval = self.max_eval_samples
        max_tree_eval = self.max_tree_eval_samples
        if is_rf:
            bg_size = min(bg_size, self.fast_background_size)
            max_eval = min(max_eval, self.fast_max_eval_samples)
            max_tree_eval = min(max_tree_eval, self.fast_max_tree_eval_samples)

        feature_lists = [list(e["feature_names"]) for e in entries]
        common = set(feature_lists[0])
        for fns in feature_lists[1:]:
            common &= set(fns)
        if not common:
            raise ValueError(f"No common features across folds for {model_name}")
        common_ordered = [f for f in feature_lists[0] if f in common]
        common_tuple = tuple(common_ordered)

        X_all, sv_all = [], []

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
            bg_n = min(bg_size, X.shape[0])
            bg_idx = self.rng.choice(X.shape[0], bg_n, replace=False)
            X_bg = X[bg_idx]

            explainer = ShapExplainerFactory.create(policy, model, X_bg) #build_shap_explainer(policy, model, X_bg)

            X_eval = X
            # Kernel explainer is always expensive.
            if policy == "kernel" and X.shape[0] > max_eval:
                idx = self.rng.choice(X.shape[0], max_eval, replace=False)
                X_eval = X[idx]
            # Tree policy can still be slow when it falls back to model-agnostic SHAP.
            # Detect fallback flag from TreeShapExplainer and cap eval rows.
            if (
                policy == "tree"
                and getattr(explainer, "_use_call_api", False)
                and X.shape[0] > max_tree_eval
            ):
                idx = self.rng.choice(X.shape[0], max_tree_eval, replace=False)
                X_eval = X[idx]

            sv = np.asarray(explainer.explain(X_eval))

            fn_item = list(item["feature_names"])
            cache_key = (model_name, tuple(fn_item), common_tuple)
            idx_map = self._idx_map_cache.get(cache_key)
            if idx_map is None:
                idx_lookup = {f: i for i, f in enumerate(fn_item)}
                idx_map = [idx_lookup[f] for f in common_ordered]
                self._idx_map_cache[cache_key] = idx_map

            X_all.append(X_eval[:, idx_map])
            sv_all.append(sv[:, idx_map])
 
        cleaned_names = [clean_feature_name(f) for f in common_ordered]
        if len(X_all) == 1:
            return X_all[0], sv_all[0], cleaned_names
        return np.vstack(X_all), np.vstack(sv_all), cleaned_names

    def beeswarm(self, model_name: str, max_display: int = 20, figsize=(10, 6), save: bool = True):
        X, sv, fn = self.compute(model_name)
        
        # --- Figure 1: Summary / beeswarm ---
        plt.figure(figsize=figsize)
        shap.summary_plot(
            sv, 
            X, 
            feature_names=fn, 
            max_display=max_display, 
            show=False
        )
        plt.title(f"SHAP Beeswarm Plot for {model_name}")
        plt.tight_layout()
        if save and self.plot_manager is not None:
            self.plot_manager.save(f"shap_beeswarm_{model_name.lower()}")
        plt.show()

        # --- Figure 2: Global importance (bar) ---
        plt.figure(figsize=figsize)
        shap.summary_plot(
            sv, 
            X, 
            plot_type="bar",
            feature_names=fn, 
            max_display=max_display, 
            show=False
        )
        plt.title(f"SHAP Global Importance Plot for {model_name}")
        plt.tight_layout()
        if save and self.plot_manager is not None:
            self.plot_manager.save(f"shap_GlobalImportance_{model_name.lower()}")
        plt.show()

        # --- Figure 3: Waterfall ---
        plt.figure(figsize=figsize)
        waterfall_values = sv
        if not isinstance(sv, shap.Explanation):
            waterfall_values = shap.Explanation(values=sv, data=X, feature_names=fn, base_values=0.0)
        # Use first instance for waterfall to avoid plotting all rows at once.
        shap.plots.waterfall(
            waterfall_values[0],
            max_display=max_display,
            show=False
        )
        plt.title(f"SHAP Waterfall Plot for {model_name}")
        plt.tight_layout()
        if save and self.plot_manager is not None:
            self.plot_manager.save(f"shap_Waterfall_{model_name.lower()}")
        plt.show()
