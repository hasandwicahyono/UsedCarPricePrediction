import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from collections import defaultdict

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
            if models is not None and d["model_name"] not in models:
                continue
            g[d["model_name"]].append(d)
        return g

    def available_models(self):
        return list(self.grouped.keys())

    def _explainer(self, model_type: str, model, X_bg):
        if model_type in ["DecisionTree", "RandomForest", "XGBoost"]:
            return shap.TreeExplainer(model)
        if model_type == "LinearRegression":
            return shap.LinearExplainer(model, X_bg)
        if model_type in ["SVR", "NeuralNetwork"]:
            return shap.KernelExplainer(model.predict, X_bg)
        raise ValueError(f"Unsupported model_type for SHAP: {model_type}")

    def compute(self, model_name: str):
        if model_name not in self.grouped:
            raise ValueError(f"No SHAP data for {model_name}")

        X_all, shap_all = [], []
        feature_names = self.grouped[model_name][0]["feature_names"]

        # Align features across folds (one-hot can differ by fold).
        all_feature_lists = [d.get("feature_names", feature_names) for d in self.grouped[model_name]]
        common = set(all_feature_lists[0])
        for fn in all_feature_lists[1:]:
            common &= set(fn)
        if not common:
            raise ValueError(f"No common features found across folds for {model_name}")
        common_ordered = [f for f in all_feature_lists[0] if f in common]

        for item in self.grouped[model_name]:
            X = item["X_test"]
            model = item["model"]
            model_type = item["model_type"]
            fn_item = item.get("feature_names", feature_names)
            if isinstance(fn_item, np.ndarray):
                fn_item = fn_item.tolist()
            index_map = [fn_item.index(f) for f in common_ordered]

            # background
            bg_n = min(self.background_size, X.shape[0])
            bg_idx = self.rng.choice(X.shape[0], size=bg_n, replace=False)
            X_bg = X[bg_idx]

            # eval subsample for kernel explainers
            X_eval = X
            if model_type in ["SVR", "NeuralNetwork"] and X.shape[0] > self.max_eval_samples:
                idx = self.rng.choice(X.shape[0], size=self.max_eval_samples, replace=False)
                X_eval = X[idx]

            explainer = self._explainer(model_type, model, X_bg)
            shap_vals = explainer.shap_values(X_eval)

            X_all.append(X_eval[:, index_map])
            shap_vals = np.asarray(shap_vals)
            shap_all.append(shap_vals[:, index_map])

        return np.vstack(X_all), np.vstack(shap_all), common_ordered

    def beeswarm(self, model_name: str, max_display: int = 20, figsize=(10, 6), save: bool = True):
        X, sv, fn = self.compute(model_name)
        plt.figure(figsize=figsize)
        shap.summary_plot(sv, X, feature_names=fn, max_display=max_display, show=False)
        plt.tight_layout()
        if save and self.plot_manager is not None:
            self.plot_manager.save(f"shap_beeswarm_{model_name.lower()}")
        plt.show()

    def mean_abs_table(self, model_name: str, top_k: int = 20):
        X, sv, fn = self.compute(model_name)
        imp = np.mean(np.abs(sv), axis=0)
        return (
            pd.DataFrame({"feature": fn, "mean_abs_shap": imp})
            .sort_values("mean_abs_shap", ascending=False)
            .head(top_k)
        )
