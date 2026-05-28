from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .schema_utils import clean_feature_name


def _as_matrix(X):
    if hasattr(X, "to_numpy"):
        X = X.to_numpy()
    elif hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=np.float64)


def _predict(model, X):
    return np.asarray(model.predict(X)).reshape(-1)


def _group_store(store: list[dict], models: list[str] | None = None) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    model_filter = set(models) if models is not None else None
    for item in store:
        labels = [item.get("model_name")]
        if item.get("model_label") and item.get("model_label") != item.get("model_name"):
            labels.append(item.get("model_label"))
        labels = [label for label in labels if label]
        if model_filter is not None and not any(label in model_filter for label in labels):
            continue
        for label in labels:
            grouped[label].append(item)
    return grouped


class ComplementaryExplainer:
    """
    Model-agnostic companion explainability for fitted outer-fold models.

    Partial dependence summarizes average prediction changes across feature grids.
    Counterfactual search reports small one-feature changes that move predictions
    by a target amount in transformed feature space.
    """

    def __init__(
        self,
        explain_store: list[dict],
        *,
        seed: int = 42,
        max_entries_per_model: int | None = 2,
        max_samples: int = 200,
        grid_size: int = 20,
        models: list[str] | None = None,
    ):
        self.store = explain_store
        self.rng = np.random.default_rng(seed)
        self.max_entries_per_model = max_entries_per_model
        self.max_samples = max_samples
        self.grid_size = grid_size
        self.grouped = _group_store(explain_store, models=models)

    def available_models(self) -> list[str]:
        return list(self.grouped.keys())

    def _entries(self, model_name: str):
        if model_name not in self.grouped:
            raise ValueError(f"No explainability data for {model_name}")
        entries = self.grouped[model_name]
        if self.max_entries_per_model is not None and len(entries) > self.max_entries_per_model:
            idx = self.rng.choice(len(entries), self.max_entries_per_model, replace=False)
            entries = [entries[i] for i in np.sort(idx)]
        return entries

    def _sample_rows(self, X):
        X = _as_matrix(X)
        if X.shape[0] <= self.max_samples:
            return X
        idx = self.rng.choice(X.shape[0], self.max_samples, replace=False)
        return X[np.sort(idx)]

    def _default_features(self, entries, max_features: int) -> list[int]:
        X = np.vstack([self._sample_rows(item["X_test"]) for item in entries])
        variances = np.nanvar(X, axis=0)
        order = np.argsort(variances)[::-1]
        return [int(i) for i in order[:max_features] if np.isfinite(variances[i]) and variances[i] > 0]

    def partial_dependence(
        self,
        model_name: str,
        *,
        features: list[str] | list[int] | None = None,
        max_features: int = 8,
        grid_size: int | None = None,
    ) -> pd.DataFrame:
        entries = self._entries(model_name)
        feature_names = list(entries[0]["feature_names"])
        cleaned_names = [clean_feature_name(f) for f in feature_names]
        grid_size = int(grid_size or self.grid_size)

        if features is None:
            feature_idx = self._default_features(entries, max_features)
        else:
            lookup = {name: i for i, name in enumerate(feature_names)}
            cleaned_lookup = {name: i for i, name in enumerate(cleaned_names)}
            feature_idx = []
            for feature in features:
                if isinstance(feature, int):
                    feature_idx.append(feature)
                elif feature in lookup:
                    feature_idx.append(lookup[feature])
                elif feature in cleaned_lookup:
                    feature_idx.append(cleaned_lookup[feature])
                else:
                    raise ValueError(f"Unknown feature for PDP: {feature}")

        rows = []
        for item in entries:
            X = self._sample_rows(item["X_test"])
            model = item["model"]
            fold = item.get("outer_fold")
            for idx in feature_idx:
                values = X[:, idx]
                lo, hi = np.nanpercentile(values, [5, 95])
                if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                    continue
                grid = np.linspace(lo, hi, grid_size)
                for value in grid:
                    X_mod = X.copy()
                    X_mod[:, idx] = value
                    preds = _predict(model, X_mod)
                    rows.append(
                        {
                            "model": model_name,
                            "outer_fold": fold,
                            "feature": cleaned_names[idx],
                            "feature_index": idx,
                            "grid_value": float(value),
                            "prediction_mean": float(np.mean(preds)),
                            "prediction_std": float(np.std(preds)),
                            "n_samples": int(len(preds)),
                        }
                    )
        return pd.DataFrame(rows)

    def counterfactuals(
        self,
        model_name: str,
        *,
        n_instances: int = 5,
        features: list[str] | list[int] | None = None,
        desired_delta: float | None = None,
        direction: str = "increase",
        grid_size: int | None = None,
    ) -> pd.DataFrame:
        entries = self._entries(model_name)
        feature_names = list(entries[0]["feature_names"])
        cleaned_names = [clean_feature_name(f) for f in feature_names]
        grid_size = int(grid_size or self.grid_size)
        direction = direction.lower()
        if direction not in {"increase", "decrease"}:
            raise ValueError("direction must be 'increase' or 'decrease'")

        if features is None:
            feature_idx = self._default_features(entries, max_features=8)
        else:
            lookup = {name: i for i, name in enumerate(feature_names)}
            cleaned_lookup = {name: i for i, name in enumerate(cleaned_names)}
            feature_idx = []
            for feature in features:
                if isinstance(feature, int):
                    feature_idx.append(feature)
                elif feature in lookup:
                    feature_idx.append(lookup[feature])
                elif feature in cleaned_lookup:
                    feature_idx.append(cleaned_lookup[feature])
                else:
                    raise ValueError(f"Unknown feature for counterfactuals: {feature}")

        rows = []
        for item in entries:
            X = self._sample_rows(item["X_test"])
            model = item["model"]
            fold = item.get("outer_fold")
            preds = _predict(model, X)
            delta = desired_delta
            if delta is None:
                pred_std = float(np.std(preds))
                delta = pred_std * 0.25 if pred_std > 0 else 0.01
            instance_idx = np.linspace(0, len(X) - 1, min(n_instances, len(X)), dtype=int)
            feature_grids = {}
            for idx in feature_idx:
                lo, hi = np.nanpercentile(X[:, idx], [5, 95])
                if np.isfinite(lo) and np.isfinite(hi) and lo != hi:
                    feature_grids[idx] = np.linspace(lo, hi, grid_size)

            for row_idx in instance_idx:
                x0 = X[row_idx].copy()
                base_pred = float(preds[row_idx])
                target = base_pred + delta if direction == "increase" else base_pred - delta
                best = None
                for idx, grid in feature_grids.items():
                    for value in grid:
                        x_cf = x0.copy()
                        x_cf[idx] = value
                        cf_pred = float(_predict(model, x_cf.reshape(1, -1))[0])
                        achieved = cf_pred >= target if direction == "increase" else cf_pred <= target
                        if not achieved:
                            continue
                        change = abs(float(value - x0[idx]))
                        candidate = {
                            "model": model_name,
                            "outer_fold": fold,
                            "instance_index": int(row_idx),
                            "feature": cleaned_names[idx],
                            "feature_index": int(idx),
                            "original_value": float(x0[idx]),
                            "counterfactual_value": float(value),
                            "absolute_change": change,
                            "original_prediction": base_pred,
                            "counterfactual_prediction": cf_pred,
                            "target_prediction": float(target),
                            "direction": direction,
                        }
                        if best is None or candidate["absolute_change"] < best["absolute_change"]:
                            best = candidate
                if best is not None:
                    rows.append(best)
        return pd.DataFrame(rows)

    def save(
        self,
        *,
        out_dir: str = "outputs/interpretability",
        models: list[str] | None = None,
        max_features: int = 8,
        n_counterfactuals: int = 5,
    ) -> dict[str, str]:
        out_path = Path(out_dir)
        fig_dir = out_path / "figures"
        out_path.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)

        selected = models or self.available_models()
        pdp_frames = []
        cf_frames = []
        for model_name in selected:
            pdp = self.partial_dependence(model_name, max_features=max_features)
            cf = self.counterfactuals(model_name, n_instances=n_counterfactuals)
            if not pdp.empty:
                pdp_frames.append(pdp)
                self._plot_pdp(pdp, fig_dir / f"partial_dependence_{model_name.lower()}.png")
            if not cf.empty:
                cf_frames.append(cf)

        pdp_all = pd.concat(pdp_frames, ignore_index=True) if pdp_frames else pd.DataFrame()
        cf_all = pd.concat(cf_frames, ignore_index=True) if cf_frames else pd.DataFrame()
        pdp_path = out_path / "partial_dependence.csv"
        cf_path = out_path / "counterfactuals.csv"
        pdp_all.to_csv(pdp_path, index=False)
        cf_all.to_csv(cf_path, index=False)
        return {"partial_dependence": str(pdp_path), "counterfactuals": str(cf_path)}

    def _plot_pdp(self, pdp: pd.DataFrame, path: Path) -> None:
        import matplotlib.pyplot as plt

        features = list(pdp["feature"].drop_duplicates())
        n = len(features)
        fig, axes = plt.subplots(n, 1, figsize=(8, max(3, 2.4 * n)), squeeze=False)
        for ax, feature in zip(axes[:, 0], features):
            data = (
                pdp[pdp["feature"].eq(feature)]
                .groupby("grid_value", as_index=False)["prediction_mean"]
                .mean()
                .sort_values("grid_value")
            )
            ax.plot(data["grid_value"], data["prediction_mean"], color="#3b6ea8")
            ax.set_title(feature)
            ax.set_xlabel("Feature value")
            ax.set_ylabel("Mean prediction")
            ax.grid(linestyle="--", alpha=0.35)
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
