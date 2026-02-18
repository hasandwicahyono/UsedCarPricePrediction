import json
from typing import Optional, List
import pandas as pd
import joblib
from pathlib import Path

import numpy as np
from collections import Counter

from .config import FeatureSchema, ExperimentConfig
from .tuning import NestedCVRunner
from .factories import build_model, build_preprocessor, get_preprocess_policy
from .policies import DEFAULT_PREPROCESS_POLICY
from .specs import PreprocessSpec
from .evaluation import paired_tests, significance_matrix, DefaultEvaluator
from .shap_analysis import ShapAnalyzer
from .data_io import DataReadConfig, read_csv_folder, coerce_dtypes, basic_clean
from .plot_manager import PlotManager
from .utils import model_label


def aggregate_hyperparams(param_dicts):
    """
    Aggregate hyperparameters across outer folds.
    - numeric → median
    - categorical → mode
    """
    if not param_dicts:
        return {}

    aggregated = {}
    keys = sorted({k for d in param_dicts for k in d.keys()})

    for k in keys:
        values = []
        all_numeric = True
        all_int = True

        for p in param_dicts:
            if k not in p:
                continue
            v = v.item() if isinstance(v, np.generic) else v # _to_builtin_scalar(p[k])
            if isinstance(v, list):
                v = tuple(v)
            values.append(v)

            if not isinstance(v, (int, float)) and not isinstance(v, bool): #_is_numeric(v):
                all_numeric = False
                all_int = False
            elif not isinstance(v, int):
                all_int = False

        if not values:
            continue

        if all_numeric:
            agg = float(np.median(np.asarray(values, dtype=float)))
            if all_int:
                agg = int(round(agg))
            aggregated[k] = agg
        else:
            mode_val = Counter(values).most_common(1)[0][0]
            if isinstance(mode_val, tuple):
                mode_val = list(mode_val)
            aggregated[k] = mode_val

    return aggregated


def aggregate_residual_cfg(residual_cfgs):
    """
    Aggregate residual configs across folds by mode on the full config dict.
    """
    if not residual_cfgs:
        return None
    serialized = [
        json.dumps(cfg, sort_keys=True) if cfg is not None else None
        for cfg in residual_cfgs
    ]
    mode_val = Counter(serialized).most_common(1)[0][0]
    if mode_val is None:
        return None
    return json.loads(mode_val)


class EnsembleModel:
    def __init__(self, models: list):
        self.models = models

    def predict(self, X):
        preds = [m.predict(X) for m in self.models]
        return np.mean(np.vstack(preds), axis=0)


class ExperimentFacade:
    def __init__(
        self,
        df: pd.DataFrame,
        schema: FeatureSchema,
        cfg: ExperimentConfig,
        model_names: List[str],
        hparam_json: Optional[str] = None,
        evaluator: Optional[DefaultEvaluator] = None,
    ):
        self.runner = NestedCVRunner(
            df, 
            schema, 
            cfg, 
            model_names, 
            hparam_json=hparam_json
        )
        self.df = df
        self.cfg = cfg
        self.schema = schema
        self.model_names = list(model_names)
        self.evaluator = evaluator or DefaultEvaluator()

    def run(self):
        self.results_ = self.runner.run()
        self._save_best_hyperparams()
        self._refit_and_save_models()
        self._save_best_model_artifact()
        return self.results_

    def _save_best_model_artifact(self, out_dir="outputs/artifacts"):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # select best by metric
        metric = self.cfg.metric_name.upper()
        ascending = self.cfg.metric_opt == "minimize"

        summary = (
            self.results_
            .groupby("model")[metric]
            .mean()
            .sort_values(ascending=ascending)
        )

        best_model = summary.index[0]

        artifact = {
            "best_model": best_model,
            "metric": self.cfg.metric_name,
            "metric_opt": self.cfg.metric_opt,
            "residual_config": self.cfg.residuals,
        }

        with open(out_dir / "best_model.json", "w") as f:
            json.dump(artifact, f, indent=2)

        return artifact

    def _save_best_hyperparams(self, out_dir="outputs/hyperparameters"):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        payload = {}
        out_path = out_dir / "best_hyperparameters.json"
        if out_path.exists():
            with open(out_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

        residuals_per_model = {}
        for rec in self.runner.best_params_records_:
            residuals_per_model.setdefault(rec["model"], []).append(rec.get("residual_cfg"))

        for model_name, params_per_fold in self.runner.best_params_.items():
            payload[model_name] = {
                "aggregated": aggregate_hyperparams(params_per_fold),
                "per_outer_fold": params_per_fold,
                "n_outer_folds": len(params_per_fold),
                "metric_optimized": self.cfg.metric_opt,
                "seed": self.cfg.seed,
                "residual_cfg": aggregate_residual_cfg(residuals_per_model.get(model_name, [])),
                "residual_cfg_per_outer_fold": residuals_per_model.get(model_name, []),
            }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"[saved] {out_path.resolve()}")

    def _refit_and_save_models(self, out_dir="outputs/models", hyperparams_dir="outputs/hyperparameters", top_k: int = 3):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # load aggregated hyperparams
        with open(Path(hyperparams_dir) / "best_hyperparameters.json") as f:
            best_params = json.load(f)

        X = self.df[self.schema.num_cols + self.schema.cat_cols]
        y = self.df[self.schema.target]

        metric_col = self.cfg.metric_name.upper()
        metric_opt = self.cfg.metric_opt
        score_lookup = {}
        if self.results_ is not None and metric_col in self.results_.columns:
            for row in self.results_[["model", "outer_fold", metric_col]].itertuples(index=False, name=None):
                key = (row[0], row[1])
                if key not in score_lookup:
                    score_lookup[key] = float(row[2]) if pd.notna(row[2]) else None
        records_by_model = {}
        for rec in self.runner.best_params_records_:
            m = rec.get("model")
            if m is None:
                continue
            records_by_model.setdefault(m, []).append(rec)
        pre_cache = {}

        for model_name in self.model_names:
            print(f"[final fit] {model_name}")

            # build preprocessing
            policy = get_preprocess_policy(model_name, DEFAULT_PREPROCESS_POLICY)
            pre_key = (policy["cat_encoding"], bool(policy["use_feature_selection"]))
            if pre_key in pre_cache:
                pre, Xp = pre_cache[pre_key]
            else:
                pre = build_preprocessor(
                    self.schema,
                    PreprocessSpec(
                        cat_encoding=policy["cat_encoding"],
                        use_feature_selection=policy["use_feature_selection"],
                        seed=self.cfg.seed,
                    ),
                )
                Xp = pre.fit_transform(X, y)
                pre_cache[pre_key] = (pre, Xp)

            # build model with BEST params
            params = best_params[model_name]["aggregated"]
            if model_name == "RandomForest" and params.get("bootstrap") is False:
                params["max_samples"] = None
            residual_cfg = best_params[model_name].get("residual_cfg")
            model = build_model(
                model_name,
                seed=self.cfg.seed,
                params=params,
                residual_cfgs=[residual_cfg] if residual_cfg else None,
            )

            model.fit(Xp, y)

            # save both
            if model_name == "NeuralNetwork":
                model.model.save(out_dir / f"{model_name}.keras")
            else:   
                label = model_name
                if residual_cfg is not None and isinstance(residual_cfg, dict):
                    kind = residual_cfg.get("kind")
                    if kind:
                        label = f"{model_name}+{kind}"
                joblib.dump(model, out_dir / f"{label}.joblib")
            
            joblib.dump(pre, out_dir / f"{model_name}_preprocessor.joblib")

            print(f"[saved] {model_name}")

            # also save all residual variants (if configured)
            residual_cfgs = self.cfg.residuals.get(model_name, []) if isinstance(self.cfg.residuals, dict) else []
            seen = set()
            for rc in residual_cfgs:
                if not rc or rc.get("kind") in (None, "None"):
                    continue
                key = json.dumps(rc, sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)

                r_model = build_model(
                    model_name,
                    seed=self.cfg.seed,
                    params=params,
                    residual_cfgs=[rc],
                )
                r_model.fit(Xp, y)
                label = f"{model_name}+{rc.get('kind')}"
                joblib.dump(r_model, out_dir / f"{label}.joblib")
                print(f"[saved] {label}")

            # optional top-k ensemble using per-fold best params
            if top_k and model_name in best_params:
                records = records_by_model.get(model_name, [])
                if records:
                    scored = []
                    for r in records:
                        label = model_label(model_name, r.get("residual_cfg"))
                        ofold = r.get("outer_fold")
                        score = score_lookup.get((label, ofold))
                        scored.append((score, r))

                    # sort by score, fallback to original order if score is None
                    if metric_opt == "minimize":
                        scored.sort(key=lambda x: (x[0] is None, x[0]))
                    else:
                        scored.sort(key=lambda x: (x[0] is None, -(x[0] if x[0] is not None else 0)))

                    top = [r for _, r in scored[: max(1, int(top_k))]]
                    models = []
                    for r in top:
                        m = build_model(
                            model_name,
                            seed=self.cfg.seed,
                            params=r.get("params", {}),
                            residual_cfgs=[r.get("residual_cfg")] if r.get("residual_cfg") else None,
                        )
                        m.fit(Xp, y)
                        models.append(m)

                    if models:
                        ensemble = EnsembleModel(models)
                        joblib.dump(ensemble, out_dir / f"{model_name}_ensemble_top{len(models)}.joblib")
                        print(f"[saved] {model_name} ensemble top{len(models)}")

    def summary(self):
        return self.evaluator.summary(self.runner.results_)

    def significance(self, metric="MAE", baseline="RandomForest", models: list[str] | None = None):
        return paired_tests(self.runner.results_, metric=metric, baseline=baseline, models=models)

    def significance_matrix(self, metric="MAE", models: list[str] | None = None):
        return self.evaluator.significance_matrix(self.runner.results_, metric=metric, models=models)

    def shap(self, plot_dir: str = "outputs/figures/shap", models: list[str] | None = None):
        pm = PlotManager(base_dir=plot_dir)
        return ShapAnalyzer(
            self.runner.shap_store_,
            background_size=self.cfg.shap_background_size,
            max_eval_samples=self.cfg.shap_max_eval_samples,
            seed=self.cfg.seed,
            plot_manager=pm,
            models=models
        )

    def save_best_params(self, out_path: str = "outputs/best_params"):
        out_dir = Path(out_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "outer_folds": self.cfg.outer_folds,
                "inner_folds": self.cfg.inner_folds,
                "n_trials": self.cfg.n_trials,
                "log_target": self.cfg.log_target
            },
            "best_params": self.runner.best_params_records_
        }
        out_path = out_dir / "best_params.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"[saved] {out_path.resolve()}")
    
    @classmethod
    def from_folder(
        cls,
        data_cfg: DataReadConfig,
        target: str,
        cfg: ExperimentConfig,
        model_names: List[str],
        hparam_json: Optional[str] = None,
        dropna_target: bool = True
    ):
        df = read_csv_folder(data_cfg)
        df = basic_clean(df, target=target, dropna_target=dropna_target)
        
        schema = FeatureSchema.from_dataframe(
            df,
            target=target,
            normalize=True,
        )
        print(f"[schema]\n  numerical cols: {schema.num_cols}\n  categorical cols: {schema.cat_cols}\n  target col: {[target]}\n  mapping: {schema.mapping}")
        df = coerce_dtypes(df, numeric_cols=schema.num_cols, categorical_cols=schema.cat_cols)
        return cls(df=df, schema=schema, cfg=cfg, model_names=model_names, hparam_json=hparam_json)
