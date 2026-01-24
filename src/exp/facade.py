import json
from typing import Optional, List
import pandas as pd
import joblib
from pathlib import Path

import numpy as np
from collections import Counter

from .config import FeatureSchema, ExperimentConfig
from .tuning import NestedCVRunner
from .models import ModelFactory
from .evaluation import summarize_mean_std, paired_tests, significance_matrix
from .shap_analysis import ShapAnalyzer
from .data_io import DataReadConfig, read_csv_folder, coerce_dtypes, basic_clean
from .plot_manager import PlotManager
from .preprocess import PreprocessorBuilder


def aggregate_hyperparams(param_dicts):
    """
    Aggregate hyperparameters across outer folds.
    - numeric → median
    - categorical → mode
    """
    aggregated = {}
    keys = param_dicts[0].keys()

    for k in keys:
        values = [p[k] for p in param_dicts]
        # make list-like hashable for mode calculation
        if any(isinstance(v, list) for v in values):
            values = [tuple(v) if isinstance(v, list) else v for v in values]

        if isinstance(values[0], (int, float)):
            agg = float(np.median(values))
            if isinstance(values[0], int):
                agg = int(round(agg))
            aggregated[k] = agg
        else:
            mode_val = Counter(values).most_common(1)[0][0]
            if isinstance(mode_val, tuple):
                mode_val = list(mode_val)
            aggregated[k] = mode_val

    return aggregated


class ExperimentFacade:
    def __init__(
        self,
        df: pd.DataFrame,
        schema: FeatureSchema,
        cfg: ExperimentConfig,
        model_names: List[str],
        hparam_json: Optional[str] = None
    ):
        self.runner = NestedCVRunner(df, schema, cfg, model_names, hparam_json=hparam_json)
        self.df = df
        self.cfg = cfg
        self.schema = schema
        self.model_names = list(model_names)

    def run(self):
        self.runner.run()
        self._save_best_hyperparams()
        self._refit_and_save_models()
        self.results_ = self.runner.results_
        return self.results_

    def _save_best_hyperparams(self, out_dir="outputs/hyperparameters"):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        payload = {}
        out_path = out_dir / "best_hyperparameters.json"
        if out_path.exists():
            with open(out_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

        for model_name, params_per_fold in self.runner.best_params_.items():
            payload[model_name] = {
                "aggregated": aggregate_hyperparams(params_per_fold),
                "per_outer_fold": params_per_fold,
                "n_outer_folds": len(params_per_fold),
                "metric_optimized": self.cfg.metric_opt,
                "seed": self.cfg.seed,
            }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"[saved] {out_path.resolve()}")

    def _refit_and_save_models(self, out_dir="outputs/models", hyperparams_dir="outputs/hyperparameters"):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # load aggregated hyperparams
        with open(Path(hyperparams_dir) / "best_hyperparameters.json") as f:
            best_params = json.load(f)

        X = self.df[self.schema.num_cols + self.schema.cat_cols]
        y = self.df[self.schema.target]

        for model_name in self.model_names:
            print(f"[final fit] {model_name}")

            # build preprocessing
            pre = PreprocessorBuilder(self.schema).build()
            Xp = pre.fit_transform(X, y)

            # build model with BEST params
            params = best_params[model_name]["aggregated"]
            model = ModelFactory.create(model_name, seed=self.cfg.seed, params=params)

            model.fit(Xp, y)

            # save both
            if model_name == "NeuralNetwork":
                model.model.save(out_dir / f"{model_name}.keras")
            else:   
                joblib.dump(model, out_dir / f"{model_name}.joblib")
            
            joblib.dump(pre, out_dir / f"{model_name}_preprocessor.joblib")

            print(f"[saved] {model_name}")

    def summary(self):
        return summarize_mean_std(self.runner.results_)

    def significance(self, metric="MAE", baseline="RandomForest", models: list[str] | None = None):
        return paired_tests(self.runner.results_, metric=metric, baseline=baseline, models=models)

    def significance_matrix(self, metric="MAE"):
        return significance_matrix(self.runner.results_, metric=metric)

    def shap(self, plot_dir: str = "outputs/figures", models: list[str] | None = None):
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
        schema: FeatureSchema,
        cfg: ExperimentConfig,
        model_names: List[str],
        hparam_json: Optional[str] = None,
        dropna_target: bool = True
    ):
        df = read_csv_folder(data_cfg)
        df = basic_clean(df, target=schema.target, dropna_target=dropna_target)
        df = coerce_dtypes(df, numeric_cols=schema.num_cols, categorical_cols=schema.cat_cols)
        return cls(df=df, schema=schema, cfg=cfg, model_names=model_names, hparam_json=hparam_json)
