import os
import random
import json
from typing import Dict, Any, Optional, Tuple, List
from collections import defaultdict

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, median_absolute_error

from .config import ExperimentConfig, FeatureSchema
from .preprocess import PreprocessorBuilder
from .models import ModelFactory
from .metrics import MetricStrategy
from .metrics import make_metric

# ======================================================
# Utilities
# ======================================================

def build_monotone_constraints(feature_names: list[str]) -> str:
    """
    Define monotonicity assumptions here.
    +1  increasing
    -1  decreasing
     0  unconstrained
    """

    constraints = []

    for f in feature_names:
        if f.startswith("mileage"):
            constraints.append(-1)   # higher mileage → lower price
        elif f.startswith("year"):
            constraints.append(+1)   # newer → higher price
        elif f.startswith("engine_size"):
            constraints.append(+1)
        else:
            constraints.append(0)

    return "(" + ",".join(map(str, constraints)) + ")"


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def suggest_from_space(trial: optuna.Trial, space: Dict[str, Any]) -> Dict[str, Any]:
    params = {}
    for name, spec in space.items():
        t = spec["type"]
        if t == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif t == "float":
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=bool(spec.get("log", False))
            )
        elif t == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unsupported hyperparameter type: {t}")
    return params


# ======================================================
# Hyperparameter Space
# ======================================================

class HyperparamSpace:
    def __init__(self, json_path: Optional[str] = None):
        self.cfg = None
        if json_path:
            with open(json_path, "r", encoding="utf-8") as f:
                self.cfg = json.load(f)

    def get(self, model_name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if self.cfg is None:
            return {}, {}
        m = self.cfg["models"].get(model_name, {})
        return dict(m.get("fixed", {})), dict(m.get("search", {}))

    def global_trials(self) -> Optional[int]:
        return self.cfg.get("global", {}).get("n_trials") if self.cfg else None

    def global_timeout(self) -> Optional[int]:
        return self.cfg.get("global", {}).get("timeout_sec") if self.cfg else None


# ======================================================
# Coverage Detection (JSON-only, semantic)
# ======================================================

def extract_coverage_params_from_json(
    hparam_space: HyperparamSpace,
    model_name: str,
) -> Dict[str, list]:
    """
    Coverage-critical parameters are categorical hyperparameters
    whose choices are ALL non-numeric (structural semantics).
    """
    _, search = hparam_space.get(model_name)
    coverage = {}

    for pname, spec in search.items():
        if spec.get("type") != "categorical":
            continue

        choices = spec.get("choices", [])
        if not choices:
            continue

        if all(not isinstance(c, (int, float)) for c in choices):
            coverage[pname] = list(choices)

    return coverage


# ======================================================
# Coverage Tracking
# ======================================================

class CoverageTracker:
    def __init__(self, coverage_params: Dict[str, list]):
        self.coverage_params = coverage_params
        self.counts = {p: defaultdict(int) for p in coverage_params}

    def update(self, trial: optuna.trial.FrozenTrial):
        for p in self.coverage_params:
            if p in trial.params:
                self.counts[p][trial.params[p]] += 1

    def is_fully_covered(self, min_count: int = 1) -> bool:
        return all(
            self.counts[p][v] >= min_count
            for p, values in self.coverage_params.items()
            for v in values
        )


# ======================================================
# Stopping Policy (Pluggable, Correct)
# ======================================================

class CoverageAwareEarlyStoppingPolicy:
    """
    Stop when:
    (1) no improvement + pruned trials >= patience
    AND
    (2) all coverage-critical categorical choices evaluated
    """

    def __init__(self, patience: int, coverage_tracker: CoverageTracker, metric: MetricStrategy):
        self.patience = patience
        self.coverage = coverage_tracker
        self.metric = metric
        self.best_value = None
        self.no_improve = 0
        self.pruned = 0

    def on_trial_complete(self, study, trial):
        self.coverage.update(trial)
        if trial.value is None:
            return

        # use study.best_value but compare direction-aware
        val = trial.value
        if self.best_value is None or self.metric.is_better(val, self.best_value):
            self.best_value = val
            self.no_improve = 0
            self.pruned = 0
        else:
            self.no_improve += 1
            
    def on_trial_pruned(self):
        self.pruned += 1

    def should_stop(self) -> bool:
        return (
            (self.no_improve + self.pruned) >= self.patience
            and self.coverage.is_fully_covered()
        )


class OptunaStoppingCallback:
    def __init__(self, policy: CoverageAwareEarlyStoppingPolicy):
        self.policy = policy

    def __call__(self, study, trial):
        self.policy.on_trial_complete(study, trial)
        if self.policy.should_stop():
            study.stop()


# ======================================================
# Nested CV Runner
# ======================================================

class NestedCVRunner:
    def __init__(
        self,
        df: pd.DataFrame,
        schema: FeatureSchema,
        cfg: ExperimentConfig,
        model_names: List[str],
        hparam_json: Optional[str] = None,
    ):
        self.df = df.copy()
        self.schema = schema
        self.cfg = cfg
        self.model_names = model_names
        self.space = HyperparamSpace(hparam_json)

        if self.space.global_trials() is not None:
            self.cfg.n_trials = int(self.space.global_trials())
        if self.space.global_timeout() is not None:
            self.cfg.timeout_sec = int(self.space.global_timeout())

        self._prepare_xy()

        self.results_ = None
        self.best_params_records_ = []
        self.best_params_ = {}
        self.shap_store_ = []
        self.feature_stability_ = []

        self.metric = make_metric(self.cfg.metric_name)

        valid_models = set(ModelFactory.MAP.keys())
        unknown = set(self.model_names) - valid_models
        if unknown:
            raise ValueError(f"Unknown model names: {unknown}")

    def _prepare_xy(self):
        y = self.df[self.schema.target].astype(float).values
        self.y_price = y
        self.y_log = np.log(y.clip(min=1.0)) if self.cfg.log_target else y
        self.X = self.df[self.schema.num_cols + self.schema.cat_cols].copy()

    def _create_study(self, seed: int, study_name: Optional[str] = None) -> optuna.Study:
        sampler = optuna.samplers.TPESampler(seed=seed)
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=self.cfg.pruner_startup_trials,
            n_warmup_steps=self.cfg.pruner_warmup_steps,
            interval_steps=self.cfg.pruner_interval_steps,
        )
        return optuna.create_study(direction=self.metric.direction, #self.cfg.metric_opt, 
                                   sampler=sampler, 
                                   pruner=pruner,
                                   study_name=study_name)

    def _inner_objective(
        self,
        trial: optuna.Trial,
        model_name: str,
        X_train: pd.DataFrame,
        y_train_log: np.ndarray,
        y_train_price: np.ndarray,
        seed: int,
        pre_cache: dict,
        stopping_policy: CoverageAwareEarlyStoppingPolicy,
    ) -> float:

        fixed, search = self.space.get(model_name)
        params = {**fixed, **suggest_from_space(trial, search)}

        inner_cv = KFold(
            n_splits=self.cfg.inner_folds,
            shuffle=True,
            random_state=seed,
        )

        builder = PreprocessorBuilder(self.schema)
        default_policy = dict(cat_encoding="onehot", use_feature_selection=False)
        scores = []
        for step, (tr, va) in enumerate(inner_cv.split(X_train)):
            Xtr_raw = X_train.iloc[tr]
            Xva_raw = X_train.iloc[va]
            ytr = y_train_log[tr]
            yva = y_train_log[va]
            yva_price = y_train_price[va]

            # key = hash(tr.tobytes())
            key = (model_name, seed, tuple(tr))
            if self.cfg.preprocessing_cache and key in pre_cache:
                pre, Xtr, Xva = pre_cache[key]
            else:
                policy = getattr(ModelFactory.MAP[model_name], "preprocess_policy", default_policy)
                pre = builder.build(
                    cat_encoding=policy["cat_encoding"],
                    use_feature_selection=policy["use_feature_selection"],
                    seed=seed
                )
                Xtr = pre.fit_transform(Xtr_raw, ytr)
                Xva = pre.transform(Xva_raw)
                if self.cfg.preprocessing_cache:
                    pre_cache[key] = (pre, Xtr, Xva)

            model = ModelFactory.create(
                model_name, 
                seed=seed, 
                params=params,
                residual_cfg=self.cfg.residuals.get(model_name)
            )
            model.fit(Xtr, ytr, Xva, yva)

            pred_log = model.predict(Xva)
            pred_price = np.exp(pred_log) if self.cfg.log_target else pred_log
            #pred_price = np.exp(pred_log)

            # Guard against NaN/inf in predictions or targets
            if not np.isfinite(pred_price).all() or not np.isfinite(yva_price).all():
                return float("-inf") if self.metric.direction == "maximize" else float("inf")

            mask = np.isfinite(pred_price) & np.isfinite(yva_price)
            if mask.sum() < 2:
                return float("-inf") if self.metric.direction == "maximize" else float("inf")

            score = self.metric.compute(yva_price[mask], pred_price[mask])
            scores.append(score)

            # metric-aware pruning: only for safe loss-like metrics
            if self.metric.supports_pruning():
                trial.report(self.metric.as_loss(score), step)
                if trial.should_prune():
                    stopping_policy.on_trial_pruned()
                    raise optuna.TrialPruned()

        return float(np.mean(scores)) #float(np.mean(maes))

    def run(self):
        set_seed(self.cfg.seed)

        model_names = list(self.model_names)
        if not model_names:
            raise ValueError("No models specified for experiment.")

        outer_cv = KFold(
            n_splits=self.cfg.outer_folds,
            shuffle=True,
            random_state=self.cfg.seed,
        )

        rows = []
        default_policy = dict(cat_encoding="onehot", use_feature_selection=False)
        for ofold, (tr, te) in enumerate(outer_cv.split(self.X), start=1):
            Xtr_raw = self.X.iloc[tr]
            Xte_raw = self.X.iloc[te]
            ytr_log = self.y_log[tr]
            ytr_price = self.y_price[tr]
            yte_price = self.y_price[te]

            fold_seed = self.cfg.seed + ofold

            for model_name in self.model_names:
                pre_cache = {}
                study_name_custom = f"{model_name}_OuterFold_{ofold}"
                study = self._create_study(seed=self.cfg.optuna_seed + ofold, 
                                           study_name=study_name_custom)

                coverage_params = extract_coverage_params_from_json(self.space, model_name)
                coverage_tracker = CoverageTracker(coverage_params)
                stopping_policy = CoverageAwareEarlyStoppingPolicy(
                    patience=self.cfg.early_stopping_patience,
                    coverage_tracker=coverage_tracker,
                    metric=self.metric,
                )

                callback = OptunaStoppingCallback(stopping_policy)

                study.optimize(
                    lambda t: self._inner_objective(
                        t,
                        model_name,
                        Xtr_raw,
                        ytr_log,
                        ytr_price,
                        fold_seed,
                        pre_cache,
                        stopping_policy,
                    ),
                    n_trials=self.cfg.n_trials,
                    timeout=self.cfg.timeout_sec,
                    callbacks=[callback],
                )

                fixed, _ = self.space.get(model_name)
                complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                # handle fallback or raise a clearer error
                if not complete_trials:
                    raise RuntimeError(f"No completed trials for model {model_name} in outer fold {ofold}.")
                
                best_params = {**fixed, **study.best_params}

                self.best_params_records_.append(
                    {"outer_fold": ofold, "model": model_name, "params": best_params}
                )
                self.best_params_.setdefault(model_name, []).append(best_params)

                ## --- retrain on full outer train ---
                policy = getattr(ModelFactory.MAP[model_name], "preprocess_policy", default_policy)

                pre = PreprocessorBuilder(self.schema).build(
                    cat_encoding=policy["cat_encoding"],
                    use_feature_selection=policy["use_feature_selection"],
                    seed=fold_seed
                )
                Xtr = pre.fit_transform(Xtr_raw, ytr_log)
                Xte = pre.transform(Xte_raw)
                
                feature_names = pre.get_feature_names_out()

                if model_name == "XGBoost":
                    best_params["monotone_constraints"] = build_monotone_constraints(
                        list(feature_names)
                    )

                model = ModelFactory.create(
                    model_name, 
                    seed=fold_seed, 
                    params=best_params,
                    residual_cfg=self.cfg.residuals.get(model_name)
                )
                model.fit(Xtr, ytr_log, None, None)

                num_pipe = pre.named_transformers_.get("num")
                
                num_feature_names, mask, coefs = None, None, None
                
                if num_pipe is not None:
                    num_feature_names = num_pipe.get_feature_names_out()

                    if "enet_fs" in num_pipe.named_steps:
                        selector = num_pipe.named_steps["enet_fs"]
                        mask = selector.get_support()
                        est = selector.estimator_
                        if hasattr(est, "coef_"):
                            coefs = np.asarray(est.coef_)
                    
                self.feature_stability_.append(
                    dict(
                        model_name=model_name,
                        outer_fold=ofold,
                        feature_names=num_feature_names,
                        selection_mask=mask,
                        coefficients=coefs,
                    )
                )
                pred = model.predict(Xte)
                pred_price = np.exp(pred) if self.cfg.log_target else pred

                r2 = r2_score(yte_price, pred_price)
                mae = mean_absolute_error(yte_price, pred_price)
                medae = median_absolute_error(yte_price, pred_price)
                mse = mean_squared_error(yte_price, pred_price)
                rmse = np.sqrt(mse)

                rows.append(
                    dict(
                        outer_fold=ofold,
                        model=model_name,
                        R2=r2,
                        MAE=mae,
                        MedAE=medae,
                        MSE=mse,
                        RMSE=rmse,
                    )
                )

                shap_model = model
                shap_model_type = model.model_type
                if hasattr(model, "base_model"):
                    shap_model = model.base_model
                    shap_model_type = model.base_model.model_type

                self.shap_store_.append(
                    dict(
                        model_name=model_name,
                        model_type=shap_model_type,
                        model=getattr(shap_model, "model", shap_model),
                        X_test=Xte,
                        feature_names=pre.get_feature_names_out(),
                    )
                )

        self.results_ = pd.DataFrame(rows)
        return self.results_
