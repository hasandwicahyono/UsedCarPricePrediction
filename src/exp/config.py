from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from .schema_utils import sanitize_columns, infer_schema
import pandas as pd

@dataclass(frozen=True)
class FeatureSchema:
    target: str
    num_cols: List[str]
    cat_cols: List[str]
    mapping: Optional[dict] = None

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        target: str,
        normalize: bool = True,
    ) -> "FeatureSchema":
        """
        Auto-detect schema from raw dataframe.
        """
        mapping = None
        if normalize:
            df, mapping = sanitize_columns(df)

        target, num_cols, cat_cols = infer_schema(df, target)
        return cls(target=target, num_cols=num_cols, cat_cols=cat_cols, mapping=mapping)

@dataclass
class ExperimentConfig:
    # CV
    seed: int = 42

    # Optuna
    optuna_seed: int = 123
    outer_folds: int = 5
    inner_folds: int = 3
    n_trials: int = 40
    timeout_sec: Optional[int] = None

    # Target
    log_target: bool = True
    
    # Early Stopping
    early_stopping_patience: int = 5

    # Metric
    metric_name: str = "mae"         # r2, mse, negmse, rmse, mae, medae 
    metric_opt: str = "minimize"    # "minimize" or "maximize", auto-inferred if None
    report_metrics: List[str] = field(default_factory=lambda: ["r2", "mae", "medae", "mse", "rmse"])

    # Residual Stacking
    residuals: Dict[str, List[Dict[str, Any]]] = field(default_factory=lambda: {
        "DecisionTree": [
            {"kind": "None", "params": {}},
            {"kind": "ElasticNet", "params": {"alpha": 0.001, "l1_ratio": 0.5}},
            #{"kind": "Quantile", "params": {"quantile": 0.75, "alpha": 0.001}},
            {"kind": "Huber", "params": {"epsilon": 1.35}},
            {
                "kind": "PseudoHuber",
                "params": {
                    "n_estimators": 200,
                    "max_depth": 2,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_alpha": 0.1,
                    "reg_lambda": 1.0,
                    "tree_method": "hist",
                    "n_jobs": -1,
                },
            },
        ],
        "RandomForest": [
            {"kind": "None", "params": {}},
            {"kind": "ElasticNet", "params": {"alpha": 0.001, "l1_ratio": 0.5}},
            #{"kind": "Quantile", "params": {"quantile": 0.75, "alpha": 0.001}},
            {"kind": "Huber", "params": {"epsilon": 1.35}},
            {
                "kind": "PseudoHuber",
                "params": {
                    "n_estimators": 200,
                    "max_depth": 2,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_alpha": 0.1,
                    "reg_lambda": 1.0,
                    "tree_method": "hist",
                    "n_jobs": -1,
                },
            },
        ],
        "XGBoost": [
            {"kind": "Huber", "params": {"epsilon": 1.35}},
            {
                "kind": "PseudoHuber",
                "params": {
                    "n_estimators": 200,
                    "max_depth": 2,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_alpha": 0.1,
                    "reg_lambda": 1.0,
                    "tree_method": "hist",
                    "n_jobs": -1,
                },
            },
        ],
    })

    # Pruning
    pruner_startup_trials: int = 10
    pruner_warmup_steps: int = 1
    pruner_interval_steps: int = 1

    # Caching
    preprocessing_cache: bool = True

    # Kernel SHAP controls (SVR/NN)
    shap_background_size: int = 100
    shap_max_eval_samples: int = 300

    def __post_init__(self):
        # --- auto-derive optimization direction ---
        metric = self.metric_name.lower()
        expected = "maximize" if metric in {"r2", "r^2", "negmse", "nmse"} else "minimize"
        if self.metric_opt is None:
            self.metric_opt = expected
        elif self.metric_opt != expected:
            raise ValueError(
                f"metric_opt='{self.metric_opt}' conflicts with metric_name='{self.metric_name}' "
                f"(expected '{expected}')"
            )

        # --- validate ---
        if self.metric_opt not in ("minimize", "maximize"):
            raise ValueError(
                f"metric_opt must be 'minimize' or 'maximize', got {self.metric_opt}"
            )
