from dataclasses import dataclass, field
from typing import List, Optional

@dataclass(frozen=True)
class FeatureSchema:
    target: str
    num_cols: List[str]
    cat_cols: List[str]

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
    metric_name: str = "r2"
    metric_opt: str = "maximize"  # or "maximize"

    # Residual Stacking
    residuals: dict = field(default_factory=lambda: {
        "XGBoost": [
            {"kind": "Quantile", "params": {"quantile": 0.9}},
            {"kind": "ElasticNet"}
        ],
        "RandomForest": [
            {"kind": "Huber", "params": {"epsilon": 1.5}},
            {"kind": "ElasticNet"}
        ]
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
        if self.metric_opt is None:
            metric = self.metric_name.lower()
            if metric in {"r2", "r^2", "negmse"}:
                self.metric_opt = "maximize"
            else:
                # MAE, RMSE, MSE, MedAE, etc.
                self.metric_opt = "minimize"

        # --- validate ---
        if self.metric_opt not in ("minimize", "maximize"):
            raise ValueError(
                f"metric_opt must be 'minimize' or 'maximize', got {self.metric_opt}"
            )
