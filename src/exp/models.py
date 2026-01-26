from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np

# =======================
# Base ML imports
# =======================
from sklearn.linear_model import (
    LinearRegression,
    ElasticNet,
    QuantileRegressor,
    HuberRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping

# =======================
# Deep learning
# =======================
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, AdamW, RMSprop, SGD

from .metrics import make_metric

# =====================================================
# Base Strategy Registries
# =====================================================

class ModelStrategy(ABC):
    registry: Dict[str, type] = {}
    model_type: str
    preprocess_policy: dict = {}
    explain_policy: str = "tree"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "model_type"):
            ModelStrategy.registry[cls.model_type] = cls

    @abstractmethod
    def fit(self, Xtr, ytr, Xva=None, yva=None): ...

    @abstractmethod
    def predict(self, Xte) -> np.ndarray: ...


class ResidualStrategy(ABC):
    registry: Dict[str, type] = {}
    model_type: str

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "model_type"):
            ResidualStrategy.registry[cls.model_type] = cls

    @abstractmethod
    def fit(self, X, residuals, X_val=None, residuals_val=None): ...

    @abstractmethod
    def predict(self, X) -> np.ndarray: ...


# =====================================================
# Base Models
# =====================================================

class LinearRegressionStrategy(ModelStrategy):
    model_type = "LinearRegression"
    preprocess_policy = dict(cat_encoding="target", use_feature_selection=True)
    explain_policy = "linear"

    def __init__(self, seed, params):
        self.model = LinearRegression(**params)

    def fit(self, Xtr, ytr, Xva=None, yva=None):
        self.model.fit(Xtr, ytr)

    def predict(self, X):
        return self.model.predict(X)


class RandomForestStrategy(ModelStrategy):
    model_type = "RandomForest"
    preprocess_policy = dict(cat_encoding="onehot", use_feature_selection=False)
    explain_policy = "tree"


    def __init__(self, seed, params):
        self.model = RandomForestRegressor(random_state=seed, **params)

    def fit(self, Xtr, ytr, Xva=None, yva=None):
        self.model.fit(Xtr, ytr)

    def predict(self, X):
        return self.model.predict(X)


class DecisionTreeStrategy(ModelStrategy):
    model_type = "DecisionTree"
    preprocess_policy = dict(cat_encoding="onehot", use_feature_selection=False)
    explain_policy = "tree"


    def __init__(self, seed: int, params: Dict[str, Any]):
        self.model = DecisionTreeRegressor(random_state=seed, **params)

    def fit(self, Xtr, ytr, Xva=None, yva=None):
        self.model.fit(Xtr, ytr)

    def predict(self, X):
        return self.model.predict(X)
    

class SVRStrategy(ModelStrategy):
    model_type = "SVR"
    preprocess_policy = dict(cat_encoding="target", use_feature_selection=True)
    explain_policy = "kernel"

    def __init__(self, seed: int, params: Dict[str, Any]):
        self.model = SVR(**params)

    def fit(self, Xtr, ytr, Xva=None, yva=None):
        self.model.fit(Xtr, ytr)

    def predict(self, X):
        return self.model.predict(X)
    

# =====================================================
# Neural Network
# =====================================================

class KerasMLPStrategy(ModelStrategy):
    model_type = "NeuralNetwork"
    preprocess_policy = dict(cat_encoding="target", use_feature_selection=True)
    explain_policy = "kernel"

    def __init__(self, seed: int, params: dict):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        self.params = params
        self.metric = make_metric(params.get("metric_name", "MAE"))
        self.model = None

    def _make_optimizer(self, name, lr, momentum, weight_decay):
        return {
            "adam": Adam(lr),
            "adamw": AdamW(lr, weight_decay=weight_decay),
            "rmsprop": RMSprop(lr, momentum=momentum),
            "sgd": SGD(lr, momentum=momentum),
        }[name.lower()]

    def _build(self, dim: int):
        p = self.params
        model = Sequential()
        for i in range(p.get("n_layers", 2)):
            model.add(Dense(p[f"units_layer_{i+1}"], activation=p.get("activation", "relu")))
        model.add(Dense(1))
        model.compile(
            optimizer=self._make_optimizer(
                p.get("optimizer", "adam"),
                p.get("learning_rate", 0.001),
                p.get("momentum", 0.0),
                p.get("weight_decay", 0.0),
            ),
            loss=p.get("loss", "mse"),
        )
        return model

    def fit(self, Xtr, ytr, Xva=None, yva=None):
        self.model = self._build(Xtr.shape[1])
        self.model.fit(
            Xtr, ytr,
            validation_data=(Xva, yva) if Xva is not None else None,
            epochs=100,
            batch_size=256,
            verbose=0,
        )

    def predict(self, X):
        return self.model.predict(X, verbose=0).ravel()

class XGBoostStrategy(ModelStrategy):
    model_type = "XGBoost"
    preprocess_policy = dict(cat_encoding="onehot", use_feature_selection=False)
    explain_policy = "tree"

    def __init__(self, seed, params):
        self.early_stopping_rounds = params.pop("early_stopping_rounds", None)
        self.model = XGBRegressor(random_state=seed, **params)

    def fit(self, Xtr, ytr, Xva=None, yva=None):
        if Xva is not None and self.early_stopping_rounds:
            try:
                self.model.fit(
                    Xtr, ytr,
                    eval_set=[(Xva, yva)],
                    early_stopping_rounds=self.early_stopping_rounds,
                    verbose=False,
                )
                return
            except TypeError:
                try:
                    self.model.fit(
                        Xtr, ytr,
                        eval_set=[(Xva, yva)],
                        callbacks=[EarlyStopping(rounds=self.early_stopping_rounds)],
                        verbose=False,
                    )
                    return
                except TypeError:
                    try:
                        self.model.fit(
                            Xtr, ytr,
                            eval_set=[(Xva, yva)],
                            verbose=False,
                        )
                        return
                    except TypeError:
                        pass
        self.model.fit(Xtr, ytr)

    def predict(self, X):
        return self.model.predict(X)


# =====================================================
# Residual Models
# =====================================================

class ElasticNetResidual(ResidualStrategy):
    model_type = "ElasticNet"

    def __init__(self, seed=42, **params: Any):#alpha=0.001, l1_ratio=0.5):
        self.model = ElasticNet(alpha=params.get("alpha", 0.001), l1_ratio=params.get("l1_ratio", 0.5), random_state=seed)

    def fit(self, X, residuals, X_val=None, residuals_val=None):
        self.model.fit(X, residuals)

    def predict(self, X):
        return self.model.predict(X)


class QuantileResidual(ResidualStrategy):
    model_type = "Quantile"

    def __init__(self, seed: int = 42, **params: Any):#quantile=0.75, alpha=0.001):
        self.model = QuantileRegressor(
            quantile=params.get("quantile", 0.75), alpha=params.get("alpha", 0.001), solver="highs"
        )

    def fit(self, X, residuals, X_val=None, residuals_val=None):
        self.model.fit(X, residuals)

    def predict(self, X):
        return self.model.predict(X)


class HuberResidual(ResidualStrategy):
    model_type = "Huber"

    def __init__(self, seed: int = 42, **params: Any):#epsilon=1.35, alpha=0.0001):
        self.model = HuberRegressor(epsilon=params.get("epsilon", 1.35), alpha=params.get("alpha", 0.0001))

    def fit(self, X, residuals, X_val=None, residuals_val=None):
        self.model.fit(X, residuals)

    def predict(self, X):
        return self.model.predict(X)


class PseudoHuberXGBResidualModel(ResidualStrategy):
    """
    Robust non-linear residual model using pseudo-Huber loss.
    Designed to dominate linear Huber when residuals are regime-dependent.
    """

    model_type = "PseudoHuber"

    def __init__(self, seed: int = 42, **params: Any):
        self.model = XGBRegressor(
            random_state=seed,
            objective="reg:pseudohubererror",
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", 3),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=params.get("subsample", 0.9),
            colsample_bytree=params.get("colsample_bytree", 0.9),
            reg_alpha=params.get("reg_alpha", 0.0),
            reg_lambda=params.get("reg_lambda", 1.0),
            tree_method=params.get("tree_method", "hist"),
            n_jobs=params.get("n_jobs", -1),
        )

    def fit(self, X, residuals, X_val=None, residuals_val=None):
        if X_val is not None:
            self.model.fit(
                X, residuals,
                eval_set=[(X_val, residuals_val)],
                verbose=False
            )
        else:
            self.model.fit(X, residuals)

    def predict(self, X):
        return self.model.predict(X)


# =====================================================
# Residual Wrapper
# =====================================================

class ResidualStackedModel(ModelStrategy):
    def __init__(self, base: ModelStrategy, residuals: list[ResidualStrategy]):
        self.base = base
        self.residuals = residuals
        self.model_type = (
            base.model_type + "+" + "+".join(r.model_type for r in residuals)
        )

    def fit(self, X, y, Xva=None, yva=None):
        self.base.fit(X, y, Xva, yva)

        current_pred = self.base.predict(X)
        current_val_pred = (
            self.base.predict(Xva) if Xva is not None else None
        )

        for r in self.residuals:
            residuals = y - current_pred
            residuals_val = (
                yva - current_val_pred if Xva is not None else None
            )

            r.fit(X, residuals, Xva, residuals_val)

            current_pred = current_pred + r.predict(X)
            if Xva is not None:
                current_val_pred = current_val_pred + r.predict(Xva)

    def predict(self, X):
        pred = self.base.predict(X)
        for r in self.residuals:
            pred = pred + r.predict(X)
        return pred


# =====================================================
# Factory (policy injected externally)
# =====================================================
class ModelFactory:
    MAP = ModelStrategy.registry
    NON_MODEL_KEYS = {
        "metric_name",
        "early_stopping_patience",
        "epochs",
        "batch_size",
    }

    @staticmethod
    def create(
        name: str, 
        seed: int, 
        params: Dict[str, Any],
        residual_cfgs: Optional[list] = None
    ) -> ModelStrategy:
        
        if name not in ModelStrategy.registry:
            raise ValueError(f"Unknown model name: {name}")

        base_params = {
            k: v for k, v in params.items()
            if k not in ModelFactory.NON_MODEL_KEYS
        }
        base = ModelStrategy.registry[name](seed, base_params)

        if not residual_cfgs:
            return base

        residuals = []
        for cfg in residual_cfgs:
            cls = ResidualStrategy.registry[cfg["kind"]]
            residuals.append(cls(seed=seed, **cfg.get("params", {})))

        return ResidualStackedModel(base, residuals)
