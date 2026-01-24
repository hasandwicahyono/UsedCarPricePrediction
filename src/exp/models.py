from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

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
from catboost import CatBoostRegressor

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
    def fit(self, X, y, base_pred): ...

    @abstractmethod
    def predict(self, X): ...


# =====================================================
# Base Models
# =====================================================

class LinearRegressionStrategy(ModelStrategy):
    model_type = "LinearRegression"
    preprocess_policy = dict(cat_encoding="target", use_feature_selection=True)

    def __init__(self, seed, params):
        self.model = LinearRegression(**params)

    def fit(self, Xtr, ytr, Xva=None, yva=None):
        self.model.fit(Xtr, ytr)

    def predict(self, X):
        return self.model.predict(X)


class RandomForestStrategy(ModelStrategy):
    model_type = "RandomForest"
    preprocess_policy = dict(cat_encoding="onehot", use_feature_selection=False)

    def __init__(self, seed, params):
        self.model = RandomForestRegressor(random_state=seed, **params)

    def fit(self, Xtr, ytr, Xva=None, yva=None):
        self.model.fit(Xtr, ytr)

    def predict(self, X):
        return self.model.predict(X)


class DecisionTreeStrategy(ModelStrategy):
    model_type = "DecisionTree"
    preprocess_policy = dict(cat_encoding="onehot", use_feature_selection=False)

    def __init__(self, seed: int, params: Dict[str, Any]):
        self.model = DecisionTreeRegressor(random_state=seed, **params)

    def fit(self, Xtr, ytr, Xva=None, yva=None):
        self.model.fit(Xtr, ytr)

    def predict(self, X):
        return self.model.predict(X)
    


class SVRStrategy(ModelStrategy):
    model_type = "SVR"
    preprocess_policy = dict(cat_encoding="target", use_feature_selection=True)

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
                self.model.fit(
                    Xtr, ytr,
                    eval_set=[(Xva, yva)],
                    callbacks=[EarlyStopping(rounds=self.early_stopping_rounds)],
                    verbose=False,
                )
                return
        self.model.fit(Xtr, ytr)

    def predict(self, X):
        return self.model.predict(X)


# =====================================================
# Residual Models
# =====================================================

class ElasticNetResidual(ResidualStrategy):
    model_type = "ElasticNet"

    def __init__(self, seed=42, alpha=0.001, l1_ratio=0.5):
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=seed)

    def fit(self, X, y, base_pred):
        self.model.fit(X, y - base_pred)

    def predict(self, X):
        return self.model.predict(X)


class QuantileResidual(ResidualStrategy):
    model_type = "Quantile"

    def __init__(self, seed: int = 42, quantile=0.75, alpha=0.001):
        self.model = QuantileRegressor(
            quantile=quantile, alpha=alpha, solver="highs"
        )

    def fit(self, X, y, base_pred):
        self.model.fit(X, y - base_pred)

    def predict(self, X):
        return self.model.predict(X)


class HuberResidual(ResidualStrategy):
    model_type = "Huber"

    def __init__(self, seed: int = 42, epsilon=1.35, alpha=0.0001):
        self.model = HuberRegressor(epsilon=epsilon, alpha=alpha)

    def fit(self, X, y, base_pred):
        self.model.fit(X, y - base_pred)

    def predict(self, X):
        return self.model.predict(X)


# =====================================================
# Residual Wrapper
# =====================================================

class ResidualStackedModel(ModelStrategy):
    def __init__(self, base: ModelStrategy, residual: ResidualStrategy):
        self.base = base
        self.residual = residual
        self.model_type = f"{base.model_type}+{residual.model_type}"

    def fit(self, X, y, Xva=None, yva=None):
        self.base.fit(X, y, Xva, yva)
        self.residual.fit(X, y, self.base.predict(X))

    def predict(self, X):
        return self.base.predict(X) + self.residual.predict(X)


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
        residual_cfg: Optional[dict] = None
    ) -> ModelStrategy:
        if name not in ModelStrategy.registry:
            raise ValueError(f"Unknown model name: {name}")

        # -------------------------------------------------
        # 1) Split params: model vs training/meta
        # -------------------------------------------------
        model_params = {
            k: v for k, v in params.items()
            if k not in ModelFactory.NON_MODEL_KEYS
        }

        # -------------------------------------------------
        # 2) Create base model
        # -------------------------------------------------
        cls = ModelStrategy.registry[name]
        base_model = cls(seed, model_params)

        # -------------------------------------------------
        # 3) Apply residual stacking if configured
        # -------------------------------------------------
        if residual_cfg is None:
            return base_model

        # allow list/tuple config by taking the first residual spec
        if isinstance(residual_cfg, (list, tuple)):
            if not residual_cfg:
                return base_model
            residual_cfg = residual_cfg[0]

        if not isinstance(residual_cfg, dict):
            raise ValueError(
                "residual_cfg must be a dict or list/tuple of dicts "
                f"(got {type(residual_cfg).__name__})"
            )

        residual_kind = residual_cfg["kind"]
        residual_cls = ResidualStrategy.registry[residual_kind]
        residual = residual_cls(seed=seed, **residual_cfg.get("params", {}))

        return ResidualStackedModel(
            base=base_model,
            residual=residual
        )
