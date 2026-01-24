from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, ElasticNet, QuantileRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping
from catboost import CatBoostRegressor

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, AdamW, RMSprop, SGD

STACKING_POLICY = {
    "CatBoost": ["ElasticNet", "Quantile"],
    "XGBoost": ["ElasticNet", "Quantile"],
}

class ModelStrategy(ABC):
    model_type: str  # used by SHAP mapping

    @abstractmethod
    def fit(self, Xtr, ytr, Xva=None, yva=None): ...
    @abstractmethod
    def predict(self, Xte) -> np.ndarray: ...

class LinearRegressionStrategy(ModelStrategy):
    model_type = "LinearRegression"
    def __init__(self, seed: int, params: Dict[str, Any]):
        self.model = LinearRegression(**params)

    def fit(self, Xtr, ytr, Xva=None, yva=None):
        self.model.fit(Xtr, ytr)

    def predict(self, Xte):
        return self.model.predict(Xte)

class DecisionTreeStrategy(ModelStrategy):
    model_type = "DecisionTree"
    def __init__(self, seed: int, params: Dict[str, Any]):
        self.model = DecisionTreeRegressor(random_state=seed, **params)

    def fit(self, Xtr, ytr, Xva=None, yva=None):
        self.model.fit(Xtr, ytr)

    def predict(self, Xte):
        return self.model.predict(Xte)

class RandomForestStrategy(ModelStrategy):
    model_type = "RandomForest"
    def __init__(self, seed: int, params: Dict[str, Any]):
        self.model = RandomForestRegressor(random_state=seed, **params)

    def fit(self, Xtr, ytr, Xva=None, yva=None):
        self.model.fit(Xtr, ytr)

    def predict(self, Xte):
        return self.model.predict(Xte)

class SVRStrategy(ModelStrategy):
    model_type = "SVR"
    def __init__(self, seed: int, params: Dict[str, Any]):
        self.model = SVR(**params)

    def fit(self, Xtr, ytr, Xva=None, yva=None):
        self.model.fit(Xtr, ytr)

    def predict(self, Xte):
        return self.model.predict(Xte)

class XGBoostStrategy(ModelStrategy):
    model_type = "XGBoost"
    def __init__(self, seed: int, params: Dict[str, Any]):
        # allow fixed params from json: tree_method, n_jobs, etc.
        self.early_stopping_rounds = params.pop("early_stopping_rounds", None)
        self.model = XGBRegressor(random_state=seed, **params)

    def fit(self, Xtr, ytr, Xva=None, yva=None):
        if Xva is not None and self.early_stopping_rounds:
            try:
                self.model.fit(
                    Xtr, ytr,
                    eval_set=[(Xva, yva)],
                    early_stopping_rounds=self.early_stopping_rounds,
                    verbose=False
                )
                return
            except TypeError:
                try:
                    self.model.fit(
                        Xtr, ytr,
                        eval_set=[(Xva, yva)],
                        callbacks=[EarlyStopping(rounds=self.early_stopping_rounds)],
                        verbose=False
                    )
                    return
                except TypeError:
                    # Fallback for xgboost versions without early-stopping kwargs.
                    self.model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
                    return
        self.model.fit(Xtr, ytr)

    def predict(self, Xte):
        return self.model.predict(Xte)

class CatBoostStrategy(ModelStrategy):
    model_type = "CatBoost"

    def __init__(self, seed: int, params: dict):
        self.cat_features = params.pop("cat_features", None)
        loss_function = params.pop("loss_function", "RMSE")
        self.model = CatBoostRegressor(
            random_seed=seed,
            loss_function=loss_function,
            verbose=False,
            **params,
        )

    def fit(self, Xtr, ytr, Xva=None, yva=None):
        if Xva is not None:
            fit_kwargs = dict(eval_set=(Xva, yva), use_best_model=True)
            if self.cat_features is not None:
                fit_kwargs["cat_features"] = self.cat_features
            self.model.fit(Xtr, ytr, **fit_kwargs)
        else:
            fit_kwargs = {}
            if self.cat_features is not None:
                fit_kwargs["cat_features"] = self.cat_features
            self.model.fit(Xtr, ytr, **fit_kwargs)

    def predict(self, X):
        return self.model.predict(X)

class KerasMLPStrategy(ModelStrategy):
    model_type = "NeuralNetwork"

    def __init__(self, seed: int, params: dict):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        self.params = params
        self.model = None

    def _make_optimizer(self, name, lr, momentum, weight_decay):
        name = name.lower()
        if name == "adam":
            return Adam(learning_rate=lr)
        if name == "adamw":
            return AdamW(learning_rate=lr, weight_decay=weight_decay)
        if name == "rmsprop":
            return RMSprop(learning_rate=lr, momentum=momentum)
        if name == "sgd":
            return SGD(learning_rate=lr, momentum=momentum)
        raise ValueError(f"Unsupported optimizer: {name}")

    def _build(self, input_dim: int):
        p = self.params

        # ---- architecture ----
        n_layers = p.get("n_layers", 2)
        hidden_layer_sizes = tuple(
            p[f"units_layer_{i+1}"] for i in range(n_layers)
        )

        activation = p.get("activation", "relu")
        optimizer_name = p.get("optimizer", "adam")
        learning_rate = p.get("learning_rate", 0.001)
        momentum = p.get("momentum", 0.0)
        weight_decay = p.get("weight_decay", 0.0)
        loss = p.get("loss", "mean_squared_error")

        model = Sequential()
        for i, units in enumerate(hidden_layer_sizes):
            if i == 0:
                model.add(Dense(units, activation=activation, input_dim=input_dim))
            else:
                model.add(Dense(units, activation=activation))
        model.add(Dense(1, activation="linear"))

        opt = self._make_optimizer(
            optimizer_name,
            learning_rate,
            momentum,
            weight_decay
        )

        model.compile(
            optimizer=opt,
            loss=loss,
            metrics=["mean_absolute_error", "mean_squared_error"]
        )
        return model

    def fit(self, Xtr, ytr, Xva=None, yva=None):
        self.model = self._build(Xtr.shape[1])

        monitor_map = {
            "MAE": "val_mean_absolute_error",
            "MSE": "val_mean_squared_error",
            "RMSE": "val_mean_squared_error",  # RMSE not a built-in metric unless you add one
            "R2": "val_loss",                  # unless you implement R2 metric callback
            "MedAE": "val_loss",
        }
        monitor = monitor_map.get(self.params.get("metric_name", "MAE"), "val_loss")

        callbacks = []
        if Xva is not None:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor=monitor,
                    mode="min",
                    patience=self.params.get("early_stopping_patience", 10),
                    restore_best_weights=True
                )
            )

        self.model.fit(
            Xtr,
            ytr,
            validation_data=(Xva, yva) if Xva is not None else None,
            epochs=100,
            batch_size=256,
            callbacks=callbacks,
            verbose=0
        )

    def predict(self, Xte):
        return self.model.predict(Xte, verbose=0).ravel()
    
class ResidualStackedModel:
    """
    y_hat = base(X) + residual(X)
    Assumes both models operate in the SAME target space (log-space here).
    """

    def __init__(self, base_model, residual_model):
        self.base_model = base_model
        self.residual_model = residual_model
        self.model_type = f"{base_model.model_type}+Residual"
        self.use_residual = True

    def _has_non_numeric(self, X) -> bool:
        try:
            if isinstance(X, pd.DataFrame):
                return not all(np.issubdtype(dt, np.number) for dt in X.dtypes)
        except Exception:
            pass
        if isinstance(X, np.ndarray):
            return X.dtype == object
        return True

    def fit(self, X, y, X_val=None, y_val=None):
        # 1) fit base model
        self.base_model.fit(X, y, X_val, y_val)

        # 2) compute residuals on TRAIN ONLY
        base_pred = self.base_model.predict(X)
        residuals = y - base_pred

        if np.std(residuals) < 1e-6:
            self.use_residual = False
            return

        # 3) fit residual model (NO validation needed)
        if self._has_non_numeric(X):
            self.use_residual = False
            return
        try:
            self.residual_model.fit(X, residuals, None, None)
        except Exception:
            self.use_residual = False

    def predict(self, X):
        base_pred = self.base_model.predict(X)
        if not self.use_residual:
            return base_pred
        residual_pred = self.residual_model.predict(X)
        return base_pred + residual_pred
    
class ElasticNetResidualModel:
    def __init__(self, alpha=0.001, l1_ratio=0.5, seed=42):
        self.model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            random_state=seed
        )
        self.model_type = "ElasticNetResidual"

    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class QuantileResidualModel:
    """
    Learns Q_tau(residual | X)
    """

    model_type = "QuantileResidual"

    def __init__(self, quantile: float = 0.5, alpha: float = 0.001):
        self.quantile = quantile
        self.model = QuantileRegressor(
            quantile=quantile,
            alpha=alpha,
            solver="highs"
        )

    def fit(self, X, y, X_val=None, y_val=None):
        # y is residuals
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class ModelFactory:
    MAP = {
        "LinearRegression": LinearRegressionStrategy,
        "DecisionTree": DecisionTreeStrategy,
        "RandomForest": RandomForestStrategy,
        "SVR": SVRStrategy,
        "XGBoost": XGBoostStrategy,
        "NeuralNetwork": KerasMLPStrategy,
        "CatBoost": CatBoostStrategy,
    }

    NON_MODEL_KEYS = {
        "metric_name",
        "early_stopping_patience",
        "epochs",
        "batch_size",
    }

    @staticmethod
    def _build_residual_model(kind: str, seed: int, params: Dict[str, Any] | None = None):
        params = params or {}
        if kind == "ElasticNet":
            return ElasticNetResidualModel(
                alpha=params.get("alpha", 0.001),
                l1_ratio=params.get("l1_ratio", 0.5),
                seed=seed,
            )
        if kind == "Quantile":
            return QuantileResidualModel(
                quantile=params.get("quantile", 0.75),
                alpha=params.get("alpha", 0.001),
            )

    @staticmethod
    def create(name: str, seed: int, params: Dict[str, Any]) -> ModelStrategy:
        if name not in ModelFactory.MAP:
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
        base_model = ModelFactory.MAP[name](seed, model_params)

        # -------------------------------------------------
        # 3) Apply residual stacking if configured
        # -------------------------------------------------
        stack_cfg = STACKING_POLICY.get(name)
        if stack_cfg is None:
            return base_model

        if isinstance(stack_cfg, dict):
            residual_kind = stack_cfg.get("residual")
        elif isinstance(stack_cfg, (list, tuple)):
            residual_kind = stack_cfg[0] if stack_cfg else None
        else:
            residual_kind = None

        if residual_kind is None:
            return base_model
        
        residual_model = ModelFactory._build_residual_model(
            residual_kind,
            seed
        )

        return ResidualStackedModel(
            base_model=base_model,
            residual_model=residual_model
        )
