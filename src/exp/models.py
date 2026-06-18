from __future__ import annotations

# NOTE: Model/Residual strategy classes below are registered and instantiated via registries/factories.
from abc import ABC
from typing import Dict, Any, Optional
import os
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
    Ridge,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from scipy.optimize import differential_evolution

from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping

from .registry import MODEL_REGISTRY, RESIDUAL_REGISTRY
from .policies import (
    DEFAULT_PREPROCESS_POLICY,
    DEFAULT_INTERACTION_POLICY,
    DEFAULT_EXPLAIN_POLICY,
    PREPROCESS_POLICIES,
    INTERACTION_FEATURE_POLICIES,
    EXPLAIN_POLICIES,
)

# =====================================================
# Base Strategy Registries
# =====================================================

class ModelStrategy(ABC):
    registry = MODEL_REGISTRY
    model_type: str
    preprocess_policy: dict = DEFAULT_PREPROCESS_POLICY
    interaction_policy: str = DEFAULT_INTERACTION_POLICY
    explain_policy: str = DEFAULT_EXPLAIN_POLICY

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "model_type"):
            ModelStrategy.registry.register(cls.model_type, cls)
            # set defaults only if child didn't override
            if not hasattr(cls, "preprocess_policy") or cls.preprocess_policy == ModelStrategy.preprocess_policy:
                cls.preprocess_policy = PREPROCESS_POLICIES.get(cls.model_type, DEFAULT_PREPROCESS_POLICY)
            if not hasattr(cls, "interaction_policy") or cls.interaction_policy == ModelStrategy.interaction_policy:
                cls.interaction_policy = INTERACTION_FEATURE_POLICIES.get(cls.model_type, DEFAULT_INTERACTION_POLICY)
            if not hasattr(cls, "explain_policy") or cls.explain_policy == ModelStrategy.explain_policy:
                cls.explain_policy = EXPLAIN_POLICIES.get(cls.model_type, DEFAULT_EXPLAIN_POLICY)


    def fit(self, Xtr, ytr, Xva=None, yva=None):
        # Optimization for sklearnex: ensure C-contiguous arrays to avoid internal copies
        X_c = np.ascontiguousarray(Xtr)
        y_c = np.ascontiguousarray(ytr)
        self.model.fit(X_c, y_c)

    def predict(self, X):
        X_c = np.ascontiguousarray(X)
        return self.model.predict(X_c)
    

class ResidualStrategy(ABC):
    registry = RESIDUAL_REGISTRY
    model_type: str

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "model_type"):
            ResidualStrategy.registry.register(cls.model_type, cls)

    def fit(self, X, residuals, X_val=None, residuals_val=None):
        # Optimization for sklearnex: ensure C-contiguous arrays
        X_c = np.ascontiguousarray(X)
        res_c = np.ascontiguousarray(residuals)
        self.model.fit(X_c, res_c)

    def predict(self, X):
        X_c = np.ascontiguousarray(X)
        return self.model.predict(X_c)


# =====================================================
# Base Models
# =====================================================

class LinearRegressionStrategy(ModelStrategy):
    model_type = "LinearRegression"

    def __init__(self, seed, params):
        self.model = LinearRegression(**params)


class RandomForestStrategy(ModelStrategy):
    model_type = "RandomForest"

    def __init__(self, seed, params):
        self.model = RandomForestRegressor(random_state=seed, **params)


class DecisionTreeStrategy(ModelStrategy):
    model_type = "DecisionTree"

    def __init__(self, seed: int, params: Dict[str, Any]):
        self.model = DecisionTreeRegressor(random_state=seed, **params)


class SVRStrategy(ModelStrategy):
    model_type = "SVR"

    def __init__(self, seed: int, params: Dict[str, Any]):
        # Optimization: Increase cache_size to 1000MB (1GB) to speed up kernel computations.
        # Standard Scikit-Learn default is 200MB, which is often too low for modern RAM.
        params.setdefault("cache_size", 1000)
        self.model = SVR(**params)

class BaggingSVRStrategy(ModelStrategy):
    model_type = "BaggingSVR"

    def __init__(self, seed: int, params: Dict[str, Any]):
        params = params.copy()
        # Extract SVR-specific hyperparams
        svr_params = {
            "C": params.pop("svr_C", 1.0),
            "gamma": params.pop("svr_gamma", "scale"),
            "epsilon": params.pop("svr_epsilon", 0.1),
            "tol": params.pop("svr_tol", 1e-4),
            "kernel": params.pop("svr_kernel", "rbf"),
            "cache_size": params.pop("svr_cache_size", 1000),
        }
        base_svr = SVR(**svr_params)
        
        # BaggingRegressor handles the parallelization via n_jobs
        self.model = BaggingRegressor(
            estimator=base_svr,
            random_state=seed,
            **params
        )

class BaggingXGBStrategy(ModelStrategy):
    model_type = "BaggingXGB"

    def __init__(self, seed: int, params: Dict[str, Any]):
        params = params.copy()
        self.params = params.copy()
        # Extract XGBoost-specific hyperparams
        xgb_params = {
            "max_depth": params.pop("xgb_max_depth", 3),
            "learning_rate": params.pop("xgb_learning_rate", 0.1),
            "n_estimators": params.pop("xgb_n_estimators", 100),
            "tree_method": params.pop("xgb_tree_method", "hist"),
            "random_state": seed,
            "n_jobs": 1, # BaggingRegressor handles parallelization
        }
        
        try:
            import torch
            if torch.cuda.is_available():
                xgb_params.setdefault("device", "cuda")
        except ImportError:
            pass

        base_xgb = XGBRegressor(**xgb_params)
        
        self.model = BaggingRegressor(
            estimator=base_xgb,
            random_state=seed,
            **params
        )

    def fit(self, Xtr, ytr, Xva=None, yva=None):
        sample_weight = None
        if self.params.get("balanced_weighting"):
            from sklearn.utils.class_weight import compute_sample_weight
            try:
                sample_weight = compute_sample_weight('balanced', ytr)
            except Exception:
                sample_weight = None
        self.model.fit(Xtr, ytr, sample_weight=sample_weight)

# =====================================================
# Neural Network
# =====================================================

class TorchMLPStrategy(ModelStrategy):
    model_type = "NeuralNetwork"

    def __init__(self, seed: int, params: Dict[str, Any]):
        import torch
        torch.manual_seed(seed)
        self.seed = seed
        self.params = params.copy()
        
        device_name = self.params.pop("device", "cpu")
        if device_name == "cpu":
            if torch.cuda.is_available():
                device_name = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device_name = "mps"
        self.device = device_name
        self.model = None

    def fit(self, Xtr, ytr, Xva=None, yva=None):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        Xtr_t = torch.tensor(np.asarray(Xtr), dtype=torch.float32)
        ytr_t = torch.tensor(np.asarray(ytr), dtype=torch.float32).view(-1, 1)

        train_loader = DataLoader(
            TensorDataset(Xtr_t, ytr_t),
            batch_size=self.params.get("batch_size", 256),
            shuffle=True
        )
       # Architecture setup
        layers = []
        curr_dim = Xtr_t.shape[1]
        p = self.params
        n_layers = p.get("n_layers", 2)
        activation_name = p.get("activation", "relu").lower()
        act_map = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid()
        }
        activation = act_map.get(activation_name, nn.ReLU())

        for i in range(n_layers):
            units = p.get(f"units_layer_{i+1}", 64)
            layers.append(nn.Linear(curr_dim, units))
            layers.append(activation)
            if p.get("use_batchnorm", True):
                layers.append(nn.BatchNorm1d(units))
            layers.append(nn.Dropout(p.get("dropout", 0.0)))
            curr_dim = units
        layers.append(nn.Linear(curr_dim, 1))
        
        self.model = nn.Sequential(*layers).to(self.device)

        # Optimizer setup
        opt_name = p.get("optimizer", "adam").lower()
        lr = p.get("learning_rate", 0.001)
        wd = p.get("weight_decay", 0.0)
        mom = p.get("momentum", 0.0)

        if opt_name == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "adamw":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "rmsprop":
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=wd, momentum=mom)
        elif opt_name == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=wd, momentum=mom)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        criterion = nn.MSELoss()
        epochs = p.get("epochs", 100)

        self.model.train()
        for _ in range(epochs):
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

    def predict(self, X):
        import torch
        if self.model is None:
            return np.zeros(len(X))
        self.model.eval()
        X_tensor = torch.tensor(np.asarray(X), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy().flatten()
        return preds


class TabNetStrategy(ModelStrategy):
    model_type = "TabNet"

    def __init__(self, seed: int, params: Dict[str, Any]):
        import torch
        from pytorch_tabnet.tab_model import TabNetRegressor
        
        params = params.copy()
        self.fit_params = {
            "max_epochs": params.pop("max_epochs", 100),
            "patience": params.pop("patience", 20),
            "batch_size": params.pop("batch_size", 256),
            "virtual_batch_size": params.pop("virtual_batch_size", 128),
            "num_workers": params.pop("num_workers", 0), 
            "drop_last": params.pop("drop_last", False),
            "pin_memory": True if torch.backends.mps.is_available() else False,
        }
        
        learning_rate = params.pop("learning_rate", 0.01)
        
        # Default to CPU for stability due to MPS backend assertion failures in Transformers
        device_name = params.pop("device", "auto")
        if device_name == "auto":
            if torch.cuda.is_available():
                device_name = "cuda"
            else:
                device_name = "cpu"
        # Fix for pytorch-tabnet compatibility with modern PyTorch versions.
        # ReduceLROnPlateau requires a metric in .step(), but pytorch-tabnet
        # only detects it as "metric-related" if 'is_better' exists on the class.
        # In modern PyTorch, this attribute only exists on the instance.
        if not hasattr(torch.optim.lr_scheduler.ReduceLROnPlateau, "is_better"):
            torch.optim.lr_scheduler.ReduceLROnPlateau.is_better = True

        self.model = TabNetRegressor(
            seed=seed,
            verbose=params.pop("verbose", 0),
            optimizer_fn=torch.optim.AdamW,
            optimizer_params={"lr": learning_rate, "weight_decay": 1e-5},
            scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
            scheduler_params={
                "mode": "min", "patience": 5, "factor": 0.5
            },
            device_name=device_name,
            **params,
        )

    def _as_matrix(self, X):
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        elif hasattr(X, "toarray"):
            X = X.toarray()
        return np.asarray(X, dtype=np.float32)

    def fit(self, Xtr, ytr, Xva=None, yva=None):
        import torch
        Xtr = self._as_matrix(Xtr)
        ytr = np.asarray(ytr, dtype=np.float32).reshape(-1, 1)
        fit_kwargs = dict(self.fit_params)
        if Xva is not None and yva is not None:
            fit_kwargs["eval_set"] = [
                (self._as_matrix(Xva), np.asarray(yva, dtype=np.float32).reshape(-1, 1))
            ]
        self.model.fit(Xtr, ytr, **fit_kwargs)

    def predict(self, X):
        return self.model.predict(self._as_matrix(X)).reshape(-1)

class FTTransformerStrategy(ModelStrategy):
    """
    FT-Transformer (Feature Tokenizer + Transformer)
    Designed for tabular data with heterogeneous features.
    """
    model_type = "FTTransformer"

    def __init__(self, seed: int, params: Dict[str, Any]):
        self.seed = seed
        self.params = params.copy()
        self.model = None
        self.trainer = None

    def _prepare_data(self, X):
        n_features = X.shape[1]
        cols = [f"c{i}" for i in range(n_features)]
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=cols)
        else:
            X = X.copy()
            X.columns = cols
        return X, cols

    def fit(self, Xtr, ytr, Xva=None, yva=None):
        import torch
        import numpy as np
        from pytorch_widedeep.models import FTTransformer, WideDeep
        from pytorch_widedeep import Trainer
        from pytorch_widedeep.callbacks import EarlyStopping

        Xtr_df, cols = self._prepare_data(Xtr)
        ytr = np.asarray(ytr, dtype=np.float32).reshape(-1, 1)
        
        Xva_df = None
        if Xva is not None:
            Xva_df, _ = self._prepare_data(Xva)
            yva = np.asarray(yva, dtype=np.float32).reshape(-1, 1)

        m_keys = {'n_blocks', 'n_heads', 'input_dim', 'ff_hidden_multiplier', 'attn_dropout', 'ff_dropout'}
        m_params = {k: self.params.get(k) for k in m_keys if k in self.params}
        
        # Ensure input_dim is divisible by n_heads to prevent WideDeep assertion failures
        if "input_dim" in m_params and "n_heads" in m_params:
            in_dim, heads = m_params["input_dim"], m_params["n_heads"]
            if in_dim % heads != 0:
                m_params["input_dim"] = max(heads, int(round(in_dim / heads)) * heads)

        ft_backbone = FTTransformer(
            column_idx={col: i for i, col in enumerate(cols)},
            cat_embed_input=[], 
            continuous_cols=cols,
            **m_params
        )
        self.model = WideDeep(deeptabular=ft_backbone)
        
        # Automatically detect acceleration device (CUDA or MPS) if not explicitly set to 'cpu'.
        device = self.params.get("device")
        if not device or device == "cpu":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # MPS is significantly faster on Mac but requires modern PyTorch/WideDeep versions.
                device = "mps"
            else:
                device = "cpu"

        # Enable PyTorch 2.0+ compilation for a potential 20-30% speed boost.
        # Triton (the default backend for torch.compile) is not supported on Windows.
        # We guard the compilation to prevent TritonMissing errors on Windows systems.
        if hasattr(torch, "compile") and device == "cuda" and os.name != "nt":
            try:
                self.model = torch.compile(self.model)
            except Exception:
                pass

        callbacks = []
        if Xva is not None and "patience" in self.params:
            callbacks.append(EarlyStopping(patience=self.params["patience"]))

        self.trainer = Trainer(
            self.model, 
            objective="mse", 
            seed=self.seed,
            optimizers=torch.optim.Adam(self.model.parameters(), lr=self.params.get("learning_rate", 0.001)),
            callbacks=callbacks,
            device=device,
            verbose=0,
            fp16=True if device == "cuda" else False, # Mixed precision speedup
            num_workers=self.params.get("num_workers", 0), # Parallel data loading
        )
        self.trainer.fit(
            X_train={"X_tab": Xtr_df.values, "target": ytr}, 
            target=None,
            X_val={"X_tab": Xva_df.values, "target": yva} if Xva_df is not None else None,
            val_target=None,
            n_epochs=self.params.get("epochs", 50),
            batch_size=self.params.get("batch_size", 128),
        )

    def predict(self, X):
        X_df, _ = self._prepare_data(X)
        return self.trainer.predict(X_test={"X_tab": X_df.values}).reshape(-1)


class XGBoostStrategy(ModelStrategy):
    model_type = "XGBoost"

    def __init__(self, seed, params):
        params = params.copy()
        self.params = params.copy()
        self.early_stopping_rounds = params.pop("early_stopping_rounds", None)

        try:
            import torch
            if torch.cuda.is_available():
                params.setdefault("device", "cuda")
                params.setdefault("tree_method", "hist")
        except ImportError:
            pass

        # Standard CPU settings: using constructor-based early stopping for XGBoost 2.x
        self.model = XGBRegressor(
            random_state=seed, 
            early_stopping_rounds=self.early_stopping_rounds,
            **params
        )

    def fit(self, Xtr, ytr, Xva=None, yva=None):
        # Handle class imbalance for ECG signals by calculating balanced 
        # sample weights if enabled in the hyperparameter search space.
        sample_weight = None
        if self.params.get("balanced_weighting"):
            from sklearn.utils.class_weight import compute_sample_weight
            try:
                sample_weight = compute_sample_weight('balanced', ytr)
            except Exception:
                sample_weight = None

        if Xva is not None:
            self.model.fit(
                Xtr, ytr,
                eval_set=[(Xva, yva)],
                sample_weight=sample_weight,
                verbose=False,
            )
        else:
            # If no validation set, we MUST disable early stopping
            # because XGBoost requires an eval_set if this param is set.
            self.model.set_params(early_stopping_rounds=None)
            self.model.fit(Xtr, ytr, sample_weight=sample_weight)


# =====================================================
# Residual Models
# =====================================================

class ElasticNetResidual(ResidualStrategy):
    model_type = "ElasticNet"

    def __init__(self, seed=42, **params: Any):
        self.model = ElasticNet(
            alpha=params.get("alpha", 0.001), 
            l1_ratio=params.get("l1_ratio", 0.5), 
            random_state=seed
        )


class QuantileResidual(ResidualStrategy):
    model_type = "Quantile"

    def __init__(self, seed: int = 42, **params: Any):
        self.model = QuantileRegressor(
            quantile=params.get("quantile", 0.75), alpha=params.get("alpha", 0.001), solver="highs"
        )


class HuberResidual(ResidualStrategy):
    model_type = "Huber"

    def __init__(self, seed: int = 42, **params: Any):
        self.model = HuberRegressor(
            epsilon=params.get("epsilon", 1.35), 
            alpha=params.get("alpha", 0.0001),
            max_iter=params.get("max_iter", 500)
        )


class FuzzyLogicResidual(ResidualStrategy):
    """
    Takagi-Sugeno-style fuzzy residual model.
    KMeans centers define fuzzy rules, inverse-distance memberships become rule
    activations, and a ridge consequent maps activations to residual corrections.
    """

    model_type = "FuzzyLogic"

    def __init__(self, seed: int = 42, **params: Any):
        self.seed = seed
        self.n_rules = int(params.get("n_rules", 5))
        self.fuzziness = float(params.get("fuzziness", 2.0))
        self.alpha = float(params.get("alpha", 0.001))
        self.max_iter = int(params.get("max_iter", 100))
        self.kmeans = None
        self.model = Ridge(alpha=self.alpha)

    def _as_matrix(self, X):
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        elif hasattr(X, "toarray"):
            X = X.toarray()
        return np.asarray(X, dtype=np.float64)

    def _memberships(self, X):
        distances = np.linalg.norm(X[:, None, :] - self.centers_[None, :, :], axis=2)
        distances = np.maximum(distances, 1e-12)
        exponent = 2.0 / max(self.fuzziness - 1.0, 1e-6)
        inv = distances ** (-exponent)
        return inv / inv.sum(axis=1, keepdims=True)

    def fit(self, X, residuals, X_val=None, residuals_val=None):
        X = self._as_matrix(X)
        residuals = np.asarray(residuals, dtype=np.float64).reshape(-1)
        n_rules = max(1, min(self.n_rules, len(X)))
        self.kmeans = KMeans(
            n_clusters=n_rules,
            random_state=self.seed,
            n_init=10,
            max_iter=self.max_iter,
        )
        self.kmeans.fit(X)
        self.centers_ = self.kmeans.cluster_centers_
        self.model.fit(self._memberships(X), residuals)

    def predict(self, X):
        return self.model.predict(self._memberships(self._as_matrix(X)))


class EvolutionaryStackingResidual(ResidualStrategy):
    """
    Residual stacker with weights optimized by differential evolution.
    Candidate residual learners are fitted first; a small evolutionary search then
    finds non-negative blending weights that minimize validation residual error.
    """

    model_type = "EvolutionaryStacking"

    def __init__(self, seed: int = 42, **params: Any):
        self.seed = seed
        self.max_iter = int(params.get("max_iter", 20))
        self.population_size = int(params.get("population_size", 8))
        self.validation_fraction = float(params.get("validation_fraction", 0.25))
        self.l2_penalty = float(params.get("l2_penalty", 0.0))
        self.tree_max_depth = int(params.get("tree_max_depth", 3))
        self.rf_n_estimators = int(params.get("rf_n_estimators", 50))
        self.models = []
        self.weights_ = None

    def _as_matrix(self, X):
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        elif hasattr(X, "toarray"):
            X = X.toarray()
        return np.asarray(X, dtype=np.float64)

    def _make_candidates(self):
        return [
            Ridge(alpha=0.001),
            HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=500),
            DecisionTreeRegressor(max_depth=self.tree_max_depth, random_state=self.seed),
            RandomForestRegressor(
                n_estimators=self.rf_n_estimators,
                max_depth=self.tree_max_depth,
                random_state=self.seed,
                n_jobs=-1,
            ),
        ]

    def _normalize_weights(self, weights):
        weights = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
        total = weights.sum()
        if total <= 0:
            return np.full_like(weights, 1.0 / len(weights))
        return weights / total

    def _prediction_matrix(self, X):
        return np.column_stack([m.predict(X) for m in self.models])

    def _optimize_weights(self, pred_matrix, target):
        target = np.asarray(target, dtype=np.float64).reshape(-1)

        def objective(raw_weights):
            weights = self._normalize_weights(raw_weights)
            pred = pred_matrix @ weights
            return float(np.mean((target - pred) ** 2) + self.l2_penalty * np.sum(weights ** 2))

        result = differential_evolution(
            objective,
            bounds=[(0.0, 1.0)] * pred_matrix.shape[1],
            maxiter=self.max_iter,
            popsize=self.population_size,
            seed=self.seed,
            polish=False,
            updating="immediate",
            workers=1,
        )
        return self._normalize_weights(result.x)

    def fit(self, X, residuals, X_val=None, residuals_val=None):
        X = self._as_matrix(X)
        residuals = np.asarray(residuals, dtype=np.float64).reshape(-1)

        if X_val is not None and residuals_val is not None:
            X_fit, y_fit = X, residuals
            X_opt = self._as_matrix(X_val)
            y_opt = np.asarray(residuals_val, dtype=np.float64).reshape(-1)
        elif len(X) >= 8:
            X_fit, X_opt, y_fit, y_opt = train_test_split(
                X,
                residuals,
                test_size=self.validation_fraction,
                random_state=self.seed,
            )
        else:
            X_fit, y_fit = X, residuals
            X_opt, y_opt = X, residuals

        self.models = self._make_candidates()
        for model in self.models:
            model.fit(X_fit, y_fit)

        self.weights_ = self._optimize_weights(self._prediction_matrix(X_opt), y_opt)

        self.models = self._make_candidates()
        for model in self.models:
            model.fit(X, residuals)

    def predict(self, X):
        return self._prediction_matrix(self._as_matrix(X)) @ self.weights_


class PseudoHuberXGBResidualModel(ResidualStrategy):
    """
    Robust non-linear residual model using pseudo-Huber loss.
    Designed to dominate linear Huber when residuals are regime-dependent.
    """

    model_type = "PseudoHuber"

    def __init__(self, seed: int = 42, **params: Any):
        self.early_stopping_rounds = params.pop("early_stopping_rounds", None)
        xgb_kwargs = dict(
            random_state=seed,
            objective="reg:pseudohubererror",
            early_stopping_rounds=self.early_stopping_rounds,
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

        try:
            import torch
            if torch.cuda.is_available():
                xgb_kwargs.setdefault("device", "cuda")
        except ImportError:
            pass

        self.model = XGBRegressor(**xgb_kwargs)

    def fit(self, X, residuals, X_val=None, residuals_val=None):
        if X_val is not None:
            self.model.fit(
                X, residuals,
                eval_set=[(X_val, residuals_val)],
                verbose=False
            )
        else:
            self.model.set_params(early_stopping_rounds=None)
            self.model.fit(X, residuals)


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

            current_pred += r.predict(X)
            if Xva is not None:
                current_val_pred += r.predict(Xva)

    def predict(self, X):
        pred = self.base.predict(X)
        for r in self.residuals:
            pred += r.predict(X)
        return pred


# =====================================================
# Factory (policy injected externally)
# =====================================================
class ModelFactory:
    MAP = ModelStrategy.registry
    NON_MODEL_KEYS = {
        "metric_name",
        "early_stopping_patience",
        "epochs"
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
            if k not in ModelFactory.NON_MODEL_KEYS and not k.startswith("residual__")
        }
        base = ModelStrategy.registry[name](seed, base_params)

        if not residual_cfgs:
            return base

        residuals = []
        for cfg in residual_cfgs:
            cls = ResidualStrategy.registry[cfg["kind"]]
            residuals.append(cls(seed=seed, **cfg.get("params", {})))

        return ResidualStackedModel(base, residuals)
