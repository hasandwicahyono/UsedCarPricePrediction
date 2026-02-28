import numpy as np
import warnings
from datetime import datetime, timezone
from pathlib import Path
import csv
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV, ElasticNetCV

from .config import FeatureSchema
from .target_encoding import LeakageSafeTargetEncoder


def make_ohe():
    encoder = None
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # sklearn < 1.2 fallback
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return encoder


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([], dtype=object)
        return np.asarray(input_features, dtype=object)

def make_identity_transformer():
    try:
        return FunctionTransformer(validate=False, feature_names_out="one-to-one")
    except TypeError:
        return IdentityTransformer()


class SafePowerTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        method: str = "yeo-johnson",
        standardize: bool = True,
        copy: bool = True,
        run_id: str | None = None,
        outer_fold: int | None = None,
        inner_fold: int | None = None,
        model_name: str | None = None,
        trial_id: int | None = None,
    ):
        self.method = method
        self.standardize = standardize
        self.copy = copy
        self.run_id = run_id
        self.outer_fold = outer_fold
        self.inner_fold = inner_fold
        self.model_name = model_name
        self.trial_id = trial_id

    def _get_feature_names(self, X):
        if hasattr(X, "columns"):
            return list(X.columns)
        n_features = X.shape[1]
        return [f"feature_{i}" for i in range(n_features)]

    def fit(self, X, y=None):
        feature_names = self._get_feature_names(X)
        all_idx = np.arange(len(feature_names), dtype=int)
        X_all = self._select_columns(X, all_idx)

        # Fast path: when all numeric columns are valid, fit once on the full matrix.
        # Fallback to column-wise probing only if this fails.
        try:
            self._pt = PowerTransformer(
                method=self.method,
                standardize=self.standardize,
                copy=self.copy,
            ).fit(X_all, y)
            self._good_idx = all_idx
            self._good_names = list(feature_names)
            return self
        except Exception:
            pass

        good_idx = []
        bad = []
        for i, name in enumerate(feature_names):
            col = X_all[:, i]
            try:
                PowerTransformer(
                    method=self.method,
                    standardize=self.standardize,
                    copy=self.copy,
                ).fit(col.reshape(-1, 1), y)
                good_idx.append(i)
            except Exception as e:
                bad.append((i, name, type(e).__name__))

        if bad:
            bad_names = [name for _, name, _ in bad]
            warnings.warn(
                f"PowerTransformer dropped {len(bad)} numeric columns due to fit errors: {bad_names}",
                RuntimeWarning,
            )
            self._persist_dropped_columns(bad)

        if not good_idx:
            bad_names = [name for _, name, _ in bad]
            raise ValueError(
                "PowerTransformer failed for all numeric columns. "
                f"Dropped columns: {bad_names}"
            )

        self._good_idx = np.array(good_idx, dtype=int)
        self._good_names = [feature_names[i] for i in self._good_idx]
        X_good = X_all[:, self._good_idx]
        self._pt = PowerTransformer(
            method=self.method,
            standardize=self.standardize,
            copy=self.copy,
        ).fit(X_good, y)
        return self

    def _persist_dropped_columns(self, bad):
        root_dir = Path(__file__).resolve().parents[2]
        log_dir = root_dir / "log"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "dropped_numerical_columns.csv"
        write_header = not log_path.exists()
        timestamp = datetime.now(timezone.utc).isoformat()
        with log_path.open("a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    [
                        "run_id",
                        "timestamp_utc",
                        "outer_fold",
                        "inner_fold",
                        "model_name",
                        "trial_id",
                        "column_name",
                        "error_type",
                    ]
                )
            for _, name, err_type in bad:
                writer.writerow(
                    [
                        self.run_id,
                        timestamp,
                        self.outer_fold,
                        self.inner_fold,
                        self.model_name,
                        self.trial_id,
                        name,
                        err_type,
                    ]
                )

    def _select_columns(self, X, idx):
        if hasattr(X, "iloc"):
            return X.iloc[:, idx].to_numpy()
        return np.asarray(X)[:, idx]

    def transform(self, X):
        X_good = self._select_columns(X, self._good_idx)
        return self._pt.transform(X_good)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray(self._good_names, dtype=object)
        return np.asarray(self._good_names, dtype=object)


class PreprocessorBuilder:
    def __init__(self, schema: FeatureSchema):
        self.schema = schema

    def build(
        self,
        *,
        cat_encoding: str = "onehot",      # "onehot" or "target"
        use_feature_selection: bool = False,
        te_smoothing: float = 10.0,
        te_min_samples_leaf: int = 1,
        te_noise_std: float = 0.0,
        seed: int = 42,
        run_id: str | None = None,
        outer_fold: int | None = None,
        inner_fold: int | None = None,
        model_name: str | None = None,
        trial_id: int | None = None,
    ) -> ColumnTransformer:
        
        # numeric pipeline (optionally with ElasticNet FS)
        num_steps = [
            (
                "yeo",
                SafePowerTransformer(
                    method="yeo-johnson",
                    run_id=run_id,
                    outer_fold=outer_fold,
                    inner_fold=inner_fold,
                    model_name=model_name,
                    trial_id=trial_id,
                ),
            ),
            ("scaler", StandardScaler()),
        ]

        if use_feature_selection:
            num_steps.append(
                ("lasso", SelectFromModel(
                    LassoCV(
                        alphas=np.logspace(-4, 1, 50),
                        cv=10,
                        max_iter=20000,
                        n_jobs=-1
                    ),
                    threshold="median"
                ))
            )

        if cat_encoding == "raw":
            num_pipe = Pipeline([("identity", make_identity_transformer()),])
            cat_pipe = Pipeline([("identity", make_identity_transformer()),])

            return ColumnTransformer(transformers=[
                                         ("num", num_pipe, self.schema.num_cols),
                                         ("cat", cat_pipe, self.schema.cat_cols),],
                                     remainder="drop",
                                     verbose_feature_names_out=False)
    
        num_pipe = Pipeline(num_steps)
        # categorical pipeline
        if cat_encoding == "onehot":
            cat_pipe = make_ohe()
        elif cat_encoding == "target":
            cat_pipe = LeakageSafeTargetEncoder(
                cols=self.schema.cat_cols,
                smoothing=te_smoothing,
                min_samples_leaf=te_min_samples_leaf,
                noise_std=te_noise_std,
                random_state=seed
            )
        else:
            raise ValueError(f"Unknown cat_encoding: {cat_encoding}")

        return ColumnTransformer(
            transformers=[
                ("num", num_pipe, self.schema.num_cols),
                ("cat", cat_pipe, self.schema.cat_cols),
            ],
            remainder="drop"
        )
    
