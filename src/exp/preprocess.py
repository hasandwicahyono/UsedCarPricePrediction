import numpy as np
import warnings
from datetime import datetime, timezone
from pathlib import Path
import csv
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler, FunctionTransformer, QuantileTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from joblib import Parallel, delayed

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


class GeneticFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Lightweight genetic algorithm for numeric feature subset selection.
    Fitness combines validation MSE from a ridge model with a subset-size penalty.
    """

    def __init__(
        self,
        population_size: int = 24,
        generations: int = 20,
        mutation_rate: float = 0.05,
        crossover_rate: float = 0.8,
        subset_penalty: float = 0.01,
        validation_fraction: float = 0.25,
        random_state: int = 42,
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.subset_penalty = subset_penalty
        self.validation_fraction = validation_fraction
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n_features = X.shape[1]
        self.n_features_in_ = n_features
        if n_features <= 1 or X.shape[0] < 8:
            self.support_ = np.ones(n_features, dtype=bool)
            return self

        rng = np.random.default_rng(self.random_state)
        Xtr, Xva, ytr, yva = train_test_split(
            X,
            y,
            test_size=self.validation_fraction,
            random_state=self.random_state,
        )

        pop_size = max(4, int(self.population_size))
        population = rng.random((pop_size, n_features)) < 0.5
        population[0, :] = True
        for row in population:
            if not row.any():
                row[rng.integers(0, n_features)] = True

        def fitness(mask):
            if not mask.any():
                return np.inf
            model = Ridge(alpha=1.0)
            model.fit(Xtr[:, mask], ytr)
            pred = model.predict(Xva[:, mask])
            mse = mean_squared_error(yva, pred)
            return float(mse + self.subset_penalty * (mask.mean()))

        # Parallelize fitness evaluation across CPU cores
        scores = np.array(Parallel(n_jobs=-1, require="sharedmem")(delayed(fitness)(mask) for mask in population))
        for _ in range(max(1, int(self.generations))):
            elite = population[np.argmin(scores)].copy()
            new_pop = [elite]
            while len(new_pop) < pop_size:
                p1 = self._tournament(population, scores, rng)
                p2 = self._tournament(population, scores, rng)
                c1, c2 = p1.copy(), p2.copy()
                if rng.random() < self.crossover_rate and n_features > 1:
                    point = rng.integers(1, n_features)
                    c1 = np.r_[p1[:point], p2[point:]]
                    c2 = np.r_[p2[:point], p1[point:]]
                for child in (c1, c2):
                    flips = rng.random(n_features) < self.mutation_rate
                    child[flips] = ~child[flips]
                    if not child.any():
                        child[rng.integers(0, n_features)] = True
                    new_pop.append(child)
                    if len(new_pop) >= pop_size:
                        break
            population = np.asarray(new_pop, dtype=bool)
            scores = np.array(Parallel(n_jobs=-1, require="sharedmem")(delayed(fitness)(mask) for mask in population))

        self.support_ = population[np.argmin(scores)].astype(bool)
        return self

    @staticmethod
    def _tournament(population, scores, rng, k: int = 3):
        k = min(k, len(population))
        idx = rng.choice(len(population), size=k, replace=False)
        return population[idx[np.argmin(scores[idx])]]

    def transform(self, X):
        return np.asarray(X)[:, self.support_]

    def get_support(self):
        return self.support_

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"feature_{i}" for i in range(self.n_features_in_)]
        return np.asarray(input_features, dtype=object)[self.support_]


class PreprocessorBuilder:
    def __init__(self, schema: FeatureSchema):
        self.schema = schema

    def build(
        self,
        *,
        cat_encoding: str = "onehot",      # "onehot" or "target"
        use_feature_selection: bool = False,
        feature_selection_method: str = "lasso",
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
        
        feature_selection_method = str(feature_selection_method or "lasso").lower()

        # Strategy: Tabular DL models benefit significantly from RankGauss (QuantileTransformer)
        # which maps numeric data to a normal distribution, reducing the impact of outliers.
        numeric_transformer = SafePowerTransformer(method="yeo-johnson",
                                                   run_id=run_id,
                                                   outer_fold=outer_fold,
                                                   inner_fold=inner_fold,
                                                   model_name=model_name,
                                                   trial_id=trial_id)
        if model_name in {"TabNet", "FTTransformer", "NeuralNetwork"}:
             numeric_transformer = QuantileTransformer(output_distribution='normal', random_state=seed)

        # numeric pipeline (optionally with feature selection)
        num_steps = [
            ("numeric_transform", numeric_transformer),
            ("scaler", StandardScaler()),
        ]

        if use_feature_selection:
            if feature_selection_method == "lasso":
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
            elif feature_selection_method in {"genetic", "ga"}:
                num_steps.append(
                    ("genetic", GeneticFeatureSelector(random_state=seed))
                )
            else:
                raise ValueError(f"Unknown feature_selection_method: {feature_selection_method}")

        if cat_encoding == "raw":
            num_pipe = Pipeline([("identity", make_identity_transformer()),])
            cat_pipe = Pipeline([("identity", make_identity_transformer()),])

            return ColumnTransformer(transformers=[
                                         ("num", num_pipe, self.schema.num_cols),
                                         ("cat", cat_pipe, self.schema.cat_cols),],
                                     remainder="drop",
                                     verbose_feature_names_out=False)
    
        num_pipe = Pipeline(num_steps, memory="cache_directory")
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
    
