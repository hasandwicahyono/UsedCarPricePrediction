import numpy as np
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
    ) -> ColumnTransformer:
        
        # numeric pipeline (optionally with ElasticNet FS)
        num_steps = [
            ("yeo", PowerTransformer(method="yeo-johnson")),
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

        
        pre = []
        if cat_encoding == "raw":
            num_pipe = Pipeline([("identity", make_identity_transformer()),])
            cat_pipe = Pipeline([("identity", make_identity_transformer()),])

            pre = ColumnTransformer(transformers=[
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

        pre = ColumnTransformer(
            transformers=[
                ("num", num_pipe, self.schema.num_cols),
                ("cat", cat_pipe, self.schema.cat_cols),
            ],
            remainder="drop"
        )
        return pre
    
