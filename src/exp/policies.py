from __future__ import annotations

DEFAULT_PREPROCESS_POLICY = dict(cat_encoding="onehot", use_feature_selection=False)
DEFAULT_INTERACTION_POLICY = "none"
DEFAULT_EXPLAIN_POLICY = "kernel"

PREPROCESS_POLICIES = {
    "LinearRegression": dict(cat_encoding="target", use_feature_selection=True),
    "RandomForest": dict(cat_encoding="target", use_feature_selection=False),
    "DecisionTree": dict(cat_encoding="target", use_feature_selection=False),
    "SVR": dict(cat_encoding="target", use_feature_selection=True),
    "NeuralNetwork": dict(cat_encoding="target", use_feature_selection=True),
    "XGBoost": dict(cat_encoding="target", use_feature_selection=False),
}

INTERACTION_FEATURE_POLICIES = {
    "LinearRegression": "none",
    "RandomForest": "none",
    "DecisionTree": "none",
    "SVR": "none",
    "NeuralNetwork": "none",
    "XGBoost": "none",
}


EXPLAIN_POLICIES = {
    "LinearRegression": "linear",
    "RandomForest": "tree",
    "DecisionTree": "tree",
    "SVR": "kernel",
    "NeuralNetwork": "gradient",
    "XGBoost": "tree",
}
