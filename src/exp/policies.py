from __future__ import annotations

DEFAULT_PREPROCESS_POLICY = dict(cat_encoding="onehot", use_feature_selection=False)
DEFAULT_INTERACTION_POLICY = "none"
DEFAULT_EXPLAIN_POLICY = "kernel"

PREPROCESS_POLICIES = {
    "LinearRegression": dict(cat_encoding="onehot", use_feature_selection=True, feature_selection_method="genetic"),
    "RandomForest": dict(cat_encoding="target", use_feature_selection=False),
    "DecisionTree": dict(cat_encoding="target", use_feature_selection=False),
    "SVR": dict(cat_encoding="onehot", use_feature_selection=True, feature_selection_method="genetic"),
    "NeuralNetwork": dict(cat_encoding="onehot", use_feature_selection=True, feature_selection_method="genetic"),
    "TabNet": dict(cat_encoding="target", use_feature_selection=False),
    "XGBoost": dict(cat_encoding="target", use_feature_selection=False),
    "FTTransformer": dict(cat_encoding="target", use_feature_selection=False),
    "BaggingSVR": dict(cat_encoding="onehot", use_feature_selection=True, feature_selection_method="genetic"),
}

INTERACTION_FEATURE_POLICIES = {
    "LinearRegression": "interaction",
    "RandomForest": "none",
    "DecisionTree": "none",
    "SVR": "interaction",
    "NeuralNetwork": "interaction",
    "TabNet": "none",
    "XGBoost": "none",
    "BaggingSVR": "interaction",
}


EXPLAIN_POLICIES = {
    "LinearRegression": "linear",
    "RandomForest": "tree",
    "DecisionTree": "tree",
    "SVR": "kernel",
    "NeuralNetwork": "gradient",
    "TabNet": "kernel",
    "XGBoost": "tree",
    "FTTransformer": "kernel",
    "BaggingSVR": "kernel",
}
