from __future__ import annotations

DEFAULT_PREPROCESS_POLICY = dict(cat_encoding="onehot", use_feature_selection=False)

PREPROCESS_POLICIES = {
    "LinearRegression": dict(cat_encoding="target", use_feature_selection=True),
    "RandomForest": dict(cat_encoding="onehot", use_feature_selection=False),
    "DecisionTree": dict(cat_encoding="onehot", use_feature_selection=False),
    "SVR": dict(cat_encoding="target", use_feature_selection=True),
    "NeuralNetwork": dict(cat_encoding="target", use_feature_selection=True),
    "XGBoost": dict(cat_encoding="onehot", use_feature_selection=False),
}

EXPLAIN_POLICIES = {
    "LinearRegression": "linear",
    "RandomForest": "tree",
    "DecisionTree": "tree",
    "SVR": "kernel",
    "NeuralNetwork": "kernel",
    "XGBoost": "tree",
}

def get_preprocess_policy(name: str) -> dict:
    return PREPROCESS_POLICIES.get(name, DEFAULT_PREPROCESS_POLICY)


class PolicyProvider:
    def get_preprocess_policy(self, name: str) -> dict:
        return get_preprocess_policy(name)

    def get_explain_policy(self, name: str) -> str:
        return EXPLAIN_POLICIES.get(name, "tree")
