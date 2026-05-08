import unittest
import numpy as np

from src.exp.factories import build_metric
from src.exp.facade import aggregate_hyperparams, _normalize_model_params
from src.exp.metrics import MAEMetric, make_metric, METRIC_REGISTRY
from src.exp.specs import MetricSpec, ModelSpec, PreprocessSpec


class TestFactoriesAndMetrics(unittest.TestCase):
    def test_metric_registry_lookup(self):
        cls = METRIC_REGISTRY.get("mae")
        self.assertIs(cls, MAEMetric)

    def test_make_metric_cached(self):
        m1 = make_metric("MAE")
        m2 = make_metric("mae")
        self.assertIs(m1, m2)
        self.assertEqual(m1.name, "MAE")
        self.assertEqual(m1.direction, "minimize")

    def test_build_metric_from_spec_and_name(self):
        m1 = build_metric("rmse")
        m2 = build_metric(MetricSpec(name="rmse"))
        self.assertEqual(m1.name, "RMSE")
        self.assertEqual(m2.name, "RMSE")

    def test_spec_validation(self):
        with self.assertRaises(ValueError):
            ModelSpec(name="").validate()
        with self.assertRaises(ValueError):
            PreprocessSpec(cat_encoding="weird").validate()

    def test_aggregate_hyperparams_handles_numpy_scalars(self):
        aggregated = aggregate_hyperparams(
            [
                {"max_depth": np.int64(3), "learning_rate": np.float64(0.1), "features": ["a", "b"]},
                {"max_depth": np.int64(5), "learning_rate": np.float64(0.2), "features": ["a", "b"]},
                {"max_depth": np.int64(7), "learning_rate": np.float64(0.3), "features": ["c"]},
            ]
        )

        self.assertEqual(aggregated["max_depth"], 5)
        self.assertAlmostEqual(aggregated["learning_rate"], 0.2)
        self.assertEqual(aggregated["features"], ["a", "b"])

    def test_aggregate_hyperparams_preserves_boolean_values(self):
        aggregated = aggregate_hyperparams(
            [
                {"bootstrap": False, "n_estimators": 100},
                {"bootstrap": False, "n_estimators": 200},
                {"bootstrap": True, "n_estimators": 300},
            ]
        )

        self.assertIs(aggregated["bootstrap"], False)
        self.assertEqual(aggregated["n_estimators"], 200)

    def test_normalize_model_params_coerces_random_forest_bool_like_bootstrap(self):
        normalized = _normalize_model_params(
            "RandomForest",
            {"bootstrap": 0, "max_samples": 0.8, "n_estimators": 100},
        )

        self.assertIs(normalized["bootstrap"], False)
        self.assertIsNone(normalized["max_samples"])


if __name__ == "__main__":
    unittest.main()
