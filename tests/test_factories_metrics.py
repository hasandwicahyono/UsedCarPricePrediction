import unittest

from src.exp.factories import build_metric
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


if __name__ == "__main__":
    unittest.main()
