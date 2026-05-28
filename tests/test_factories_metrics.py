import unittest
import numpy as np
import optuna

from src.exp.factories import build_metric, get_interaction_policy, get_preprocess_policy
from src.exp.facade import aggregate_hyperparams, _normalize_model_params
from src.exp.interpretability import ComplementaryExplainer
from src.exp.metrics import MAEMetric, make_metric, METRIC_REGISTRY
from src.exp.models import ModelFactory, ResidualStrategy
from src.exp.plot_manager import PlotManager
from src.exp.preprocess import GeneticFeatureSelector
from src.exp.sensitivity import summarize_hyperparameter_sensitivity
from src.exp.specs import MetricSpec, ModelSpec, PreprocessSpec
from src.exp.tuning import suggest_from_space


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
        with self.assertRaises(ValueError):
            PreprocessSpec(feature_selection_method="weird").validate()

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

    def test_tabnet_registered_with_expected_policies(self):
        self.assertIn("TabNet", ModelFactory.MAP)
        self.assertEqual(
            get_preprocess_policy("TabNet", {}),
            {"cat_encoding": "target", "use_feature_selection": False},
        )
        self.assertEqual(get_interaction_policy("TabNet"), "none")

    def test_ai_residual_alternatives_registered(self):
        self.assertIn("FuzzyLogic", ResidualStrategy.registry)
        self.assertIn("EvolutionaryStacking", ResidualStrategy.registry)

    def test_fuzzy_logic_residual_predicts_flat_residuals(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(20, 4))
        residuals = X[:, 0] * 0.5 - X[:, 1] * 0.25
        model = ResidualStrategy.registry["FuzzyLogic"](
            seed=42,
            n_rules=3,
            fuzziness=2.0,
            alpha=0.001,
        )

        model.fit(X, residuals)
        pred = model.predict(X[:5])

        self.assertEqual(pred.shape, (5,))
        self.assertTrue(np.isfinite(pred).all())

    def test_evolutionary_stacking_residual_predicts_flat_residuals(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(16, 3))
        residuals = X[:, 0] * 0.3 + rng.normal(scale=0.01, size=16)
        model = ResidualStrategy.registry["EvolutionaryStacking"](
            seed=42,
            max_iter=1,
            population_size=2,
            tree_max_depth=2,
            rf_n_estimators=5,
        )

        model.fit(X, residuals)
        pred = model.predict(X[:4])

        self.assertEqual(pred.shape, (4,))
        self.assertTrue(np.isfinite(pred).all())

    def test_suggest_from_space_respects_integer_step(self):
        trial = optuna.trial.FixedTrial({"n_d": 16})
        params = suggest_from_space(
            trial,
            {"n_d": {"type": "int", "low": 8, "high": 64, "step": 8}},
        )

        self.assertEqual(params["n_d"], 16)

    def test_hyperparameter_sensitivity_summarizes_numeric_and_categorical_params(self):
        import pandas as pd

        trials = pd.DataFrame(
            {
                "model": ["XGBoost"] * 4,
                "base_model": ["XGBoost"] * 4,
                "residual_kind": ["base"] * 4,
                "metric": ["MAE"] * 4,
                "direction": ["minimize"] * 4,
                "state": ["COMPLETE"] * 4,
                "value": [4.0, 3.0, 2.0, 1.0],
                "param__max_depth": [2, 3, 4, 5],
                "param__booster": ["a", "a", "b", "b"],
            }
        )

        summary = summarize_hyperparameter_sensitivity(trials)

        self.assertIn("max_depth", set(summary["parameter"]))
        self.assertIn("booster", set(summary["parameter"]))
        self.assertGreater(
            float(summary.loc[summary["parameter"].eq("max_depth"), "sensitivity"].iloc[0]),
            0.0,
        )

    def test_complementary_explainer_partial_dependence_and_counterfactuals(self):
        class DummyModel:
            def predict(self, X):
                X = np.asarray(X)
                return X[:, 0] * 2.0 + X[:, 1] * 0.5

        X = np.column_stack([
            np.linspace(0.0, 1.0, 12),
            np.linspace(1.0, 2.0, 12),
        ])
        store = [
            {
                "model_name": "Dummy",
                "model_label": "Dummy",
                "model": DummyModel(),
                "X_test": X,
                "feature_names": ["num__x0", "num__x1"],
                "outer_fold": 1,
            }
        ]
        explainer = ComplementaryExplainer(
            store,
            max_samples=12,
            grid_size=5,
            max_entries_per_model=1,
            seed=42,
        )

        pdp = explainer.partial_dependence("Dummy", features=["x0"], grid_size=5)
        cf = explainer.counterfactuals("Dummy", n_instances=2, features=["x0"], desired_delta=0.2)

        self.assertEqual(set(pdp["feature"]), {"x0"})
        self.assertEqual(len(pdp), 5)
        self.assertFalse(cf.empty)
        self.assertTrue((cf["counterfactual_prediction"] >= cf["target_prediction"]).all())

    def test_genetic_feature_selector_selects_valid_subset(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(30, 5))
        y = X[:, 0] * 2.0 - X[:, 2] + rng.normal(scale=0.01, size=30)
        selector = GeneticFeatureSelector(
            population_size=6,
            generations=2,
            mutation_rate=0.1,
            random_state=42,
        )

        Xt = selector.fit_transform(X, y)

        self.assertEqual(selector.get_support().shape, (5,))
        self.assertGreaterEqual(Xt.shape[1], 1)
        self.assertLessEqual(Xt.shape[1], 5)

    def test_metric_plots_with_ci_and_baseline_delta(self):
        import pandas as pd

        results = pd.DataFrame(
            {
                "outer_fold": [1, 2, 3, 1, 2, 3],
                "model": ["RandomForest", "RandomForest", "RandomForest", "XGBoost", "XGBoost", "XGBoost"],
                "MAE": [10.0, 11.0, 9.0, 8.0, 9.0, 7.0],
            }
        )
        pm = PlotManager(base_dir="/private/tmp/newijai-test-plots")
        summary = pm.metric_summary(results, metric="MAE")
        fig_ci = pm.plot_metric_ci(results, metric="MAE", baseline="RandomForest")
        fig_delta = pm.plot_baseline_delta(results, metric="MAE", baseline="RandomForest")

        self.assertEqual(set(summary["model"]), {"RandomForest", "XGBoost"})
        self.assertEqual(len(fig_ci.axes), 1)
        self.assertEqual(len(fig_delta.axes), 1)


if __name__ == "__main__":
    unittest.main()
