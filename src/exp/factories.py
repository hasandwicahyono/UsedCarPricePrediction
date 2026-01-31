from __future__ import annotations

from typing import Optional

from .metrics import MetricStrategy, make_metric
from .models import ModelFactory, ModelStrategy
from .preprocess import PreprocessorBuilder
from .config import FeatureSchema
from .shap_explainers import ShapExplainerFactory
from .specs import MetricSpec, ModelSpec, PreprocessSpec


def build_metric(spec_or_name: MetricSpec | str) -> MetricStrategy:
    if isinstance(spec_or_name, MetricSpec):
        spec = spec_or_name.normalize()
        name = spec.name
    else:
        name = spec_or_name
    return make_metric(name)


def build_model(
    spec_or_name: ModelSpec | str,
    *,
    seed: int,
    params: Optional[dict] = None,
    residual_cfgs: Optional[list] = None,
) -> ModelStrategy:
    if isinstance(spec_or_name, ModelSpec):
        spec = spec_or_name.normalize()
        spec.validate()
        return ModelFactory.create(
            spec.name,
            seed=seed,
            params=spec.params,
            residual_cfgs=spec.residual_cfgs,
        )
    return ModelFactory.create(
        spec_or_name,
        seed=seed,
        params=params or {},
        residual_cfgs=residual_cfgs,
    )


def get_model_class(name: str) -> type:
    return ModelFactory.MAP[name]


def get_preprocess_policy(name: str, default: dict) -> dict:
    cls = get_model_class(name)
    return getattr(cls, "preprocess_policy", default)


def get_model_names():
    return ModelFactory.MAP.keys()


def build_preprocessor(schema: FeatureSchema, spec: Optional[PreprocessSpec] = None):
    builder = PreprocessorBuilder(schema)
    if spec is None:
        return builder.build()
    spec = spec.normalize()
    spec.validate()
    return builder.build(
        cat_encoding=spec.cat_encoding,
        use_feature_selection=spec.use_feature_selection,
        te_smoothing=spec.te_smoothing,
        te_min_samples_leaf=spec.te_min_samples_leaf,
        te_noise_std=spec.te_noise_std,
        seed=spec.seed,
        run_id=spec.run_id,
        outer_fold=spec.outer_fold,
        inner_fold=spec.inner_fold,
        model_name=spec.model_name,
        trial_id=spec.trial_id,
    )


def build_shap_explainer(policy: str, model, X_background):
    return ShapExplainerFactory.create(policy, model, X_background)
