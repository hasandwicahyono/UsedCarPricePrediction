from abc import ABC, abstractmethod

# NOTE: Explainer classes below are registered and instantiated via the SHAP explainer registry/factory.
import shap
from .registry import SHAP_EXPLAINER_REGISTRY

class BaseShapExplainer(ABC):
    registry = SHAP_EXPLAINER_REGISTRY

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        policy = getattr(cls, "policy", None)
        if policy:
            BaseShapExplainer.registry.register(policy, cls)

    def __init__(self, model, X_background):
        self.model = model
        self.X_background = X_background

    @abstractmethod
    def explain(self, X): ...

class TreeShapExplainer(BaseShapExplainer):
    policy = "tree"

    def __init__(self, model, X_background):
        super().__init__(model, X_background)
        self.explainer = shap.TreeExplainer(model)

    def explain(self, X):
        # Disable strict additivity check to avoid failures from small numerical drift.
        return self.explainer.shap_values(X, check_additivity=False)
    
class LinearShapExplainer(BaseShapExplainer):
    policy = "linear"

    def __init__(self, model, X_background):
        super().__init__(model, X_background)
        self.explainer = shap.LinearExplainer(model, X_background)

    def explain(self, X):
        return self.explainer.shap_values(X)

class KernelShapExplainer(BaseShapExplainer):
    policy = "kernel"

    def __init__(self, model, X_background):
        super().__init__(model, X_background)
        self.explainer = shap.KernelExplainer(
            model.predict,
            X_background
        )

    def explain(self, X):
        return self.explainer.shap_values(X)

class ShapExplainerFactory:
    @staticmethod
    def create(policy: str, model, X_background):
        explainer_cls = BaseShapExplainer.registry.get(policy)
        if explainer_cls is None:
            raise ValueError(f"Unknown explain_policy: {policy}")
        return explainer_cls(model, X_background)


class ExplainerFactory:
    @staticmethod
    def create(policy: str, model, X_background):
        return ShapExplainerFactory.create(policy, model, X_background)
