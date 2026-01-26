from abc import ABC, abstractmethod
import shap

class BaseShapExplainer(ABC):
    registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        policy = getattr(cls, "policy", None)
        if policy:
            BaseShapExplainer.registry[policy] = cls

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
    _CACHE = {}

    @staticmethod
    def create(policy: str, model, X_background):
        key = (policy, type(model))
        if key in ShapExplainerFactory._CACHE:
            return ShapExplainerFactory._CACHE[key]

        explainer_cls = BaseShapExplainer.registry.get(policy)
        if explainer_cls is None:
            raise ValueError(f"Unknown explain_policy: {policy}")
        explainer = explainer_cls(model, X_background)

        ShapExplainerFactory._CACHE[key] = explainer
        return explainer
