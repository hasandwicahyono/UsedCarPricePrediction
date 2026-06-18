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

    @staticmethod
    def _as_2d_values(values):
        # Normalize SHAP outputs across explainers/versions:
        # - Explanation -> .values
        # - list (multi-output) -> first output
        # - 3D arrays (n, f, o) -> first output
        if hasattr(values, "values"):
            values = values.values
        if isinstance(values, list):
            values = values[0] if values else values
        if values is None:
            raise ValueError("SHAP explainer returned no values")
        if getattr(values, "ndim", 0) == 3:
            values = values[:, :, 0]
        return values

    @abstractmethod
    def explain(self, X): ...

class TreeShapExplainer(BaseShapExplainer):
    policy = "tree"

    def __init__(self, model, X_background):
        super().__init__(model, X_background)
        self._use_call_api = False
        try:
            self.explainer = shap.TreeExplainer(model)
        except Exception:
            # Some SHAP/sklearn combinations reject otherwise tree-based models.
            # Fall back to the generic explainer API to keep analysis running.
            self.explainer = shap.Explainer(model.predict, X_background)
            self._use_call_api = True

    def explain(self, X):
        if self._use_call_api:
            return self._as_2d_values(self.explainer(X))
        # Disable strict additivity check to avoid failures from small numerical drift.
        return self._as_2d_values(self.explainer.shap_values(X, check_additivity=False))
    
class LinearShapExplainer(BaseShapExplainer):
    policy = "linear"

    def __init__(self, model, X_background):
        super().__init__(model, X_background)
        self.explainer = shap.LinearExplainer(model, X_background)

    def explain(self, X):
        return self._as_2d_values(self.explainer.shap_values(X))

class KernelShapExplainer(BaseShapExplainer):
    policy = "kernel"

    def __init__(self, model, X_background):
        super().__init__(model, X_background)

        if hasattr(model, "predict"):
            predict_fn = model.predict
        elif callable(model):
            def predict_fn(x):
                import numpy as np
                try:
                    import torch
                    if isinstance(model, torch.nn.Module):
                        model.eval()
                        device = next(model.parameters()).device
                        with torch.no_grad():
                            x_tensor = torch.tensor(np.asarray(x), dtype=torch.float32).to(device)
                            
                            if type(model).__name__ == "WideDeep":
                                x_in = {}
                                for comp in ["wide", "deeptabular", "deeptext", "deepimage"]:
                                    if getattr(model, comp, None) is not None:
                                        x_in[comp] = x_tensor
                                if not x_in:
                                    x_in["deeptabular"] = x_tensor
                                out = model(x_in)
                            else:
                                out = model(x_tensor)
                                
                            if isinstance(out, torch.Tensor):
                                return out.cpu().numpy()
                            if isinstance(out, (tuple, list)) and isinstance(out[0], torch.Tensor):
                                return out[0].cpu().numpy()
                            return np.asarray(out)
                except ImportError:
                    pass
                return np.asarray(model(x))
        else:
            predict_fn = model

        self.explainer = shap.KernelExplainer(
            predict_fn,
            X_background
        )

    def explain(self, X):
        return self._as_2d_values(self.explainer.shap_values(X))

class DeepShapExplainer(BaseShapExplainer):
    policy = "deep"

    def __init__(self, model, X_background):
        super().__init__(model, X_background)
        self.explainer = shap.DeepExplainer(model, X_background)

    def explain(self, X):
        values = self.explainer.shap_values(X, check_additivity=False)
        return self._as_2d_values(values)

class GradientShapExplainer(BaseShapExplainer):
    policy = "gradient"

    def __init__(self, model, X_background):
        super().__init__(model, X_background)
        self.explainer = shap.GradientExplainer(model, X_background)

    def explain(self, X):
        values = self.explainer.shap_values(X)
        return self._as_2d_values(values)

class ShapExplainerFactory:
    @staticmethod
    def create(policy: str, model, X_background):
        explainer_cls = BaseShapExplainer.registry.get(policy)
        if explainer_cls is None:
            raise ValueError(f"Unknown explain_policy: {policy}")
        return explainer_cls(model, X_background)
