from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class MetricSpec:
    name: str

    def normalize(self) -> "MetricSpec":
        return MetricSpec(name=self.name.strip().lower())


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    residual_cfgs: Optional[list] = None

    def normalize(self) -> "ModelSpec":
        return ModelSpec(
            name=self.name,
            params=dict(self.params),
            residual_cfgs=self.residual_cfgs,
        )

    def validate(self) -> None:
        if not self.name:
            raise ValueError("ModelSpec.name is required")


@dataclass(frozen=True)
class PreprocessSpec:
    cat_encoding: str = "onehot"
    use_feature_selection: bool = False
    te_smoothing: float = 10.0
    te_min_samples_leaf: int = 1
    te_noise_std: float = 0.0
    seed: int = 42
    run_id: Optional[str] = None
    outer_fold: Optional[int] = None
    inner_fold: Optional[int] = None
    model_name: Optional[str] = None
    trial_id: Optional[int] = None

    def normalize(self) -> "PreprocessSpec":
        return PreprocessSpec(
            cat_encoding=self.cat_encoding.strip().lower(),
            use_feature_selection=bool(self.use_feature_selection),
            te_smoothing=float(self.te_smoothing),
            te_min_samples_leaf=int(self.te_min_samples_leaf),
            te_noise_std=float(self.te_noise_std),
            seed=int(self.seed),
            run_id=None if self.run_id is None else str(self.run_id),
            outer_fold=None if self.outer_fold is None else int(self.outer_fold),
            inner_fold=None if self.inner_fold is None else int(self.inner_fold),
            model_name=None if self.model_name is None else str(self.model_name),
            trial_id=None if self.trial_id is None else int(self.trial_id),
        )

    def validate(self) -> None:
        if self.cat_encoding not in {"onehot", "target", "raw"}:
            raise ValueError(f"Unknown cat_encoding: {self.cat_encoding}")
