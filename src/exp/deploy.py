from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Tuple

import joblib
import numpy as np
from tensorflow.keras.models import load_model


def load_best_model_name(artifact_path: str = "outputs/artifacts/best_model.json") -> str:
    path = Path(artifact_path)
    if not path.exists():
        raise FileNotFoundError(f"Best-model artifact not found: {path.resolve()}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    name = payload.get("best_model")
    if not name:
        raise ValueError("best_model not found in artifact payload.")
    return name


def load_model_artifacts(
    model_name: str,
    model_dir: str = "outputs/models",
) -> Tuple[object, object, bool]:
    model_dir = Path(model_dir)
    pre = joblib.load(model_dir / f"{model_name}_preprocessor.joblib")
    if model_name == "NeuralNetwork":
        model = load_model(model_dir / f"{model_name}.keras")
        is_keras = True
    else:
        model = joblib.load(model_dir / f"{model_name}.joblib")
        is_keras = False
    return model, pre, is_keras


def make_predictor(
    *,
    model_name: str | None = None,
    model_dir: str = "outputs/models",
    artifact_path: str = "outputs/artifacts/best_model.json",
    log_target: bool = True,
) -> Tuple[Callable, object, object]:
    if model_name is None:
        model_name = load_best_model_name(artifact_path)

    model, pre, is_keras = load_model_artifacts(model_name, model_dir=model_dir)

    def predict(X):
        Xp = pre.transform(X)
        if is_keras:
            pred = model.predict(Xp, verbose=0).reshape(-1)
        else:
            pred = np.asarray(model.predict(Xp)).reshape(-1)
        return np.exp(pred) if log_target else pred

    return predict, model, pre
