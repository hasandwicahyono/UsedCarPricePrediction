"""Measures single-instance inference latency and artifact size for the
selected best model (XGBoost+PseudoHuber), reporting evidence for the
real-time deployment claim in the manuscript's Economic Impact section.

Run from the project root:
    python -m src.exp.benchmark_latency
"""
from __future__ import annotations

import glob
import json
import os
import platform
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DATA_COLUMNS = ["model", "year", "transmission", "mileage", "fuelType", "tax", "mpg", "engineSize"]


def load_sample(n: int = 500, seed: int = 42) -> pd.DataFrame:
    frames = []
    for f in glob.glob(str(ROOT / "Dataset" / "data" / "*.csv")):
        try:
            df = pd.read_csv(f)
            df.columns = [c.strip() for c in df.columns]
            if set(DATA_COLUMNS).issubset(df.columns):
                frames.append(df[DATA_COLUMNS])
        except Exception:
            continue
    data = pd.concat(frames, ignore_index=True).dropna()
    data["model"] = data["model"].astype(str).str.strip()
    return data.sample(n=min(n, len(data)), random_state=seed).reset_index(drop=True)


def predict_batch(model, X):
    pred = np.asarray(model.predict(X)).reshape(-1)
    pred = np.clip(pred, a_min=None, a_max=15.0)
    return np.exp(pred)


def main(model_name: str = "XGBoost+PseudoHuber", base_model_name: str = "XGBoost",
         n_single: int = 300, n_batch_runs: int = 20) -> dict:
    model_dir = ROOT / "outputs" / "models"
    pre_path = model_dir / f"{base_model_name}_preprocessor.joblib"
    model_path = model_dir / f"{model_name}.joblib"

    pre = joblib.load(pre_path)
    model = joblib.load(model_path)

    sample = load_sample()
    Xp = pre.transform(sample)

    # warm-up
    _ = predict_batch(model, Xp[:5])

    rng = np.random.RandomState(0)
    idx = rng.randint(0, Xp.shape[0], size=n_single)
    single_times_ms = []
    for i in idx:
        row = Xp[i:i + 1]
        t0 = time.perf_counter()
        _ = predict_batch(model, row)
        t1 = time.perf_counter()
        single_times_ms.append((t1 - t0) * 1000.0)
    single_times_ms = np.array(single_times_ms)

    batch_times = []
    for _ in range(n_batch_runs):
        t0 = time.perf_counter()
        _ = predict_batch(model, Xp)
        t1 = time.perf_counter()
        batch_times.append(t1 - t0)
    batch_times = np.array(batch_times)

    result = {
        "model_name": model_name,
        "n_single_instance_runs": n_single,
        "single_instance_latency_ms": {
            "mean": float(single_times_ms.mean()),
            "median": float(np.median(single_times_ms)),
            "p95": float(np.percentile(single_times_ms, 95)),
            "p99": float(np.percentile(single_times_ms, 99)),
            "max": float(single_times_ms.max()),
        },
        "batch_throughput": {
            "batch_size": int(Xp.shape[0]),
            "n_runs": n_batch_runs,
            "mean_batch_time_ms": float(batch_times.mean() * 1000.0),
            "per_instance_batched_ms": float((batch_times.mean() / Xp.shape[0]) * 1000.0),
        },
        "artifact_size_mb": {
            "model": os.path.getsize(model_path) / 1e6,
            "preprocessor": os.path.getsize(pre_path) / 1e6,
        },
        "hardware": {
            "processor": platform.processor() or platform.machine(),
            "platform": platform.platform(),
        },
    }
    return result


if __name__ == "__main__":
    out = main()
    out_dir = ROOT / "outputs" / "deploy"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "latency_benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")
