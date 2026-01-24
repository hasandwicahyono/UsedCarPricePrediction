from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

@dataclass
class DataReadConfig:
    root_dir: str = "Dataset/data"
    pattern: str = "*.csv"          # only csv for now
    recursive: bool = True

    # cleaning
    strip_colnames: bool = True
    lowercase_colnames: bool = False

    # filtering
    exclude_filenames: Optional[List[str]] = None  # exact filenames to exclude (e.g., ["cars.csv"])

    # provenance
    add_source_column: bool = True
    source_column_name: str = "_source_file"

    # reading behavior
    encoding: Optional[str] = None  # e.g., "utf-8", "latin-1"
    sep: Optional[str] = None       # if None, pandas will infer default ','


def _normalize_columns(df: pd.DataFrame, cfg: DataReadConfig) -> pd.DataFrame:
    cols = list(df.columns)
    if cfg.strip_colnames:
        cols = [c.strip() for c in cols]
    if cfg.lowercase_colnames:
        cols = [c.lower() for c in cols]
    df.columns = cols
    return df


def read_csv_folder(cfg: DataReadConfig) -> pd.DataFrame:
    """
    Read all CSV files under cfg.root_dir and concatenate into one dataframe.

    Notes:
    - This does NOT do train/test split (kept separate).
    - It only assembles data.
    """
    root = Path(cfg.root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data root_dir not found: {root.resolve()}")

    exclude = set(cfg.exclude_filenames or [])

    # file discovery
    if cfg.recursive:
        files = sorted(root.rglob(cfg.pattern))
    else:
        files = sorted(root.glob(cfg.pattern))

    files = [p for p in files if p.is_file() and p.name not in exclude]

    if not files:
        raise FileNotFoundError(
            f"No files matched in {root.resolve()} with pattern={cfg.pattern} (recursive={cfg.recursive})."
        )

    frames = []
    for p in files:
        df = pd.read_csv(p, encoding=cfg.encoding, sep=cfg.sep)

        # Rename the 'tax(£)' column to 'tax' if it exists.
        if 'tax(£)' in df.columns:
            df = df.rename(columns={'tax(£)': 'tax'})
        
        df = _normalize_columns(df, cfg)
        if cfg.add_source_column:
            df[cfg.source_column_name] = str(p.as_posix())
        frames.append(df)

    out = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    return out


def coerce_dtypes(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    errors: str = "coerce"
) -> pd.DataFrame:
    """
    Lightweight dtype normalization to reduce silent issues:
    - numeric cols -> to_numeric
    - categorical cols -> astype('category') (optional)
    """
    df = df.copy()

    if numeric_cols:
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors=errors)

    if categorical_cols:
        for c in categorical_cols:
            if c in df.columns:
                df[c] = df[c].astype("category")

    return df


def basic_clean(
    df: pd.DataFrame,
    target: str,
    dropna_target: bool = True
) -> pd.DataFrame:
    """
    Basic cleaning rules that are safe and generic.
    - Optionally drop rows with missing target.
    """
    df = df.copy()
    if dropna_target and target in df.columns:
        df = df.dropna(subset=[target])
    return df
