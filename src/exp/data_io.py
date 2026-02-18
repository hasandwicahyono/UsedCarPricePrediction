from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
from .patterns import DataSource

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
    cols = df.columns
    if cfg.strip_colnames:
        cols = cols.str.strip()
    if cfg.lowercase_colnames:
        cols = cols.str.lower()
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
    glob_method = root.rglob if cfg.recursive else root.glob
    files = sorted(p for p in glob_method(cfg.pattern) if p.is_file() and p.name not in exclude)

    if not files:
        raise FileNotFoundError(
            f"No files matched in {root.resolve()} with pattern={cfg.pattern} (recursive={cfg.recursive})."
        )

    read_kwargs = {}
    if cfg.encoding is not None:
        read_kwargs["encoding"] = cfg.encoding
    if cfg.sep is not None:
        read_kwargs["sep"] = cfg.sep

    frames = []
    for p in files:
        df = pd.read_csv(p, **read_kwargs)

        # Rename the 'tax(£)' column to 'tax' if it exists.
        if 'tax(£)' in df.columns:
            df.rename(columns={'tax(£)': 'tax'}, inplace=True)
        
        df = _normalize_columns(df, cfg)
        if cfg.add_source_column:
            df[cfg.source_column_name] = str(p.as_posix())
        frames.append(df)

    out = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    return out


@dataclass
class CsvFolderSource(DataSource):
    cfg: DataReadConfig

    def read(self) -> pd.DataFrame:
        return read_csv_folder(self.cfg)


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
    valid_numeric = [c for c in (numeric_cols or []) if c in df.columns]
    valid_categorical = [c for c in (categorical_cols or []) if c in df.columns]
    if not valid_numeric and not valid_categorical:
        return df.copy()

    df = df.copy()
    if valid_numeric:
        df[valid_numeric] = df[valid_numeric].apply(pd.to_numeric, errors=errors)
    if valid_categorical:
        df[valid_categorical] = df[valid_categorical].astype("category")

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
    if dropna_target and target in df.columns:
        return df.dropna(subset=[target])
    return df.copy()
