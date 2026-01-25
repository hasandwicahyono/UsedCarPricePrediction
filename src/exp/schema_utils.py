import re
import pandas as pd
from typing import Tuple, List

VALID_NAME = re.compile(r"[^a-zA-Z0-9_]+")

def sanitize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Returns:
      - df with sanitized column names
      - mapping {old_name -> new_name}
    """
    mapping = {}
    used = set()

    for col in df.columns:
        new = str(col)

        # replace invalid chars
        new = VALID_NAME.sub("_", new)

        # if starts with digit → prefix
        if new[0].isdigit():
            new = f"f_{new}"

        # collapse multiple underscores
        new = re.sub(r"_+", "_", new)

        # strip underscores
        new = new.strip("_")

        # avoid collisions
        base = new
        i = 1
        while new in used:
            new = f"{base}_{i}"
            i += 1

        used.add(new)
        mapping[col] = new

    return df.rename(columns=mapping), mapping

def infer_schema(
    df: pd.DataFrame,
    target: str,
    cat_threshold: int = 20
) -> Tuple[str, List[str], List[str]]:
    """
    Heuristic:
    - numeric dtype → numerical
    - object/category → categorical
    - int with low cardinality → categorical
    """
    num_cols = []
    cat_cols = []

    for c in df.columns:
        if c == target:
            continue

        s = df[c]

        if pd.api.types.is_numeric_dtype(s):
            if pd.api.types.is_integer_dtype(s) and s.nunique() <= cat_threshold:
                cat_cols.append(c)
            else:
                num_cols.append(c)
        else:
            cat_cols.append(c)

    return target, num_cols, cat_cols
