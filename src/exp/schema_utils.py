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

def clean_feature_name(name: str) -> str:
    """
    Final SHAP feature name sanitizer.
    """
    # pat = re.compile(
    #     r"^(?:cat_te_|cat__|cat_|num_)"   # 1) leading prefixes
    #     r"|(?:^|_)te_"                    # 2) leftover te_
    #     r"|(?:_te|_tr|_test|_train)$"     # 3) trailing suffixes
    #     r"| {2,}"                         # 4) multiple spaces
    #     r"|_{2,}"                         # 5) multiple underscores
    #     r"|^_|_$"                         # 6) leading/trailing underscore
    #     r"|_ "                            # 7) underscore-space -> underscore
    # )

    # repl = lambda m: (
    #     "_" if m.group(0) == "_ "
    #     else "_" if (m.group(0).startswith("_") and set(m.group(0)) == {"_"} and len(m.group(0)) > 1)
    #     else " " if m.group(0).strip() == ""
    #     else ""
    # )

    # return re.sub(pat, repl, name)

    # # 4. Replace double-space with single space
    name = name.replace("  ", " ")

    # # 7. Replace underscore-space with underscore
    name = name.replace("_ ", "_")

    # # 8. Replace space with underscore
    name = name.replace(" ", "_")

    # # 5. Collapse multiple underscores
    name = re.sub(r"__+", "_", name)

    # # 1. Remove known prefixes anywhere at the start
    name = re.sub(r"^(cat_te_|cat__|cat_|num_)", "", name)

    # # 2. Remove leftover 'te_' if it survived
    name = re.sub(r"(^|_)te_", r"\1", name)

    # # 3. Remove train/test suffixes
    name = re.sub(r"(_te|_tr|_test|_train)$", "", name)
    
    # # 6. Remove leading/trailing underscores
    name = name.strip("_")

    return name