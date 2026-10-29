"""Dataset loading, schema validation, duplicate handling and splitting."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


class SchemaError(ValueError):
    """Raised when the dataset does not match the expected schema."""


def load_raw(path: str | Path = config.DATA_PATH) -> pd.DataFrame:
    """Load the original CSV exactly as shipped (no cleaning)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def validate_schema(df: pd.DataFrame) -> None:
    """Validate columns, dtypes, missing values, label and category codes."""
    expected = set(config.FEATURES) | {config.TARGET}
    missing, extra = expected - set(df.columns), set(df.columns) - expected
    if missing or extra:
        raise SchemaError(f"Column mismatch. Missing: {sorted(missing)}; unexpected: {sorted(extra)}")
    if df[list(expected)].isna().any().any():
        bad = df[list(expected)].columns[df[list(expected)].isna().any()].tolist()
        raise SchemaError(f"Missing values found in: {bad}")
    for col in expected:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise SchemaError(f"Column '{col}' must be numeric, got {df[col].dtype}")
    if not set(df[config.TARGET].unique()) <= {0, 1}:
        raise SchemaError("Target must contain only 0/1.")
    for col in config.BINARY + config.CATEGORICAL:
        unexpected = set(df[col].unique()) - set(config.data_choices(col))
        if unexpected:
            raise SchemaError(f"Column '{col}' has unexpected codes: {sorted(unexpected)}")
    for col in config.NUMERIC:
        spec = config.FEATURE_SPEC[col]
        if df[col].min() < spec["min"] or df[col].max() > spec["max"]:
            raise SchemaError(f"Column '{col}' has values outside [{spec['min']}, {spec['max']}]")


def duplicate_summary(df: pd.DataFrame) -> dict:
    """Quantify exact duplicate rows (all 14 columns identical)."""
    dup_mask = df.duplicated(keep="first")
    group_sizes = df[df.duplicated(keep=False)].groupby(list(df.columns)).size()
    feats = config.FEATURES
    uniq = df.drop_duplicates()
    return {
        "rows": int(len(df)),
        "unique_rows": int(len(uniq)),
        "redundant_rows": int(dup_mask.sum()),
        "redundant_fraction": float(dup_mask.mean()),
        "rows_in_a_duplicated_group": int(df.duplicated(keep=False).sum()),
        "max_copies_of_one_record": int(group_sizes.max()) if len(group_sizes) else 1,
        "mean_copies_of_duplicated_records": float(group_sizes.mean()) if len(group_sizes) else 1.0,
        "unique_rows_with_same_features_but_different_label": int(
            uniq.duplicated(subset=feats, keep=False).sum()),
    }


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the first copy of each fully identical row."""
    return df.drop_duplicates(keep="first").reset_index(drop=True)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X in canonical feature order, y)."""
    return df[config.FEATURES].copy(), df[config.TARGET].copy()


def make_split(
    df_unique: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    seed: int = config.RANDOM_STATE,
):
    """Stratified, seeded train/test split of an already de-duplicated frame."""
    X, y = split_features_target(df_unique)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
