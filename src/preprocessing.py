"""Preprocessing: one ColumnTransformer shared by every candidate model."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def build_preprocessor(scale: bool) -> ColumnTransformer:
    """Numeric features: optional scaling. Nominal codes (cp, restecg, slope, thal): one-hot,
    so arbitrary codes are not treated as ordered measurements. Binary flags pass through.
    `ca` is a genuine vessel count and is treated as numeric.
    The transformer is fitted only inside a Pipeline, i.e. on training folds only."""
    numeric = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", numeric, config.NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), config.CATEGORICAL),
            ("bin", "passthrough", config.BINARY),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
