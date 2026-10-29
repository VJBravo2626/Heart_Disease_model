"""Reusable prediction module: strict input validation + loading of the saved pipeline.

Only load model files you trained yourself: joblib/pickle can execute arbitrary code.
`load()` verifies the SHA-256 recorded in the metadata before deserialising.

CLI (from the repository root):
    python -m src.predict --example
    python -m src.predict --input patient.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Mapping

import joblib
import pandas as pd

from . import config, data


class InputValidationError(ValueError):
    """Raised for invalid user input; `.errors` lists every problem found."""

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


class ModelLoadError(RuntimeError):
    """Raised when the model artifact is missing, altered or cannot be deserialised."""


EXAMPLE_INPUT = {"age": 55, "sex": 1, "cp": 0, "trestbps": 130, "chol": 240, "fbs": 0, "restecg": 1,
                 "thalach": 150, "exang": 0, "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2}


def _as_number(value):
    """Return a finite float for int/float/numeric-string input, else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None
    if isinstance(value, Real):
        value = float(value)
        return value if math.isfinite(value) else None
    return None


def validate_input(record: Mapping, training_ranges: dict | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Validate one record. Returns (single-row DataFrame in canonical column order, warnings).
    Raises InputValidationError listing all problems (missing, unknown, wrong type, out of range)."""
    if not isinstance(record, Mapping):
        raise InputValidationError(["Input must be a mapping of feature name -> value."])
    errors, warnings, clean = [], [], {}
    missing = [f for f in config.FEATURES if f not in record or record[f] is None]
    unknown = [k for k in record if k not in config.FEATURES]
    if missing:
        errors.append(f"Missing required feature(s): {missing}")
    if unknown:
        errors.append(f"Unknown feature(s): {unknown}. Expected: {config.FEATURES}")
    for f in config.FEATURES:
        if f in missing:
            continue
        spec = config.FEATURE_SPEC[f]
        num = _as_number(record[f])
        if num is None:
            errors.append(f"'{f}' must be a finite number, got {record[f]!r}.")
            continue
        if spec["kind"] == "numeric":
            if num < spec["min"] or num > spec["max"]:
                errors.append(f"'{f}' ({spec['label']}) must be between {spec['min']} and {spec['max']}, got {num:g}.")
                continue
            if spec.get("integer") and num != int(num):
                errors.append(f"'{f}' ({spec['label']}) must be a whole number, got {num:g}.")
                continue
            clean[f] = int(num) if spec.get("integer") else num
            if training_ranges and f in training_ranges:
                lo, hi = training_ranges[f]
                if num < lo or num > hi:
                    warnings.append(f"'{f}'={num:g} is outside the range seen in training data "
                                    f"({lo:g}-{hi:g}); the prediction is an extrapolation.")
        else:
            if num != int(num) or int(num) not in config.feature_choices(f):
                errors.append(f"'{f}' ({spec['label']}) must be one of {config.feature_choices(f)}, got {record[f]!r}.")
                continue
            clean[f] = int(num)
    if errors:
        raise InputValidationError(errors)
    return pd.DataFrame([clean], columns=config.FEATURES), warnings


class HeartDiseasePredictor:
    """Wraps the fitted preprocessing+model pipeline together with its metadata."""

    def __init__(self, estimator, metadata: dict, training_ranges: dict | None = None):
        self.estimator, self.metadata, self.training_ranges = estimator, metadata, training_ranges
        if not hasattr(estimator, "predict_proba"):
            raise ModelLoadError("Loaded estimator does not support predict_proba.")
        if list(metadata.get("feature_names", [])) != config.FEATURES:
            raise ModelLoadError("Feature names in metadata do not match the expected schema.")

    @classmethod
    def load(cls, model_path: str | Path = config.MODEL_PATH,
             metadata_path: str | Path = config.METADATA_PATH) -> "HeartDiseasePredictor":
        model_path, metadata_path = Path(model_path), Path(metadata_path)
        if not model_path.exists() or not metadata_path.exists():
            raise ModelLoadError(f"Model artifacts not found ({model_path}, {metadata_path}). "
                                 "Train first: python -m src.train")
        metadata = json.loads(metadata_path.read_text())
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if digest != metadata.get("artifact_sha256"):
            raise ModelLoadError("Model file checksum does not match its metadata; refusing to load it.")
        try:
            estimator = joblib.load(model_path)
        except Exception as exc:  # incompatible library versions, corrupt file, ...
            env = metadata.get("environment", {})
            raise ModelLoadError(f"Could not load the model ({exc!r}). It was trained with "
                                 f"scikit-learn {env.get('scikit_learn')}; retrain with: python -m src.train") from exc
        ranges = None
        if config.DATA_PATH.exists():
            try:
                df = data.drop_exact_duplicates(data.load_raw())
                ranges = {c: (float(df[c].min()), float(df[c].max())) for c in config.NUMERIC}
            except Exception:
                ranges = None
        return cls(estimator, metadata, ranges)

    def predict(self, record: Mapping) -> dict:
        """Validate `record` and return class, model score and context."""
        X, warnings = validate_input(record, self.training_ranges)
        score = float(self.estimator.predict_proba(X[config.FEATURES])[0, 1])
        threshold = float(self.metadata["threshold"])
        pred = int(score >= threshold)
        calibrated = bool(self.metadata.get("calibrated", False))
        return {
            "prediction": pred,
            "label": config.POSITIVE_LABEL_TEXT if pred else config.NEGATIVE_LABEL_TEXT,
            "class1_score": score,
            "score_is_calibrated": calibrated,
            "score_note": ("Sigmoid-calibrated on a small (n~300) public dataset; still NOT a medically "
                           "validated probability of disease." if calibrated else
                           "Raw model score; NOT a calibrated or medically valid probability of disease."),
            "threshold": threshold,
            "warnings": warnings,
            "model_version": self.metadata.get("model_version"),
        }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Predict with the saved model (educational use only).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", type=Path, help="JSON file with the 13 features")
    g.add_argument("--example", action="store_true", help="use a built-in example record")
    a = p.parse_args(argv)
    record = EXAMPLE_INPUT if a.example else json.loads(a.input.read_text())
    try:
        print(json.dumps(HeartDiseasePredictor.load().predict(record), indent=2))
    except (InputValidationError, ModelLoadError) as exc:
        raise SystemExit(f"Error: {exc}")


if __name__ == "__main__":
    main()
