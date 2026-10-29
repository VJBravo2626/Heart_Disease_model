"""Central configuration: paths, feature schema, seeds and domain constraints."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "heart.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODEL_PATH = MODELS_DIR / "heart_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
METRICS_PATH = REPORTS_DIR / "metrics.json"

MODEL_VERSION = "1.0.0"
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "target"
POSITIVE_LABEL_TEXT = "Class 1 (labelled 'heart disease present' in the dataset)"
NEGATIVE_LABEL_TEXT = "Class 0 (labelled 'no heart disease' in the dataset)"

# Original column order of heart.csv (also the model's expected feature order).
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]  # ca = vessel count
BINARY = ["sex", "fbs", "exang"]
CATEGORICAL = ["cp", "restecg", "slope", "thal"]  # nominal codes -> one-hot

# Per-feature schema. "min"/"max" are hard input limits for numeric features;
# "choices" are the codes accepted at prediction time; "data_choices" (if
# present) are the codes that actually occur in heart.csv (a superset).
# Labels follow the commonly circulated data dictionary of this dataset;
# several code meanings are disputed (see README), so they are shown with
# that caveat in the app and thal uses neutral labels.
FEATURE_SPEC: dict[str, dict] = {
    "age": {"label": "Age", "unit": "years", "kind": "numeric", "min": 18, "max": 100, "integer": True,
            "help": "Age in years."},
    "sex": {"label": "Sex", "kind": "binary", "choices": {0: "Female (0)", 1: "Male (1)"},
            "help": "Coded 0/1 in the dataset."},
    "cp": {"label": "Chest pain type", "kind": "categorical",
           "choices": {0: "0 - typical angina", 1: "1 - atypical angina",
                       2: "2 - non-anginal pain", 3: "3 - asymptomatic"},
           "help": "Code descriptions follow the common data dictionary for this dataset."},
    "trestbps": {"label": "Resting blood pressure", "unit": "mm Hg", "kind": "numeric",
                 "min": 70, "max": 250, "integer": True, "help": "Resting systolic blood pressure."},
    "chol": {"label": "Serum cholesterol", "unit": "mg/dl", "kind": "numeric",
             "min": 100, "max": 700, "integer": True, "help": "Serum cholesterol."},
    "fbs": {"label": "Fasting blood sugar > 120 mg/dl", "kind": "binary",
            "choices": {0: "No (0)", 1: "Yes (1)"}, "help": "Whether fasting blood sugar exceeds 120 mg/dl."},
    "restecg": {"label": "Resting ECG result", "kind": "categorical",
                "choices": {0: "0 - normal", 1: "1 - ST-T wave abnormality", 2: "2 - LV hypertrophy"},
                "help": "Resting electrocardiographic result code."},
    "thalach": {"label": "Maximum heart rate achieved", "unit": "bpm", "kind": "numeric",
                "min": 60, "max": 220, "integer": True, "help": "Maximum heart rate during the exercise test."},
    "exang": {"label": "Exercise-induced angina", "kind": "binary",
              "choices": {0: "No (0)", 1: "Yes (1)"}, "help": "Angina induced by exercise."},
    "oldpeak": {"label": "ST depression (oldpeak)", "unit": "mm", "kind": "numeric",
                "min": 0.0, "max": 10.0, "integer": False,
                "help": "ST depression induced by exercise relative to rest."},
    "slope": {"label": "Slope of peak-exercise ST segment", "kind": "categorical",
              "choices": {0: "0 - upsloping", 1: "1 - flat", 2: "2 - downsloping"},
              "help": "Code descriptions follow the common data dictionary for this dataset."},
    "ca": {"label": "Major vessels coloured by fluoroscopy", "kind": "numeric",
           "min": 0, "max": 4, "integer": True,
           "help": "Count (0-4). Value 4 occurs in the data but is not a standard value."},
    "thal": {"label": "Thalassemia test result (code)", "kind": "categorical",
             "choices": {1: "Code 1", 2: "Code 2", 3: "Code 3"}, "data_choices": [0, 1, 2, 3],
             "help": "Code meanings differ between published versions of this dataset; "
                     "neutral labels are used. Code 0 occurs in 2 unique training rows but is undocumented."},
}

# Values reported in the ORIGINAL notebook's saved outputs (verified by reading
# the notebook JSON). They are kept for comparison only - they are inflated by
# duplicate rows and, for tuned KNN, by test-set selection.
ORIGINAL_NOTEBOOK_REPORTED = {
    "Logistic Regression (default, 80/20 split)": 0.7853658536585366,
    "KNN (default, 80/20 split)": 0.7317073170731707,
    "Random Forest (default, 80/20 split)": 0.9853658536585366,
    "KNN (best k chosen on the TEST set)": 0.98536585,
    "Random Forest (RandomizedSearchCV, test score)": 0.9024390243902439,
    "Logistic Regression (GridSearchCV, test score)": 0.7853658536585366,
    "Logistic Regression (5-fold CV accuracy on all 1,025 rows)": 0.8478048780487806,
}


def feature_choices(name: str) -> list[int]:
    """Codes accepted at prediction time for a binary/categorical feature."""
    return list(FEATURE_SPEC[name]["choices"].keys())


def data_choices(name: str) -> list[int]:
    """Codes allowed when validating the training CSV."""
    spec = FEATURE_SPEC[name]
    return list(spec.get("data_choices", spec["choices"].keys()))
