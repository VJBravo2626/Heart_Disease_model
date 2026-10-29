import json
import subprocess
import sys
from pathlib import Path

import pytest

from src import config
from src.predict import (EXAMPLE_INPUT, HeartDiseasePredictor, InputValidationError, ModelLoadError,
                         validate_input)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def predictor(quick_artifacts):
    m = quick_artifacts["models"]
    return HeartDiseasePredictor.load(m / "heart_pipeline.joblib", m / "model_metadata.json")


def test_output_format(predictor):
    r = predictor.predict(EXAMPLE_INPUT)
    assert r["prediction"] in (0, 1) and 0.0 <= r["class1_score"] <= 1.0
    assert isinstance(r["score_is_calibrated"], bool) and "NOT" in r["score_note"]
    assert r["prediction"] == int(r["class1_score"] >= r["threshold"])
    assert r["model_version"] == config.MODEL_VERSION and isinstance(r["warnings"], list)


def test_feature_order_independent(predictor):
    shuffled = dict(reversed(list(EXAMPLE_INPUT.items())))
    assert predictor.predict(shuffled)["class1_score"] == predictor.predict(EXAMPLE_INPUT)["class1_score"]


def test_numeric_strings_and_integral_floats_accepted(predictor):
    rec = {**EXAMPLE_INPUT, "age": "55", "cp": 0.0}
    assert predictor.predict(rec)["prediction"] in (0, 1)


@pytest.mark.parametrize("bad,fragment", [
    ({}, "Missing required"),
    ({"extra": 1}, "Unknown feature"),
    ({"age": None}, "Missing required"),
    ({"age": "abc"}, "finite number"),
    ({"age": float("nan")}, "finite number"),
    ({"age": float("inf")}, "finite number"),
    ({"age": True}, "finite number"),
    ({"age": 5}, "between"),
    ({"age": 55.5}, "whole number"),
    ({"chol": 9999}, "between"),
    ({"oldpeak": -1}, "between"),
    ({"sex": 2}, "one of"),
    ({"cp": 7}, "one of"),
    ({"thal": 0}, "one of"),
    ({"cp": 1.5}, "one of"),
])
def test_invalid_inputs_rejected(predictor, bad, fragment):
    rec = {**EXAMPLE_INPUT, **bad}
    if "extra" in bad:
        rec = {**EXAMPLE_INPUT, "extra": 1}
    if bad == {}:
        rec = {}
    with pytest.raises(InputValidationError, match=fragment):
        predictor.predict(rec)


def test_non_mapping_rejected():
    with pytest.raises(InputValidationError):
        validate_input([1, 2, 3])


def test_all_errors_reported_together():
    with pytest.raises(InputValidationError) as e:
        validate_input({**EXAMPLE_INPUT, "age": 5, "cp": 9})
    assert len(e.value.errors) == 2


def test_out_of_training_range_warns(predictor):
    assert predictor.predict({**EXAMPLE_INPUT, "age": 95})["warnings"]


def test_tampered_model_rejected(quick_artifacts, tmp_path):
    m = quick_artifacts["models"]
    (tmp_path / "model_metadata.json").write_text((m / "model_metadata.json").read_text())
    (tmp_path / "heart_pipeline.joblib").write_bytes((m / "heart_pipeline.joblib").read_bytes() + b"x")
    with pytest.raises(ModelLoadError, match="checksum"):
        HeartDiseasePredictor.load(tmp_path / "heart_pipeline.joblib", tmp_path / "model_metadata.json")


def test_missing_model_message(tmp_path):
    with pytest.raises(ModelLoadError, match="python -m src.train"):
        HeartDiseasePredictor.load(tmp_path / "a.joblib", tmp_path / "b.json")


def test_load_and_predict_in_fresh_process(quick_artifacts):
    m = quick_artifacts["models"]
    code = (
        "import json;from src.predict import HeartDiseasePredictor, EXAMPLE_INPUT;"
        f"p=HeartDiseasePredictor.load(r'{m / 'heart_pipeline.joblib'}', r'{m / 'model_metadata.json'}');"
        "print(json.dumps(p.predict(EXAMPLE_INPUT)))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout.strip().splitlines()[-1])["prediction"] in (0, 1)
