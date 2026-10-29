"""Streamlit smoke tests using the committed model artifacts (skipped if they are absent)."""
from pathlib import Path

import pytest

from src import config

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")
needs_artifacts = pytest.mark.skipif(
    not (config.MODEL_PATH.exists() and config.METRICS_PATH.exists()),
    reason="run `python -m src.train` first")


def run(page=None):
    at = AppTest.from_file(APP, default_timeout=60).run()
    if page:
        at.sidebar.radio[0].set_value(page).run()
    return at


@needs_artifacts
@pytest.mark.parametrize("page", ["Home", "Prediction", "Model Performance", "Data Insights", "About & Limitations"])
def test_every_page_renders_without_exception(page):
    at = run(page)
    assert not at.exception, [e.value for e in at.exception]
    assert at.warning or at.title  # disclaimer / title present


@needs_artifacts
def test_prediction_flow():
    at = run("Prediction")
    at.button[0].click().run()
    assert not at.exception
    assert any("Model output" in i.value for i in at.info)
    assert any("Model score" in m.label for m in at.metric)
