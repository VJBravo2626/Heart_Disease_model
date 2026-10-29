import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def quick_artifacts(tmp_path_factory):
    """Train once in quick mode into a temp dir (never touches the committed artifacts)."""
    from src import train
    base = tmp_path_factory.mktemp("artifacts")
    models, reports = base / "models", base / "reports"
    metrics = train.run_training(models, reports, quick=True)
    return {"models": models, "reports": reports, "metrics": metrics}
