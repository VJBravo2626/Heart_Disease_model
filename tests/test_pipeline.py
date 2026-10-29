import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src import config, data, evaluate, train
from src.preprocessing import build_preprocessor


@pytest.fixture(scope="module")
def split():
    return data.make_split(data.drop_exact_duplicates(data.load_raw()))


def test_scaler_fitted_on_training_data_only(split):
    Xtr, Xte, ytr, _ = split
    prep = build_preprocessor(True).fit(Xtr)
    scaler = prep.named_transformers_["num"]
    assert np.allclose(scaler.mean_, Xtr[config.NUMERIC].mean().to_numpy())
    assert not np.allclose(scaler.mean_, pd.concat([Xtr, Xte])[config.NUMERIC].mean().to_numpy())


def test_onehot_for_nominal_codes_and_unknown_ignored(split):
    Xtr, Xte, *_ = split
    prep = build_preprocessor(False).fit(Xtr)
    out = prep.transform(Xte)
    names = prep.get_feature_names_out()
    assert any(n.startswith("cp_") for n in names) and "cp" not in names
    unseen = Xte.iloc[:1].copy(); unseen["cp"] = 99
    assert prep.transform(unseen).shape[1] == out.shape[1]  # no exception


def test_scaling_only_for_scale_sensitive_models():
    cands = train.candidate_models(quick=True)
    for name in ("Logistic Regression", "KNN"):
        assert type(cands[name][0].named_steps["prep"].transformers[0][1]).__name__ == "StandardScaler"
    for name in ("Random Forest", "Gradient Boosting"):
        assert cands[name][0].named_steps["prep"].transformers[0][1] == "passthrough"


def test_all_candidates_fit_and_predict_proba(split):
    Xtr, Xte, ytr, _ = split
    for name, (pipe, _) in train.candidate_models(quick=True).items():
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)
        assert p.shape == (len(Xte), 2) and np.allclose(p.sum(axis=1), 1)


def test_select_model_prefers_simpler_within_one_se():
    mk = lambda m, se: {"roc_auc": {"mean": m, "stderr": se}}
    res = {"Logistic Regression": mk(0.90, 0.02), "Random Forest": mk(0.91, 0.02), "KNN": mk(0.80, 0.02),
           "Gradient Boosting": mk(0.85, 0.02)}
    assert train.select_model(res)[0] == "Logistic Regression"
    res["Logistic Regression"] = mk(0.80, 0.02)
    assert train.select_model(res)[0] == "Random Forest"


def test_threshold_keeps_default_unless_clear_gain():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = np.clip(y * 0.3 + rng.normal(0.35, 0.15, 200), 0, 1)
    out = evaluate.best_threshold_f1(y, s)
    assert 0.2 <= out["threshold"] <= 0.8 and "oof_f1_at_default" in out


def test_compute_metrics_confusion_matrix():
    m = evaluate.compute_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], 0.5)
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert m["accuracy"] == 0.5 and m["roc_auc"] == 0.75


def test_leakage_audit_detects_duplicates():
    audit = evaluate.leakage_audit(data.load_raw(), n_seeds=1)
    assert audit["notebook_split_replication"]["fraction_test_rows_with_copy_in_train"] > 0.9
    assert audit["multi_seed"]["mean_fraction_test_rows_with_train_copy_dedup"] == 0.0


def test_training_artifacts_and_serialization(quick_artifacts):
    models, reports, metrics = (quick_artifacts[k] for k in ("models", "reports", "metrics"))
    meta = json.loads((models / "model_metadata.json").read_text())
    assert meta["feature_names"] == config.FEATURES and 0 < meta["threshold"] < 1
    assert meta["artifact_sha256"] and meta["environment"]["scikit_learn"]
    est = joblib.load(models / "heart_pipeline.joblib")
    X = data.load_raw()[config.FEATURES].head(5)
    assert est.predict_proba(X).shape == (5, 2)
    saved = json.loads((reports / "metrics.json").read_text())
    assert saved["test_metrics_final_model"]["n"] == saved["dataset"]["n_test"]
    assert set(saved["nested_cv"]) == {"Logistic Regression", "KNN", "Random Forest", "Gradient Boosting"}
    assert saved["dataset"]["unique_rows"] == 302
    assert metrics["selected_model"] in saved["nested_cv"]
