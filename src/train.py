"""Training pipeline: nested CV model comparison -> selection -> calibration check ->
threshold selection (training data only) -> ONE final evaluation on the untouched test set.

Run from the repository root:  python -m src.train
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_validate,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from . import __version__, config, data, evaluate
from .preprocessing import build_preprocessor

# Used for the "prefer the simpler model when scores are statistically close" rule.
COMPLEXITY_RANK = {"Logistic Regression": 0, "KNN": 1, "Random Forest": 2, "Gradient Boosting": 3}
SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
           "f1": "f1", "roc_auc": "roc_auc", "average_precision": "average_precision"}


def candidate_models(quick: bool = False) -> dict[str, tuple[Pipeline, dict]]:
    """name -> (pipeline, param_grid). Scaling only where the algorithm needs it."""
    rs = config.RANDOM_STATE
    grids = {
        "Logistic Regression": {"clf__C": [0.01, 0.1, 1, 10]} if quick else {"clf__C": np.logspace(-3, 2, 6).tolist()},
        "KNN": {"clf__n_neighbors": [5, 15]} if quick else
               {"clf__n_neighbors": [5, 9, 15, 21, 31], "clf__weights": ["uniform", "distance"]},
        "Random Forest": {"clf__max_depth": [3, None]} if quick else
                         {"clf__max_depth": [3, 5, None], "clf__min_samples_leaf": [1, 3, 5]},
        "Gradient Boosting": {"clf__max_depth": [2]} if quick else
                             {"clf__learning_rate": [0.05, 0.1], "clf__max_depth": [2, 3]},
    }
    n_trees = 50 if quick else 200
    pipes = {
        "Logistic Regression": Pipeline([("prep", build_preprocessor(True)),
                                         ("clf", LogisticRegression(max_iter=5000, random_state=rs))]),
        "KNN": Pipeline([("prep", build_preprocessor(True)), ("clf", KNeighborsClassifier())]),
        "Random Forest": Pipeline([("prep", build_preprocessor(False)),
                                   ("clf", RandomForestClassifier(n_estimators=n_trees, random_state=rs))]),
        "Gradient Boosting": Pipeline([("prep", build_preprocessor(False)),
                                       ("clf", GradientBoostingClassifier(n_estimators=min(n_trees, 100),
                                                                          random_state=rs))]),
    }
    return {k: (pipes[k], grids[k]) for k in pipes}


def _grid_search(pipe, grid, quick):
    return GridSearchCV(pipe, grid, scoring="roc_auc", n_jobs=1,
                        cv=StratifiedKFold(3 if quick else 5, shuffle=True, random_state=config.RANDOM_STATE))


def nested_cv_comparison(X_train, y_train, quick: bool = False) -> dict:
    """Same outer protocol for every candidate; hyper-parameters tuned in the inner loop only."""
    outer = RepeatedStratifiedKFold(n_splits=3 if quick else 5, n_repeats=1 if quick else 3,
                                    random_state=config.RANDOM_STATE)
    results = {}
    for name, (pipe, grid) in candidate_models(quick).items():
        t0 = time.time()
        cv = cross_validate(_grid_search(pipe, grid, quick), X_train, y_train, cv=outer,
                            scoring=SCORING, n_jobs=1)
        n_folds = outer.get_n_splits()
        row = {"n_outer_folds": n_folds, "wall_time_s": round(time.time() - t0, 1),
               "mean_fit_time_s": float(np.mean(cv["fit_time"]))}
        for m in SCORING:
            v = cv[f"test_{m}"]
            row[m] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                      "stderr": float(v.std(ddof=1) / np.sqrt(n_folds))}
        results[name] = row
        print(f"  {name:20s} AUC {row['roc_auc']['mean']:.3f} +/- {row['roc_auc']['std']:.3f} "
              f"acc {row['accuracy']['mean']:.3f}  ({row['wall_time_s']}s)")
    return results


def select_model(cv_results: dict) -> tuple[str, str]:
    """Highest mean nested-CV ROC-AUC; among models within one standard error of the best,
    prefer the simplest. Returns (name, explanation)."""
    best = max(cv_results, key=lambda k: cv_results[k]["roc_auc"]["mean"])
    best_auc, best_se = cv_results[best]["roc_auc"]["mean"], cv_results[best]["roc_auc"]["stderr"]
    close = [k for k, v in cv_results.items() if v["roc_auc"]["mean"] >= best_auc - best_se]
    chosen = min(close, key=lambda k: COMPLEXITY_RANK[k])
    why = (f"Highest nested-CV ROC-AUC: {best} ({best_auc:.3f}). Models within one standard error "
           f"({best_se:.3f}): {sorted(close)}. The simplest of these was chosen: {chosen}.")
    return chosen, why


def calibration_decision(estimator, X_train, y_train, quick: bool) -> dict:
    """Compare out-of-fold Brier score of the raw vs sigmoid-calibrated estimator (training data only)."""
    cv = StratifiedKFold(3 if quick else 5, shuffle=True, random_state=config.RANDOM_STATE + 1)
    raw = cross_val_predict(clone(estimator), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    cal_est = CalibratedClassifierCV(clone(estimator), method="sigmoid", cv=3 if quick else 5)
    cal = cross_val_predict(cal_est, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    b_raw, b_cal = brier_score_loss(y_train, raw), brier_score_loss(y_train, cal)
    use_cal = bool(b_cal < b_raw - 0.002)  # require a meaningful improvement
    scores = cal if use_cal else raw
    return {"use_calibration": use_cal, "oof_brier_raw": float(b_raw), "oof_brier_sigmoid": float(b_cal),
            "oof_auc_raw": float(roc_auc_score(y_train, raw)), "oof_auc_sigmoid": float(roc_auc_score(y_train, cal)),
            "oof_reliability_raw": evaluate.calibration_table(y_train, raw),
            "oof_reliability_sigmoid": evaluate.calibration_table(y_train, cal),
            "_oof_scores": scores, "cal_estimator": cal_est}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_training(models_dir: Path = config.MODELS_DIR, reports_dir: Path = config.REPORTS_DIR,
                 data_path: Path = config.DATA_PATH, quick: bool = False,
                 leakage_seeds: int = 20) -> dict:
    t_start = time.time()
    models_dir, reports_dir = Path(models_dir), Path(reports_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=UserWarning)

    print("[1/7] Loading and validating data")
    raw = data.load_raw(data_path)
    data.validate_schema(raw)
    dup = data.duplicate_summary(raw)
    class_raw = raw[config.TARGET].value_counts().sort_index().to_dict()

    print("[2/7] Leakage audit on the original (raw) data")
    audit = evaluate.leakage_audit(raw, n_seeds=2 if quick else leakage_seeds)

    print("[3/7] De-duplicating and splitting (stratified, seed=%d)" % config.RANDOM_STATE)
    uniq = data.drop_exact_duplicates(raw)
    X_train, X_test, y_train, y_test = data.make_split(uniq)
    key = lambda f: f.astype(str).agg("|".join, axis=1)
    assert not set(key(X_train)) & set(key(X_test)), "train/test overlap after de-duplication"

    print("[4/7] Nested cross-validation on the training set only")
    cv_results = nested_cv_comparison(X_train, y_train, quick)
    chosen, why = select_model(cv_results)
    print("  ->", why)

    print("[5/7] Tuning the selected model, calibration check, threshold (training data only)")
    pipe, grid = candidate_models(quick)[chosen]
    search = _grid_search(pipe, grid, quick).fit(X_train, y_train)
    tuned = search.best_estimator_
    cal = calibration_decision(tuned, X_train, y_train, quick)
    final_estimator = cal.pop("cal_estimator") if cal["use_calibration"] else clone(tuned)
    cal.pop("cal_estimator", None)
    oof_scores = cal.pop("_oof_scores")
    thr = evaluate.best_threshold_f1(y_train, oof_scores)
    threshold = thr["threshold"]
    final_estimator.fit(X_train, y_train)
    train_metrics = evaluate.compute_metrics(y_train, final_estimator.predict_proba(X_train)[:, 1], threshold)
    train_metrics.pop("roc_curve")

    print("[6/7] Final evaluation on the untouched test set (performed once)")
    test_scores = final_estimator.predict_proba(X_test)[:, 1]
    test_metrics = evaluate.compute_metrics(y_test, test_scores, threshold)
    test_metrics["bootstrap"] = evaluate.bootstrap_ci(y_test, test_scores, threshold,
                                                      n_boot=200 if quick else 2000)
    test_metrics["reliability"] = evaluate.calibration_table(y_test, test_scores, n_bins=3)
    perm = permutation_importance(final_estimator, X_test, y_test, scoring="roc_auc",
                                  n_repeats=5 if quick else 30, random_state=config.RANDOM_STATE)
    importance = sorted(
        [{"feature": f, "mean": float(m), "std": float(s)}
         for f, m, s in zip(config.FEATURES, perm.importances_mean, perm.importances_std)],
        key=lambda d: -d["mean"])

    print("[7/7] Saving artifacts")
    model_path = models_dir / "heart_pipeline.joblib"
    joblib.dump(final_estimator, model_path, compress=3)
    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metadata = {
        "model_version": config.MODEL_VERSION, "project_version": __version__, "trained_at_utc": trained_at,
        "selected_model": chosen, "selection_rationale": why, "best_params": search.best_params_,
        "calibrated": cal["use_calibration"], "threshold": threshold,
        "positive_class_text": config.POSITIVE_LABEL_TEXT, "negative_class_text": config.NEGATIVE_LABEL_TEXT,
        "feature_names": config.FEATURES,
        "input_schema": {f: {k: v for k, v in config.FEATURE_SPEC[f].items() if k not in ("help",)}
                         for f in config.FEATURES},
        "training_config": {"random_state": config.RANDOM_STATE, "test_size": config.TEST_SIZE,
                            "dedup": "exact duplicate rows removed (keep first) before splitting",
                            "n_train": int(len(X_train)), "n_test": int(len(X_test)),
                            "outer_cv": "RepeatedStratifiedKFold", "inner_cv": "StratifiedKFold, ROC-AUC",
                            "quick_mode": quick},
        "test_metrics_summary": {k: test_metrics[k] for k in
                                 ("accuracy", "precision_class1", "recall_class1", "f1_class1", "roc_auc")},
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                        "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__},
        "artifact_sha256": _sha256(model_path),
    }
    (models_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2))

    metrics = {
        "trained_at_utc": trained_at, "model_version": config.MODEL_VERSION, "selected_model": chosen,
        "selection_rationale": why,
        "dataset": {"raw_rows": int(len(raw)), "unique_rows": int(len(uniq)), "class_counts_raw": class_raw,
                    "class_counts_unique": uniq[config.TARGET].value_counts().sort_index().to_dict(),
                    "n_train": int(len(X_train)), "n_test": int(len(X_test)), "duplicates": dup},
        "leakage_audit": audit,
        "original_notebook_reported": config.ORIGINAL_NOTEBOOK_REPORTED,
        "nested_cv": cv_results,
        "calibration": cal, "threshold_selection": thr,
        "train_metrics_final_model": train_metrics,
        "test_metrics_final_model": test_metrics,
        "permutation_importance_test": importance,
        "runtime_s": round(time.time() - t_start, 1),
    }
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Done in {metrics['runtime_s']}s. Test accuracy {test_metrics['accuracy']:.3f}, "
          f"ROC-AUC {test_metrics['roc_auc']:.3f}")
    return metrics


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Train the heart-disease model.")
    p.add_argument("--data", type=Path, default=config.DATA_PATH)
    p.add_argument("--models-dir", type=Path, default=config.MODELS_DIR)
    p.add_argument("--reports-dir", type=Path, default=config.REPORTS_DIR)
    p.add_argument("--quick", action="store_true", help="tiny grids / few folds (for smoke tests only)")
    a = p.parse_args(argv)
    run_training(a.models_dir, a.reports_dir, a.data, a.quick)


if __name__ == "__main__":
    main()
