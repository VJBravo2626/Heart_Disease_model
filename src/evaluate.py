"""Metrics, bootstrap intervals, calibration summaries and the leakage audit."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from . import config


def compute_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    """Threshold-dependent and threshold-free metrics for a binary problem."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    report = classification_report(y_true, y_pred, labels=[0, 1], output_dict=True, zero_division=0)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_class1": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_class1": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_class1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "classification_report": {k: v for k, v in report.items()},
        "roc_curve": {"fpr": fpr.round(4).tolist(), "tpr": tpr.round(4).tolist()},
    }


def bootstrap_ci(y_true, y_score, threshold: float, n_boot: int = 2000,
                 seed: int = config.RANDOM_STATE) -> dict:
    """Percentile 95% bootstrap intervals (accuracy, ROC-AUC) - wide for small test sets."""
    rng = np.random.default_rng(seed)
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    accs, aucs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        accs.append(accuracy_score(y_true[idx], (y_score[idx] >= threshold).astype(int)))
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    lo_hi = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"accuracy_95ci": lo_hi(accs), "roc_auc_95ci": lo_hi(aucs), "n_boot": len(accs)}


def calibration_table(y_true, y_score, n_bins: int = 5) -> list[dict]:
    """Quantile-binned reliability table (mean predicted score vs observed rate)."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_score)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    g = df.groupby("bin").agg(mean_score=("p", "mean"), observed_rate=("y", "mean"), n=("y", "size"))
    return g.round(4).reset_index(drop=True).to_dict(orient="records")


def best_threshold_f1(y_true, y_score, default: float = 0.5, min_gain: float = 0.01) -> dict:
    """Pick a decision threshold on TRAINING-ONLY out-of-fold scores. Maximises F1 but keeps
    the default unless the gain exceeds `min_gain`, to avoid fitting noise."""
    grid = np.round(np.arange(0.20, 0.801, 0.01), 2)
    f1s = np.array([f1_score(y_true, (np.asarray(y_score) >= t).astype(int), zero_division=0) for t in grid])
    f1_default = float(f1_score(y_true, (np.asarray(y_score) >= default).astype(int), zero_division=0))
    best = int(np.argmax(f1s))
    chosen = float(grid[best]) if f1s[best] - f1_default > min_gain else default
    return {"threshold": chosen, "oof_f1_at_default": f1_default,
            "oof_f1_at_best": float(f1s[best]), "best_grid_threshold": float(grid[best])}


def _fit_score(model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    return model.predict(Xte)


def leakage_audit(df_raw: pd.DataFrame, n_seeds: int = 20) -> dict:
    """Quantify the effect of duplicate rows on the original (unscaled, default-parameter) models.

    1. Replicate the notebook's split (np.random.seed(42); 80/20) and measure how many test
       rows have an identical copy in the training set.
    2. Compare accuracy on test rows WITH vs WITHOUT a training copy.
    3. Repeat over `n_seeds` random splits on the raw data vs the de-duplicated data."""
    feats, target = config.FEATURES, config.TARGET
    X, y = df_raw[feats], df_raw[target]

    def models():
        return {
            "Logistic Regression": LogisticRegression(max_iter=100),
            "KNN (k=5)": KNeighborsClassifier(),
            "Random Forest": RandomForestClassifier(),
            "KNN (k=1)": KNeighborsClassifier(n_neighbors=1),
        }

    # 1 + 2: exact notebook replication
    np.random.seed(config.RANDOM_STATE)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2)
    key = lambda f: f.astype(str).agg("|".join, axis=1)
    train_keys = set(key(Xtr))
    has_copy = key(Xte).isin(train_keys).to_numpy()
    replication, by_overlap = {}, {}
    import warnings
    for name, m in models().items():
        np.random.seed(config.RANDOM_STATE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = _fit_score(m, Xtr, ytr, Xte, yte)
        ok = (pred == yte.to_numpy())
        replication[name] = float(ok.mean())
        by_overlap[name] = {
            "acc_test_rows_with_train_copy": float(ok[has_copy].mean()) if has_copy.any() else None,
            "acc_test_rows_without_train_copy": float(ok[~has_copy].mean()) if (~has_copy).any() else None,
        }

    # 3: many seeds, raw vs de-duplicated
    uniq = df_raw.drop_duplicates().reset_index(drop=True)
    Xu, yu = uniq[feats], uniq[target]
    raw_accs = {n: [] for n in models()}
    dedup_accs = {n: [] for n in models()}
    overlap_raw, overlap_dedup = [], []
    for seed in range(n_seeds):
        a, b, c, d = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
        overlap_raw.append(float(key(b).isin(set(key(a))).mean()))
        a2, b2, c2, d2 = train_test_split(Xu, yu, test_size=0.2, random_state=seed, stratify=yu)
        overlap_dedup.append(float(key(b2).isin(set(key(a2))).mean()))
        for name in models():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m1 = models()[name]; m1.set_params(**({"random_state": seed} if "Forest" in name else {}))
                raw_accs[name].append(float((_fit_score(m1, a, c, b, d) == d.to_numpy()).mean()))
                m2 = models()[name]; m2.set_params(**({"random_state": seed} if "Forest" in name else {}))
                dedup_accs[name].append(float((_fit_score(m2, a2, c2, b2, d2) == d2.to_numpy()).mean()))
    summ = lambda v: {"mean": float(np.mean(v)), "std": float(np.std(v))}
    return {
        "notebook_split_replication": {
            "test_rows": int(len(Xte)),
            "test_rows_with_identical_copy_in_train": int(has_copy.sum()),
            "fraction_test_rows_with_copy_in_train": float(has_copy.mean()),
            "default_model_accuracy": replication,
            "accuracy_by_overlap": by_overlap,
        },
        "multi_seed": {
            "n_seeds": n_seeds,
            "raw_data_random_split": {n: summ(v) for n, v in raw_accs.items()},
            "deduplicated_stratified_split": {n: summ(v) for n, v in dedup_accs.items()},
            "mean_fraction_test_rows_with_train_copy_raw": float(np.mean(overlap_raw)),
            "mean_fraction_test_rows_with_train_copy_dedup": float(np.mean(overlap_dedup)),
        },
    }
