# Heart Disease Prediction

An **educational** machine-learning project: classical classifiers trained on a public heart-disease dataset, an honest evaluation protocol, a reusable prediction module and a Streamlit app.

> **Not a medical device.** This project cannot diagnose, rule out, or independently assess heart disease and must not replace evaluation by a qualified healthcare professional.

## 1. Overview

The original notebook (first Kaggle project) reported **98.5 %** accuracy for a Random Forest. That number was inflated by data leakage: `heart.csv` has 1,025 rows but only **302 unique records**, and 97 % of the notebook's test rows had an identical copy in the training set. This version fixes the evaluation, keeps the exploratory notebook for reference, and wraps everything in a reproducible pipeline and app.

## 2. Features

- Schema-validated data loading and explicit duplicate handling (`src/data.py`)
- `Pipeline` + `ColumnTransformer` preprocessing (scaling only where needed, one-hot for nominal codes)
- Four candidates (Logistic Regression, KNN, Random Forest, Gradient Boosting) compared with **nested cross-validation**
- Leakage audit that quantifies the duplicate effect
- Calibration check and decision-threshold selection using training data only; **one** final test evaluation
- Saved pipeline (`joblib`) + JSON metadata with checksum; strict input validation; CLI and Python API
- Five-page Streamlit app; 52 automated tests

## 3. Repository structure

```
├── app.py                  Streamlit app (entry point)
├── heart.csv               original dataset (unchanged)
├── requirements.txt        runtime dependencies
├── requirements-dev.txt    + pytest
├── .streamlit/config.toml  theme, usage stats off
├── src/
│   ├── config.py           paths, seeds, feature schema, domain limits
│   ├── data.py             load, validate, de-duplicate, split
│   ├── preprocessing.py    ColumnTransformer
│   ├── train.py            training / selection / evaluation CLI
│   ├── evaluate.py         metrics, bootstrap CI, calibration, leakage audit
│   └── predict.py          validated prediction API + CLI
├── notebooks/heart-disease-analysis.ipynb   original EDA notebook (banner added, path fixed)
├── models/                 heart_pipeline.joblib + model_metadata.json (committed on purpose)
├── reports/metrics.json    every number shown in the app and below
└── tests/                  test_data.py, test_pipeline.py, test_prediction.py, test_app.py
```

## 4. Dataset

| | |
|---|---|
| File | `heart.csv`, 1,025 rows × 14 columns, no missing values |
| Source | Matches the widely shared Kaggle "Heart Disease" file, understood to derive from the UCI Heart Disease collection (Cleveland subset, ~303 patients). **Provenance and licence could not be verified from the repository** - check the original source's terms before any use beyond education. |
| Target | `target` (0/1), documented as "heart disease present" = 1. See the label-direction caveat below. |
| Features | age, sex, cp (chest-pain type), trestbps (resting BP), chol, fbs, restecg, thalach (max heart rate), exang, oldpeak, slope, ca (vessel count), thal |
| Patients | No identifiers: these are anonymous records, so independence between rows cannot be established. |

**Limitations found during the audit**

- **70.5 % of rows are exact duplicates.** Every one of the 302 unique records appears 3-8 times (mean 3.4). No two unique rows share features with different labels. Repetition this regular is consistent with artificial replication, not repeated clinical observations, so duplicates were treated as erroneous copies.
- **Label direction looks inverted.** On unique records, class 1 has *lower* exercise-induced angina (14 % vs 55 %), ST depression (0.59 vs 1.59), vessel count (0.34 vs 1.17) and *higher* max heart rate (158 vs 139) than class 0 - the opposite of clinical expectation for disease. This could not be verified offline, so the app reports "Class 0/1" and never makes clinical statements.
- Undocumented codes: `ca = 4` (4 unique rows) and `thal = 0` (2 unique rows). Code meanings for cp/restecg/slope/thal differ between published versions of this dataset.
- Small (302), old, single-source, ~70 % male: generalisation to other populations is unknown and bias is unmeasured.

## 5. Setup

Python 3.11+ (tested with 3.12).

**Windows PowerShell**
```powershell
git clone https://github.com/VJBravo2626/Heart_Disease_model.git
cd Heart_Disease_model
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements-dev.txt
```
**macOS / Linux**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```
All commands below run from the repository root.

## 6. Training

```powershell
python -m src.train
```
Takes ~4 minutes on one CPU core (Random Forest dominates). `--quick` runs a tiny smoke-test configuration (do not use its output). It writes `models/heart_pipeline.joblib`, `models/model_metadata.json` and `reports/metrics.json`.

## 7. Evaluation methodology

1. Validate schema, then **remove exact duplicate rows** (keep first) *before* splitting - 302 unique records.
2. Seeded (42), **stratified 80/20 split**: 241 train / 61 test, verified to share no identical record. No grouped split: there are no genuine patient IDs and none were invented.
3. **Nested CV on the training set only** (outer: 5-fold × 3 repeats; inner: 5-fold grid search on ROC-AUC). Every transformation is fitted inside the pipeline on training folds only.
4. **Selection rule (fixed in advance):** highest mean nested ROC-AUC; among models within one standard error, take the simplest.
5. Calibration check (raw vs sigmoid, out-of-fold Brier) and F1-optimal threshold (kept at 0.5 unless the gain exceeds 0.01) - both from out-of-fold *training* predictions.
6. **One** evaluation of the final model on the test set. Bootstrap 95 % intervals reported because n = 61.

## 8. Results (all from `reports/metrics.json`)

### Model comparison - nested CV on the training set (mean ± sd over 15 outer folds)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Fit time / fold |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.848 ± 0.065 | 0.832 ± 0.071 | 0.911 ± 0.070 | 0.867 ± 0.056 | 0.910 ± 0.045 | 0.5 s |
| KNN | 0.815 ± 0.065 | 0.797 ± 0.061 | 0.888 ± 0.084 | 0.838 ± 0.060 | 0.893 ± 0.042 | 0.8 s |
| Random Forest | 0.848 ± 0.061 | 0.841 ± 0.072 | 0.898 ± 0.072 | 0.866 ± 0.052 | 0.913 ± 0.044 | 10.3 s |
| Gradient Boosting | 0.820 ± 0.062 | 0.814 ± 0.065 | 0.873 ± 0.078 | 0.840 ± 0.057 | 0.899 ± 0.045 | 1.8 s |

The models are statistically indistinguishable. Random Forest had the highest mean AUC, but Logistic Regression was within one standard error and is simpler, so **Logistic Regression** (C = 0.1) was selected.

### Final model on the untouched test set (n = 61, threshold 0.46)

| Accuracy | Precision (1) | Recall (1) | F1 (1) | ROC-AUC | Avg. precision |
|---|---|---|---|---|---|
| **0.852** (95 % CI 0.75-0.93) | 0.853 | 0.879 | 0.866 | **0.911** (95 % CI 0.83-0.98) | 0.931 |

Confusion matrix: TN 23, FP 5, FN 4, TP 29. Class 0: precision/recall/F1 = 0.852 / 0.821 / 0.836; class 1: 0.853 / 0.879 / 0.866; macro F1 0.851. Training-set accuracy 0.863 (not an honest estimate). Probabilities were **not** calibrated (out-of-fold Brier 0.1250 raw vs 0.1244 sigmoid - no meaningful gain), so the app shows a "model score", never a medical probability.

*False positives* (5) mean unnecessary follow-up and worry; *false negatives* (4) mean missed class-1 records, the costlier error in a medical setting. With 61 test records, one prediction changes accuracy by 1.6 points - treat the interval, not the point value, as the result.

### Original notebook vs corrected

| Quantity | Original notebook | Corrected |
|---|---|---|
| Random Forest (default) | 98.5 % | 81.1 ± 5.0 % (20 stratified splits of de-duplicated data) |
| KNN k=5 (default) | 73.2 % | 65.7 ± 6.2 % |
| Logistic Regression (default, unscaled) | 78.5 % | 84.2 ± 3.4 % |
| "Best KNN" | 98.5 % (k chosen on the test set) | invalid; k=1 is memorisation (59.3 % de-duplicated) |
| Final selected model | n/a | LR, 85.2 % test / 0.911 AUC, with scaling and tuned `C` |

The corrected pipeline's headline accuracy is **far below** 98.5 %: that drop is the leakage disappearing, not a regression. The original 5-fold CV figure (84.8 %) was also computed over duplicated rows.

## 9. Data-leakage investigation

- Replicating the notebook's split (`np.random.seed(42)`, 80/20) reproduces its scores exactly (LR 78.54 %, KNN 73.17 %, RF 98.54 %).
- **199 of 205 test rows (97.1 %)** had an identical copy in the training set. Over 20 random raw splits the share averages 97.8 %; after de-duplication it is 0 %.
- On raw data Random Forest averages **99.8 %** and 1-NN **99.6 %** - the signature of memorisation. After de-duplication, RF falls to 81 % and 1-NN to 59 %.
- Logistic Regression barely changes (it cannot memorise), which is why the model ranking flips once duplicates are removed.
- Of the 6 notebook test rows *without* a training copy, RF got 50 % right - far too few to interpret, but consistent with the above.
- **Corrective actions:** de-duplicate before splitting, stratify, nested CV, no test-set selection, scaling inside pipelines, one-hot for nominal codes, `max_iter` raised so the original convergence warnings no longer occur.
- **Remaining caveat:** without patient IDs, near-duplicates from the same person (if any) would still leak.

## 10. Predictions

```powershell
python -m src.predict --example
python -m src.predict --input patient.json     # JSON object with the 13 features
```
```python
from src.predict import HeartDiseasePredictor
p = HeartDiseasePredictor.load()
p.predict({"age": 55, "sex": 1, "cp": 0, "trestbps": 130, "chol": 240, "fbs": 0, "restecg": 1,
           "thalach": 150, "exang": 0, "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2})
```
Invalid input raises `InputValidationError` listing every problem (missing/unknown features, non-numeric, NaN/inf, out-of-range, bad category codes). Values outside the training range produce a warning. **Only load model files you trained yourself** - `joblib` can execute code; `load()` verifies the SHA-256 recorded in the metadata first.

## 11. Streamlit app

```powershell
streamlit run app.py
```
Pages: Home, Prediction, Model Performance, Data Insights, About & Limitations. Inputs are never stored or logged.

## 12. Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub (the 4 KB model artifact and `reports/metrics.json` are committed so the app works on a fresh checkout).
2. On <https://share.streamlit.io> choose **New app**, select the repo and branch `main`, set **Main file path** to `app.py`, then deploy.
3. Dependencies come from `requirements.txt`. `scikit-learn` is pinned to the exact training version; if you change it, retrain.
4. No secrets are needed. **Updating the model:** run `python -m src.train`, run the tests, commit `models/` and `reports/`, push - the app redeploys automatically.

This repository does not claim a live deployment; none was performed.

## 13. Tests

```powershell
python -m pytest -q
```
52 tests cover schema validation, duplicate handling, splitting, preprocessing (no train/test leakage in scaling), model selection, metrics, leakage audit, serialisation, checksum tamper detection, prediction format, invalid/out-of-range input, a fresh-process load, and Streamlit `AppTest` rendering of every page plus a prediction. App tests use the committed artifacts and skip if they are absent.

## 14. Limitations, bias and disclaimer

See §4 and §9. In short: tiny single-source data, probable label-direction problem, unverified provenance and licence, no external or prospective validation, no fairness analysis, wide confidence intervals, uncalibrated scores. Nothing here is clinically validated. **For education only.**

## 15. Future work

- Obtain the original UCI data and verify label direction, code meanings and licence
- Larger, multi-site data with patient IDs enabling grouped splits and external validation
- Subgroup (sex/age) performance analysis; formal calibration assessment on more data
- Scheduled retraining and artifact versioning in CI
