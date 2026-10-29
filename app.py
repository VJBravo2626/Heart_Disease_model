"""Streamlit app for the Heart Disease Prediction educational project.

Run from the repository root:  streamlit run app.py
No user input is stored or logged by this app.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import config, data
from src.predict import HeartDiseasePredictor, InputValidationError, ModelLoadError, EXAMPLE_INPUT

st.set_page_config(page_title="Heart Disease Prediction", page_icon="🫀", layout="wide")

DISCLAIMER = (
    "**Educational project - not a medical device.** This tool cannot diagnose, rule out, or independently "
    "assess heart disease and must not replace evaluation by a qualified healthcare professional. "
    "If you have symptoms or concerns, contact a clinician or local emergency services."
)
PAGES = ["Home", "Prediction", "Model Performance", "Data Insights", "About & Limitations"]


# ----------------------------------------------------------------------------- cached loaders
@st.cache_resource(show_spinner="Loading model...")
def get_predictor() -> HeartDiseasePredictor:
    return HeartDiseasePredictor.load()


@st.cache_data
def get_metrics() -> dict:
    return json.loads(config.METRICS_PATH.read_text())


@st.cache_data
def get_unique_data() -> pd.DataFrame:
    return data.drop_exact_duplicates(data.load_raw())


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def pm(d: dict) -> str:
    return f"{d['mean']:.3f} ± {d['std']:.3f}"


# ----------------------------------------------------------------------------- pages
def page_home() -> None:
    st.title("🫀 Heart Disease Prediction")
    st.warning(DISCLAIMER)
    st.subheader("Objective")
    st.write(
        "Given 13 routine clinical measurements, a classical machine-learning model predicts the label "
        "recorded in a public heart-disease dataset. The project's main purpose is to demonstrate an "
        "**honest evaluation workflow**: the original notebook's very high accuracy was inflated by duplicated "
        "rows, so this version removes the duplicates, evaluates with nested cross-validation, and touches the "
        "held-out test set only once.")
    try:
        m = get_metrics()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows in CSV", m["dataset"]["raw_rows"])
        c2.metric("Unique records", m["dataset"]["unique_rows"])
        c3.metric("Selected model", m["selected_model"])
        c4.metric("Held-out test accuracy", pct(m["test_metrics_final_model"]["accuracy"]))
    except FileNotFoundError:
        st.error("Metrics file not found. Run `python -m src.train` first.")
    left, right = st.columns(2)
    with left:
        st.subheader("Workflow")
        st.markdown(
            "1. Load & validate `heart.csv` (schema, ranges, codes)\n"
            "2. Quantify duplicates and leakage in the original setup\n"
            "3. Remove exact duplicate rows, then stratified train/test split\n"
            "4. Compare Logistic Regression, KNN, Random Forest and Gradient Boosting with **nested CV**\n"
            "5. Select the model, check calibration, choose the threshold - all on training data only\n"
            "6. Evaluate **once** on the untouched test set\n"
            "7. Save the full pipeline with `joblib` and serve it here")
    with right:
        st.subheader("Technology & architecture")
        st.markdown(
            "- **Python, Pandas, NumPy, scikit-learn** (`Pipeline` + `ColumnTransformer`)\n"
            "- **joblib** artifact + JSON metadata (schema, threshold, checksum)\n"
            "- **Streamlit** front-end; `src/` modules hold all reusable logic\n"
            "- **pytest** test-suite for data, pipeline, prediction and app")
    st.subheader("Features")
    rows = [{"Feature": f, "Description": config.FEATURE_SPEC[f]["label"] +
             (f" ({config.FEATURE_SPEC[f]['unit']})" if "unit" in config.FEATURE_SPEC[f] else "")}
            for f in config.FEATURES]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def page_prediction() -> None:
    st.title("Prediction")
    st.warning(DISCLAIMER)
    try:
        predictor = get_predictor()
    except ModelLoadError as exc:
        st.error(f"The model could not be loaded: {exc}")
        return
    st.caption("Nothing you enter is stored or logged. No names or contact details are requested.")
    with st.form("prediction_form"):
        cols = st.columns(3)
        values = {}
        for i, f in enumerate(config.FEATURES):
            spec, col, default = config.FEATURE_SPEC[f], cols[i % 3], EXAMPLE_INPUT[f]
            label = spec["label"] + (f" ({spec['unit']})" if "unit" in spec else "")
            with col:
                if spec["kind"] == "numeric":
                    is_int = spec.get("integer")
                    values[f] = st.number_input(
                        label, min_value=spec["min"], max_value=spec["max"],
                        value=int(default) if is_int else float(default),
                        step=1 if is_int else 0.1, help=spec["help"], key=f"in_{f}")
                else:
                    choices = config.feature_choices(f)
                    values[f] = st.selectbox(label, choices, index=choices.index(default),
                                             format_func=lambda c, s=spec: s["choices"][c],
                                             help=spec["help"], key=f"in_{f}")
        submitted = st.form_submit_button("Predict", type="primary")
    if not submitted:
        return
    try:
        with st.spinner("Computing..."):
            result = predictor.predict(values)
    except InputValidationError as exc:
        for e in exc.errors:
            st.error(e)
        return
    except Exception as exc:  # unexpected failure: show it, do not crash the app
        st.error(f"Prediction failed: {exc}")
        return
    st.subheader("Result")
    st.info(f"**Model output: {result['label']}**")
    for w in result["warnings"]:
        st.warning(w)
    st.metric("Model score for class 1", f"{result['class1_score']:.2f}",
              help=f"Decision threshold: {result['threshold']:.2f}. Scores at or above it map to class 1.")
    st.caption(result["score_note"])
    with st.expander("What this result means - and does not mean", expanded=True):
        st.markdown(
            "- It is the model's output for **one set of numbers**, learned from ~300 patient records from a "
            "single, decades-old public dataset.\n"
            "- It is **not** a diagnosis, a risk percentage, or advice. A different output does not rule "
            "disease in or out.\n"
            "- The dataset's label direction and some code meanings are disputed (see *About & Limitations*).\n"
            "- Please discuss any health concern with a qualified clinician.")


def page_performance() -> None:
    st.title("Model Performance")
    try:
        m = get_metrics()
    except FileNotFoundError:
        st.error("Metrics file not found. Run `python -m src.train` first.")
        return
    st.caption(f"Model v{m['model_version']} · trained {m['trained_at_utc']} · all numbers below were "
               "generated by `python -m src.train`.")
    ds, aud = m["dataset"], m["leakage_audit"]

    st.header("1. Corrected vs. original results")
    rep, ms = aud["notebook_split_replication"], aud["multi_seed"]
    st.markdown(
        f"The CSV has **{ds['raw_rows']} rows but only {ds['unique_rows']} unique records**; each record "
        f"appears {ds['duplicates']['mean_copies_of_duplicated_records']:.1f} times on average (max "
        f"{ds['duplicates']['max_copies_of_one_record']}). In the notebook's 80/20 split, "
        f"**{rep['test_rows_with_identical_copy_in_train']} of {rep['test_rows']} test rows "
        f"({pct(rep['fraction_test_rows_with_copy_in_train'])}) had an identical copy in the training set.** "
        "A Random Forest (or 1-NN) can simply memorise them.")
    rows = []
    for name in ["Logistic Regression", "KNN (k=5)", "Random Forest", "KNN (k=1)"]:
        rows.append({
            "Model (default settings)": name,
            "Notebook split (replicated)": pct(rep["default_model_accuracy"][name]),
            f"Raw data, {ms['n_seeds']} random splits": pm(ms["raw_data_random_split"][name]),
            f"De-duplicated, {ms['n_seeds']} stratified splits": pm(ms["deduplicated_stratified_split"][name])})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Random Forest drops from ~98-100% to the mid-80s once duplicates cannot leak between train and "
               "test. The drop is the leakage being removed, not a worse model.")

    st.header("2. Model comparison (nested cross-validation, training set only)")
    cv = m["nested_cv"]
    table = pd.DataFrame({
        name: {"Accuracy": pm(v["accuracy"]), "Precision": pm(v["precision"]), "Recall": pm(v["recall"]),
               "F1": pm(v["f1"]), "ROC-AUC": pm(v["roc_auc"]), "Avg precision": pm(v["average_precision"]),
               "Fit time / fold (s)": f"{v['mean_fit_time_s']:.2f}"} for name, v in cv.items()}).T
    st.dataframe(table, use_container_width=True)
    st.caption(f"Mean ± standard deviation over {next(iter(cv.values()))['n_outer_folds']} outer folds "
               "(hyper-parameters tuned in an inner loop). Small data -> large variability.")
    fig, ax = plt.subplots(figsize=(6, 2.8))
    names = list(cv)
    ax.barh(names, [cv[n]["roc_auc"]["mean"] for n in names], xerr=[cv[n]["roc_auc"]["std"] for n in names],
            color="#7F8FA6", capsize=3)
    ax.set_xlim(0.5, 1.0); ax.set_xlabel("Nested-CV ROC-AUC (mean ± sd)"); ax.invert_yaxis()
    st.pyplot(fig); plt.close(fig)
    st.success(f"**Selected: {m['selected_model']}.** {m['selection_rationale']}")

    st.header("3. Final model on the untouched test set")
    t = m["test_metrics_final_model"]
    ci = t["bootstrap"]
    c = st.columns(5)
    c[0].metric("Accuracy", pct(t["accuracy"])); c[1].metric("Precision (cls 1)", pct(t["precision_class1"]))
    c[2].metric("Recall (cls 1)", pct(t["recall_class1"])); c[3].metric("F1 (cls 1)", pct(t["f1_class1"]))
    c[4].metric("ROC-AUC", f"{t['roc_auc']:.3f}")
    st.caption(f"n = {t['n']} test records. 95% bootstrap intervals: accuracy "
               f"{pct(ci['accuracy_95ci'][0])}-{pct(ci['accuracy_95ci'][1])}, ROC-AUC "
               f"{ci['roc_auc_95ci'][0]:.2f}-{ci['roc_auc_95ci'][1]:.2f}. With so few records the test estimate "
               "is imprecise. Decision threshold "
               f"{t['threshold']:.2f} was chosen on training data only.")
    left, right = st.columns(2)
    cm = t["confusion_matrix"]
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    with left:
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
        ax.imshow(mat, cmap="Blues")
        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, int(v), ha="center", va="center", fontsize=14,
                    color="white" if v > mat.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Pred 0", "Pred 1"]); ax.set_yticks([0, 1], ["True 0", "True 1"])
        ax.set_title("Confusion matrix (test)")
        st.pyplot(fig); plt.close(fig)
    with right:
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
        ax.plot(t["roc_curve"]["fpr"], t["roc_curve"]["tpr"], color="#C0392B", label=f"AUC {t['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="grey"); ax.set_xlabel("False-positive rate")
        ax.set_ylabel("True-positive rate"); ax.set_title("ROC curve (test)"); ax.legend(loc="lower right")
        st.pyplot(fig); plt.close(fig)
    rpt = t["classification_report"]
    cls = pd.DataFrame({k: {x: f"{rpt[k][x]:.3f}" for x in ("precision", "recall", "f1-score")} | {"support": int(rpt[k]["support"])}
                        for k in ("0", "1", "macro avg", "weighted avg")}).T
    st.dataframe(cls, use_container_width=True)
    st.markdown(
        f"**False positives ({cm['fp']})**: records predicted as class 1 that are labelled 0 - in a screening "
        "context this means unnecessary follow-up testing and worry. **False negatives "
        f"({cm['fn']})**: class-1 records the model misses - the costlier error in a medical setting, "
        "because a real problem could be overlooked. Lowering the threshold trades FPs for fewer FNs.")

    st.header("4. Training vs. cross-validation vs. test")
    tr = m["train_metrics_final_model"]
    sel = cv[m["selected_model"]]
    st.dataframe(pd.DataFrame({
        "Accuracy": [pct(tr["accuracy"]), pm(sel["accuracy"]), pct(t["accuracy"])],
        "ROC-AUC": [f"{tr['roc_auc']:.3f}", pm(sel["roc_auc"]), f"{t['roc_auc']:.3f}"]},
        index=["Training set (optimistic)", "Nested CV (training set)", "Held-out test set"]),
        use_container_width=True)

    st.header("5. Calibration & threshold")
    cal, th = m["calibration"], m["threshold_selection"]
    st.markdown(
        f"Out-of-fold Brier score (lower is better): raw **{cal['oof_brier_raw']:.3f}**, sigmoid-calibrated "
        f"**{cal['oof_brier_sigmoid']:.3f}** -> calibration "
        f"{'**was applied**' if cal['use_calibration'] else '**was not applied** (no meaningful gain)'}. "
        f"Threshold: {th['threshold']:.2f} (out-of-fold F1 at 0.50 = {th['oof_f1_at_default']:.3f}; best on "
        f"grid = {th['oof_f1_at_best']:.3f} at {th['best_grid_threshold']:.2f}; default kept unless gain > 0.01).")

    st.header("6. The original notebook's reported numbers")
    st.dataframe(pd.DataFrame({"Reported accuracy": {k: pct(v) for k, v in m["original_notebook_reported"].items()}}),
                 use_container_width=True)
    st.caption("Verified against the notebook's saved outputs and replicated above. They are inflated by duplicate "
               "rows; 'best k on the test set' additionally selects a hyper-parameter using test data.")


def page_insights() -> None:
    st.title("Data Insights")
    st.caption("Computed on the de-duplicated records. Associations are **not** causal medical explanations.")
    df = get_unique_data()
    c1, c2 = st.columns(2)
    with c1:
        counts = df[config.TARGET].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(["Class 0", "Class 1"], counts.values, color=["#7F8FA6", "#C0392B"])
        for i, v in enumerate(counts.values):
            ax.text(i, v + 1, str(v), ha="center")
        ax.set_title("Class distribution (unique records)")
        st.pyplot(fig); plt.close(fig)
    with c2:
        feat = st.selectbox("Feature distribution by class", config.FEATURES,
                            format_func=lambda f: config.FEATURE_SPEC[f]["label"])
        fig, ax = plt.subplots(figsize=(4.4, 3))
        spec = config.FEATURE_SPEC[feat]
        if spec["kind"] == "numeric" and feat != "ca":
            for k, col in ((0, "#7F8FA6"), (1, "#C0392B")):
                ax.hist(df.loc[df[config.TARGET] == k, feat], bins=15, alpha=0.6, color=col, label=f"Class {k}")
        else:
            ct = pd.crosstab(df[feat], df[config.TARGET])
            ct.plot(kind="bar", ax=ax, color=["#7F8FA6", "#C0392B"], rot=0)
        ax.legend(); ax.set_xlabel(feat)
        st.pyplot(fig); plt.close(fig)
    st.subheader("Correlation matrix")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90); ax.set_yticks(range(len(corr)), corr.columns)
    fig.colorbar(im, ax=ax, shrink=0.8)
    st.pyplot(fig); plt.close(fig)
    st.caption("Pearson correlation treats nominal codes (cp, restecg, slope, thal) as numbers - read those cells "
               "as rough indications only.")
    st.subheader("Permutation importance (selected model, test set)")
    try:
        imp = pd.DataFrame(get_metrics()["permutation_importance_test"]).set_index("feature")
        fig, ax = plt.subplots(figsize=(6, 3.8))
        imp = imp.iloc[::-1]
        ax.barh(imp.index, imp["mean"], xerr=imp["std"], color="#7F8FA6", capsize=2)
        ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled")
        st.pyplot(fig); plt.close(fig)
        st.caption("Computed on only a few dozen test records, so the ranking is noisy. It describes what this "
                   "model uses, not what causes heart disease.")
    except FileNotFoundError:
        st.info("Run `python -m src.train` to generate importance values.")


def page_about() -> None:
    st.title("About & Limitations")
    st.warning(DISCLAIMER)
    try:
        meta = get_predictor().metadata
        st.markdown(f"**Model version:** {meta['model_version']} · **Last trained (UTC):** {meta['trained_at_utc']} · "
                    f"**Estimator:** {meta['selected_model']} · **Decision threshold:** {meta['threshold']:.2f}")
    except ModelLoadError as exc:
        st.error(str(exc))
    st.subheader("Dataset provenance")
    st.markdown(
        "`heart.csv` (1,025 rows × 14 columns) matches the widely shared Kaggle \"Heart Disease\" file, which is "
        "understood to derive from the UCI Heart Disease collection (Cleveland subset; ~303 patients). The exact "
        "origin and licence **could not be verified from the repository** and no patient identifiers are present. "
        "Check the original source's terms before any reuse beyond education.")
    st.subheader("Evaluation methodology")
    st.markdown(
        "Exact duplicates removed before splitting (302 unique records) · stratified 80/20 split, seed 42 · "
        "nested cross-validation for model comparison · calibration and threshold chosen from out-of-fold "
        "training predictions · one final test evaluation. No patient IDs exist, so independence between "
        "records cannot be proven; near-duplicates of the same person, if any, would still leak.")
    st.subheader("Known limitations")
    st.markdown(
        "- **Tiny data** (≈300 records, ≈60 test records): metrics have wide uncertainty.\n"
        "- **Old, single-source, mostly male data**: no evidence of generalisation to other populations; bias "
        "against under-represented groups is likely and unmeasured.\n"
        "- **Label direction anomaly**: in this file, class 1 is associated with *lower* exercise-induced angina, "
        "ST depression and vessel counts and *higher* max heart rate - the opposite of what is clinically "
        "expected for disease. The label may be inverted relative to its documentation. This could not be "
        "verified offline, so outputs are reported as \"Class 0/1\" and must not be read as clinical statements.\n"
        "- **Code meanings** for cp, restecg, slope and thal differ between published versions; `ca = 4` and "
        "`thal = 0` are undocumented values present in the data.\n"
        "- Scores are not medically validated probabilities.\n"
        "- No external validation, no prospective testing, no regulatory review.")


# ----------------------------------------------------------------------------- navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", PAGES, label_visibility="collapsed")
st.sidebar.info("Educational ML project. Not a medical device.")
{"Home": page_home, "Prediction": page_prediction, "Model Performance": page_performance,
 "Data Insights": page_insights, "About & Limitations": page_about}[page]()
