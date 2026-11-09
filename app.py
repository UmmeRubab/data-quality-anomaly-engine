"""DataGuard AI — interactive data quality and model robustness dashboard."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from phase2_backend.dataset_engine import (
    MAX_BYTES,
    bank_marketing,
    normalize_unknown,
    parse_csv,
    profile,
    run_experiment,
)
from src.detection.anomaly_detector import (
    check_duplicates,
    check_missing_values,
    compute_health_score,
    detect_outliers_isolation_forest,
)

ROOT = Path(__file__).resolve().parent
SAMPLE_PATH = ROOT / "data" / "transactions.csv"

st.set_page_config(
    page_title="DataGuard AI | Data Quality & ML Robustness",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {max-width: 1350px; padding-top: 1.8rem;}
h1, h2, h3 {letter-spacing: -.025em;}
div[data-testid="stMetric"] {background: rgba(120,130,155,.07); padding: 1rem; border-radius: .75rem; border: 1px solid rgba(140,145,160,.16);}
[data-testid="stSidebar"] {border-right: 1px solid rgba(140,145,160,.14);}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_PATH)


@st.cache_data(show_spinner="Loading the UCI Bank Marketing dataset...")
def load_bank() -> pd.DataFrame:
    return bank_marketing()


@st.cache_data(show_spinner=False)
def profile_cached(df: pd.DataFrame, name: str) -> dict:
    return profile(df, name)


@st.cache_data(show_spinner="Comparing clean and corrupted training data...")
def experiment_cached(
    df: pd.DataFrame, target: str, model: str, corruption: str, rate: float, seed: int, drop_duration: bool
) -> dict:
    return run_experiment(df, target, model, corruption, rate, seed, drop_duration)


def download_json(label: str, data: dict, filename: str) -> None:
    st.download_button(label, json.dumps(data, indent=2, default=str), filename, "application/json")


def quality_overview(df: pd.DataFrame, feature: str, contamination: float) -> tuple[dict, pd.DataFrame]:
    """Use the repo's actual detection helpers without modifying user-uploaded rows."""
    work = normalize_unknown(df)
    missing = check_missing_values(work)
    # For arbitrary uploads, full-row duplicate detection is safer than assuming a particular schema.
    duplicates, duplicate_mask = check_duplicates(work, subset_cols=list(work.columns))
    numeric = pd.to_numeric(work[feature], errors="coerce")
    flags = pd.DataFrame(columns=work.columns)
    if numeric.notna().sum() >= 10 and numeric.nunique(dropna=True) > 1:
        temporary = work.copy()
        temporary[feature] = numeric
        flags, _ = detect_outliers_isolation_forest(
            temporary, feature_col=feature, contamination=contamination
        )
    score = compute_health_score(work, missing, duplicates, len(flags))
    return {
        "rows": len(work), "columns": len(work.columns),
        "missing_cells": int(work.isna().sum().sum()),
        "missing_by_column_pct": missing,
        "duplicate_rows": duplicates,
        "anomaly_rows": len(flags),
        "health_score": score,
        "duplicate_mask": duplicate_mask,
    }, flags


def suggest_review_actions(df: pd.DataFrame, numeric_col: str, anomalies: pd.DataFrame) -> pd.DataFrame:
    """Make conservative, non-destructive suggestions for generic CSVs."""
    work = normalize_unknown(df)
    actions = []
    for col in work.columns:
        n = int(work[col].isna().sum())
        if n:
            actions.append({
                "ID": f"MISSING-{col}", "Issue": "Missing values",
                "Column": str(col), "Affected rows": n,
                "Suggestion": "Review missingness; consider an appropriate imputation strategy after checking the column's meaning.",
                "Priority": "Medium",
            })
    n_dups = int(work.duplicated().sum())
    if n_dups:
        actions.append({
            "ID": "DUPLICATES", "Issue": "Duplicate rows", "Column": "All columns",
            "Affected rows": n_dups,
            "Suggestion": "Check whether identical rows are genuine repeat transactions before removing duplicates.",
            "Priority": "Medium",
        })
    if not anomalies.empty:
        actions.append({
            "ID": f"ANOMALIES-{numeric_col}", "Issue": "Statistical anomalies",
            "Column": numeric_col, "Affected rows": len(anomalies),
            "Suggestion": "Inspect extreme values and validate against business rules; do not automatically delete outliers.",
            "Priority": "Low",
        })
    return pd.DataFrame(actions, columns=["ID", "Issue", "Column", "Affected rows", "Suggestion", "Priority"])


st.title("🔍 DataGuard")
st.markdown(
    """
    <p style="color: #FFFFFF !important; font-size: 14px;">
        Data Quality &amp; Anomaly Detection Engine · Evidence-aware checks · Human-in-the-loop remediation · ML robustness
    </p>
    """,
    unsafe_allow_html=True
)
st.info("Exploratory application, not a production validator. Use synthetic or non-sensitive datasets only. Uploaded files are processed in this app session; do not upload confidential data.")

with st.sidebar:
    st.header("Dataset")
    choice = st.radio("Choose data source", ["Sample transactions", "Upload CSV", "UCI Bank Marketing"], index=0)
    uploaded = st.file_uploader("Choose a UTF-8 CSV (maximum 8 MB)", type=["csv"]) if choice == "Upload CSV" else None
    st.divider()
    st.caption("No Vercel, Render, Next.js, or external FastAPI service is required for this Streamlit dashboard.")

try:
    if choice == "Sample transactions":
        df = load_sample()
        dataset_name = "Synthetic transactions"
    elif choice == "UCI Bank Marketing":
        df = load_bank()
        dataset_name = "UCI Bank Marketing (bank-full.csv)"
    elif uploaded is None:
        st.warning("Upload a CSV file to begin, or select a sample dataset in the sidebar.")
        st.stop()
    else:
        raw = uploaded.getvalue()
        if len(raw) > MAX_BYTES:
            st.error("The CSV exceeds the 8 MB limit.")
            st.stop()
        df = parse_csv(raw)
        dataset_name = uploaded.name
except (ValueError, OSError, pd.errors.ParserError) as exc:
    st.error(f"Unable to load dataset: {exc}")
    st.stop()

st.caption(f"Current dataset: **{dataset_name}** · {len(df):,} rows × {len(df.columns)} columns")

tab_quality, tab_review, tab_ml, tab_about = st.tabs([
    "📊 Data quality", "✅ Review suggestions", "🧪 ML robustness", "ℹ️ About"
])

with tab_quality:
    st.subheader("Data quality scorecard")
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        controls = st.columns([3, 2])
        with controls[0]:
            numeric_col = st.selectbox("Numeric column to screen for anomalies", numeric_cols, index=numeric_cols.index("amount") if "amount" in numeric_cols else 0)
        with controls[1]:
            contamination = st.slider("Expected anomaly fraction", 0.01, 0.15, 0.03, 0.01)
    else:
        numeric_col = None
        contamination = 0.03
        st.warning("This dataset has no numeric columns. Isolation Forest detection is unavailable.")

    if numeric_col:
        try:
            scorecard, flagged = quality_overview(df, numeric_col, contamination)
        except (ValueError, TypeError) as exc:
            st.warning(f"Anomaly detection could not run: {exc}")
            scorecard, flagged = None, pd.DataFrame()
    else:
        clean = normalize_unknown(df)
        missing = check_missing_values(clean)
        dups = int(clean.duplicated().sum())
        scorecard = {
            "rows": len(df), "columns": len(df.columns),
            "missing_cells": int(clean.isna().sum().sum()),
            "missing_by_column_pct": missing, "duplicate_rows": dups,
            "anomaly_rows": 0,
            "health_score": compute_health_score(clean, missing, dups, 0),
        }
        flagged = pd.DataFrame()

    if scorecard:
        a, b, c, d, e = st.columns(5)
        a.metric("Data health score", f"{scorecard['health_score']:.1f}/100")
        b.metric("Rows", f"{scorecard['rows']:,}")
        c.metric("Missing cells", f"{scorecard['missing_cells']:,}")
        d.metric("Duplicate rows", f"{scorecard['duplicate_rows']:,}")
        e.metric("Anomalies flagged", f"{scorecard['anomaly_rows']:,}")
        st.caption("Health score is a heuristic based on missingness, duplicates and anomaly counts—not a validated quality metric. Full-row duplicates are used for generic uploads.")

        prof = profile_cached(df, dataset_name)
        st.markdown("#### Column profiling")
        cols = pd.DataFrame(prof["column_profiles"])
        display_cols = [c for c in ["name", "type", "missing", "missing_pct", "unique", "mean", "median", "outliers_iqr"] if c in cols.columns]
        st.dataframe(cols[display_cols], use_container_width=True, hide_index=True)
        left, right = st.columns(2)
        with left:
            st.download_button("Download column profile (CSV)", cols.to_csv(index=False).encode("utf-8"), "dataguard-column-profile.csv", "text/csv")
        with right:
            download_json("Download quality report (JSON)", {k:v for k,v in scorecard.items() if k != "duplicate_mask"}, "dataguard-quality-report.json")

        st.markdown("#### Dataset preview")
        st.dataframe(df.head(30), use_container_width=True, hide_index=True)
        st.markdown("#### Missing values by column")
        missing_counts = normalize_unknown(df).isna().sum().sort_values(ascending=False)
        if missing_counts.max() > 0:
            st.bar_chart(missing_counts[missing_counts > 0])
        else:
            st.success("No missing values detected after normalizing known missing-value tokens.")
        st.markdown("#### Flagged anomalous rows")
        if not flagged.empty:
            st.dataframe(flagged.head(100), use_container_width=True)
            st.download_button("Download flagged anomalies (CSV)", flagged.to_csv(index=False).encode("utf-8"), "dataguard-anomalies.csv", "text/csv")
        else:
            st.caption("No anomalies were flagged for the selected numeric feature, or anomaly detection was unavailable.")

with tab_review:
    st.subheader("Human-reviewed remediation suggestions")
    st.caption("Actions are suggestions only. No uploaded or stored data is modified by review decisions.")
    if scorecard:
        suggestions = suggest_review_actions(df, numeric_col or "(none)", flagged)
        if suggestions.empty:
            st.success("No missing-value, duplicate-row, or numerical-anomaly review suggestions were generated.")
        else:
            st.dataframe(suggestions, hide_index=True, use_container_width=True)
            choice_id = st.selectbox("Select a suggestion to review", suggestions["ID"].tolist())
            reviewer = st.text_input("Reviewer name", "Demo reviewer", max_chars=80)
            decision = st.radio("Decision", ["Approved for review", "Rejected", "Needs investigation"], horizontal=True)
            if st.button("Record review decision", type="primary"):
                if not reviewer.strip():
                    st.error("Enter a reviewer name.")
                else:
                    if "review_logs" not in st.session_state:
                        st.session_state.review_logs = {}
                    key = f"{dataset_name}:{len(df)}:{choice_id}"
                    st.session_state.review_logs[key] = {
                        "Dataset": dataset_name, "Suggestion ID": choice_id,
                        "Reviewer": reviewer.strip(), "Decision": decision,
                    }
                    st.success("Review decision recorded for this browser session. No changes applied to data.")
            logs = [v for k,v in st.session_state.get("review_logs", {}).items() if k.startswith(f"{dataset_name}:{len(df)}:")]
            if logs:
                st.markdown("#### Session review log")
                logs_df = pd.DataFrame(logs)
                st.dataframe(logs_df, hide_index=True, use_container_width=True)
                st.download_button("Download review log (CSV)", logs_df.to_csv(index=False).encode("utf-8"), "dataguard-review-log.csv", "text/csv")
                st.caption("Session log is not persistent or shared across reviewers. Export it before closing the session.")

with tab_ml:
    st.subheader("Controlled ML robustness experiment")
    st.write("Train equivalent models on clean versus artificially corrupted training data, then evaluate both on the same untouched holdout test set.")
    st.caption("Experimental results are exploratory. Data leakage and feature suitability must be reviewed for each uploaded dataset.")
    target = st.selectbox("Classification target column", list(df.columns), index=list(df.columns).index("y") if "y" in df.columns else len(df.columns)-1)
    left, middle, right = st.columns(3)
    with left:
        model = st.selectbox("Model", ["logistic_regression", "random_forest", "gradient_boosting"], format_func=lambda s:s.replace("_", " ").title())
    with middle:
        corruption = st.selectbox("Training-data corruption", ["label_noise", "missing_values", "outliers"], format_func=lambda s:s.replace("_", " ").title())
    with right:
        rate = st.slider("Corruption rate", 0.0, 0.5, 0.2, 0.05)
    left, right = st.columns(2)
    with left:
        seed = st.number_input("Random seed", min_value=0, max_value=4294967295, value=42, step=1)
    with right:
        drop_duration = st.checkbox("Exclude 'duration' for Bank Marketing y target (leakage prevention)", value=True)

    if st.button("Run clean vs. corrupted comparison", type="primary"):
        try:
            result = experiment_cached(df, target, model, corruption, float(rate), int(seed), drop_duration)
            st.session_state.ml_result = {"config": (dataset_name, target, model, corruption, rate, seed, drop_duration), "data": result}
        except (ValueError, TypeError, KeyError) as exc:
            st.error(f"Experiment could not run: {exc}")

    cfg = (dataset_name, target, model, corruption, rate, seed, drop_duration)
    saved = st.session_state.get("ml_result")
    if saved and saved["config"] == cfg:
        result = saved["data"]
        a, b, c, d = st.columns(4)
        a.metric("Clean accuracy", f"{result['clean']['accuracy']:.1%}")
        b.metric("Corrupted accuracy", f"{result['corrupted']['accuracy']:.1%}")
        c.metric("Accuracy change", f"{result['accuracy_change'] * 100:+.2f} pp")
        d.metric("Macro F1 change", f"{result['f1_change'] * 100:+.2f} pp")
        comparison = pd.DataFrame({
            "Clean": result["clean"], "Corrupted": result["corrupted"]
        })
        st.markdown("#### Performance on untouched test data")
        st.bar_chart(comparison)
        st.dataframe(comparison, use_container_width=True)
        st.caption(f"Train: {result['train_rows']:,} · Test: {result['test_rows']:,}. {result['protocol']} {result['limitations']}")
        if result["excluded_features"]:
            st.info("Excluded features: " + ", ".join(result["excluded_features"]))
        download_json("Download experiment report (JSON)", result, "dataguard-experiment.json")

with tab_about:
    st.subheader("About DataGuard AI")
    st.markdown("""
DataGuard AI demonstrates **data profiling**, **unsupervised anomaly screening**, **human-in-the-loop remediation review**, and **controlled ML sensitivity experiments** using Python.

- **Stack:** Streamlit, pandas, NumPy, scikit-learn, Isolation Forest.
- **Data:** Synthetic transactions, user-uploaded CSVs, or the public UCI Bank Marketing dataset.
- **Evaluation:** Clean/corrupted train-only comparisons using Logistic Regression, Random Forest, or Gradient Boosting. Single-split results are illustrative, not proof of generalizability.
- **Safety:** This application does not automatically fix records or store approvals in a persistent database.
- **Privacy:** Do not upload personal, confidential, financial, or proprietary data to a public demo.

**Dataset citation:** [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing).

**Source code:** [GitHub](https://github.com/UmmeRubab/data-quality-anomaly-engine).
""")
