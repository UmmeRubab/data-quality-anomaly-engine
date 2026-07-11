"""
Data Quality & Anomaly Detection Engine
Detects: missing values, duplicates, and statistical outliers (Isolation Forest).
Outputs a Data Health Score (0-100) and a detailed report.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


def load_data(path):
    return pd.read_csv(path, parse_dates=["transaction_date"])


def check_missing_values(df):
    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    return missing_pct[missing_pct > 0].to_dict()


def check_duplicates(df, subset_cols):
    dup_mask = df.duplicated(subset=subset_cols, keep=False)
    return int(dup_mask.sum()), dup_mask


def detect_outliers_isolation_forest(df, feature_col="amount", contamination=0.03):
    clean = df.dropna(subset=[feature_col]).copy()
    model = IsolationForest(contamination=contamination, random_state=42)
    clean["anomaly_flag"] = model.fit_predict(clean[[feature_col]])
    # -1 = anomaly, 1 = normal
    anomalies = clean[clean["anomaly_flag"] == -1]
    return anomalies, model


def compute_health_score(df, missing_report, dup_count, anomaly_count):
    total_rows = len(df)
    missing_penalty = sum(missing_report.values()) * 0.5
    dup_penalty = (dup_count / total_rows * 100) * 0.7
    anomaly_penalty = (anomaly_count / total_rows * 100) * 0.8

    score = 100 - missing_penalty - dup_penalty - anomaly_penalty
    return round(max(score, 0), 1)


def run_quality_check(path):
    df = load_data(path)

    missing_report = check_missing_values(df)
    dup_count, dup_mask = check_duplicates(
        df, subset_cols=["customer_id", "transaction_date", "amount"]
    )
    anomalies, model = detect_outliers_isolation_forest(df)

    health_score = compute_health_score(
        df, missing_report, dup_count, len(anomalies)
    )

    report = {
        "total_rows": len(df),
        "missing_values_pct_by_column": missing_report,
        "duplicate_rows": dup_count,
        "anomalous_rows_detected": len(anomalies),
        "sample_anomalies": anomalies[["transaction_id", "customer_id", "amount"]]
        .head(5)
        .to_dict(orient="records"),
        "data_health_score": health_score,
    }
    return report


if __name__ == "__main__":
    report = run_quality_check("/home/claude/data-quality-engine/data/transactions.csv")

    print("=" * 55)
    print("DATA QUALITY SCORECARD")
    print("=" * 55)
    print(f"Total rows scanned:        {report['total_rows']}")
    print(f"Duplicate rows found:      {report['duplicate_rows']}")
    print(f"Anomalous rows detected:   {report['anomalous_rows_detected']}")
    print("Missing values by column:")
    for col, pct in report["missing_values_pct_by_column"].items():
        print(f"   - {col}: {pct}%")
    print("\nSample flagged anomalies (extreme amounts):")
    for row in report["sample_anomalies"]:
        print(f"   - Txn #{row['transaction_id']} | Customer {row['customer_id']} | Amount: ${row['amount']}")
    print("-" * 55)
    print(f"DATA HEALTH SCORE: {report['data_health_score']} / 100")
    print("=" * 55)
