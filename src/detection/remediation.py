"""
Remediation Suggestion Engine
Takes detected data quality issues and generates human-readable,
reviewable fix suggestions. Does NOT modify the data automatically —
mirrors real data governance workflow: detect -> suggest -> human approves -> apply.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


def load_data(path):
    return pd.read_csv(path, parse_dates=["transaction_date"])


def suggest_missing_value_fixes(df, column):
    """Suggest fill values based on category-level median (a defensible, explainable rule)."""
    suggestions = []
    missing_rows = df[df[column].isna()]
    medians_by_category = df.groupby("category")[column].median()

    for idx, row in missing_rows.iterrows():
        suggested_value = medians_by_category.get(row["category"], df[column].median())
        suggestions.append({
            "transaction_id": row["transaction_id"],
            "issue": f"Missing '{column}'",
            "suggested_fix": f"Fill with category median (${suggested_value:.2f}) for '{row['category']}'",
            "suggested_value": round(float(suggested_value), 2),
            "confidence": "Medium",
            "requires_review": True,
        })
    return suggestions


def suggest_duplicate_fixes(df, subset_cols):
    suggestions = []
    dup_mask = df.duplicated(subset=subset_cols, keep=False)
    dup_groups = df[dup_mask].groupby(subset_cols)

    for _, group in dup_groups:
        keep_id = group.iloc[0]["transaction_id"]
        drop_ids = group.iloc[1:]["transaction_id"].tolist()
        for drop_id in drop_ids:
            suggestions.append({
                "transaction_id": drop_id,
                "issue": "Duplicate row",
                "suggested_fix": f"Remove — exact duplicate of transaction #{keep_id}",
                "suggested_value": None,
                "confidence": "High",
                "requires_review": True,
            })
    return suggestions


def suggest_outlier_fixes(df, feature_col="amount", contamination=0.03):
    clean = df.dropna(subset=[feature_col]).copy()
    model = IsolationForest(contamination=contamination, random_state=42)
    clean["anomaly_flag"] = model.fit_predict(clean[[feature_col]])
    anomalies = clean[clean["anomaly_flag"] == -1]

    median_by_cat = clean.groupby("category")[feature_col].median()
    suggestions = []
    for _, row in anomalies.iterrows():
        typical_value = median_by_cat.get(row["category"], clean[feature_col].median())
        deviation = row[feature_col] / typical_value if typical_value else None
        suggestions.append({
            "transaction_id": row["transaction_id"],
            "issue": f"Outlier amount (${row[feature_col]:.2f})",
            "suggested_fix": (
                f"Flag for manual review — {deviation:.1f}x typical amount for '{row['category']}'"
                if deviation else "Flag for manual review"
            ),
            "suggested_value": None,
            "confidence": "Low",
            "requires_review": True,
        })
    return suggestions


def build_remediation_report(path):
    df = load_data(path)

    missing_amount_fixes = suggest_missing_value_fixes(df, "amount")
    dup_fixes = suggest_duplicate_fixes(df, subset_cols=["customer_id", "transaction_date", "amount"])
    outlier_fixes = suggest_outlier_fixes(df)

    all_suggestions = missing_amount_fixes + dup_fixes + outlier_fixes

    summary = {
        "total_suggestions": len(all_suggestions),
        "by_confidence": pd.Series([s["confidence"] for s in all_suggestions]).value_counts().to_dict(),
        "auto_applyable_if_approved": sum(1 for s in all_suggestions if s["confidence"] == "High"),
    }
    return all_suggestions, summary


if __name__ == "__main__":
    suggestions, summary = build_remediation_report(
        "/home/claude/data-quality-engine/data/transactions.csv"
    )

    print("=" * 60)
    print("REMEDIATION SUGGESTIONS")
    print("=" * 60)
    print(f"Total suggestions generated: {summary['total_suggestions']}")
    print(f"By confidence level: {summary['by_confidence']}")
    print(f"High-confidence (safe to batch-approve): {summary['auto_applyable_if_approved']}")
    print("-" * 60)
    print("Sample suggestions:\n")
    for s in suggestions[:8]:
        print(f"Txn #{s['transaction_id']} | {s['issue']}")
        print(f"   -> {s['suggested_fix']} [{s['confidence']} confidence]")
    print("=" * 60)
