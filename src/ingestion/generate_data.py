"""
Generates a synthetic 'messy' transactions dataset that mimics real-world
data quality issues: missing values, duplicates, outliers, and drift.
"""
import pandas as pd
import numpy as np

np.random.seed(42)

N = 5000  # total rows

# --- Base clean data ---
customer_ids = np.random.randint(1000, 2000, N)
transaction_amount = np.random.normal(loc=150, scale=50, size=N).round(2)
transaction_amount = np.clip(transaction_amount, 5, None)

categories = np.random.choice(
    ["Groceries", "Electronics", "Clothing", "Fuel", "Dining"], N
)

dates = pd.date_range("2026-01-01", periods=N, freq="h")

df = pd.DataFrame({
    "transaction_id": range(1, N + 1),
    "customer_id": customer_ids,
    "transaction_date": dates,
    "amount": transaction_amount,
    "category": categories,
})

# --- Inject missing values (~4% of amount field) ---
missing_idx = np.random.choice(df.index, size=int(N * 0.04), replace=False)
df.loc[missing_idx, "amount"] = np.nan

# --- Inject missing customer_id (~2%) ---
missing_cust_idx = np.random.choice(df.index, size=int(N * 0.02), replace=False)
df.loc[missing_cust_idx, "customer_id"] = np.nan

# --- Inject extreme outliers (~1%) ---
remaining_idx = df.index.difference(missing_idx)
outlier_idx = np.random.choice(remaining_idx, size=int(N * 0.01), replace=False)
df.loc[outlier_idx, "amount"] = np.random.uniform(5000, 50000, size=len(outlier_idx)).round(2)

# --- Inject a drift segment: last 500 rows (by original order) have a shifted distribution ---
drift_idx = df.index[-500:]
df.loc[drift_idx, "amount"] = np.random.normal(loc=300, scale=80, size=len(drift_idx)).round(2)

# --- Inject duplicate rows (~3%) -- done LAST so duplicates are exact copies
#     of already-modified rows (matches the real-world case of a row being
#     accidentally re-inserted after processing) ---
dup_rows = df.sample(frac=0.03, random_state=1)
df = pd.concat([df, dup_rows], ignore_index=True)

# Shuffle rows so issues aren't neatly grouped
df = df.sample(frac=1, random_state=7).reset_index(drop=True)

df.to_csv("/home/claude/data-quality-engine/data/transactions.csv", index=False)
print(f"Generated {len(df)} rows -> data/transactions.csv")
print(df.head(10))
print("\nMissing values per column:")
print(df.isna().sum())
print(f"\nDuplicate rows: {df.duplicated(subset=['customer_id','transaction_date','amount']).sum()}")
