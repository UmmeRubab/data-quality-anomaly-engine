"""Reproducible data corruption utilities for DataGuard AI."""

import numpy as np
import pandas as pd


def inject_missing_values(
    df: pd.DataFrame,
    columns: list[str],
    rate: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """Randomly replace a specified fraction of each column with missing values."""
    if not 0 <= rate <= 1:
        raise ValueError("rate must be between 0 and 1")

    if not columns:
        raise ValueError("At least one column must be specified")

    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Unknown columns: {missing_columns}")

    result = df.copy(deep=True)
    rng = np.random.default_rng(seed)
    n_rows = len(result)
    n_corrupt = round(n_rows * rate)

    for column in columns:
        positions = rng.choice(n_rows, size=n_corrupt, replace=False)
        result.iloc[positions, result.columns.get_loc(column)] = np.nan

    return result

def inject_outliers(
    df: pd.DataFrame,
    columns: list[str],
    rate: float = 0.1,
    severity: float = 5.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Inject reproducible extreme values into numeric columns."""
    if not 0 <= rate <= 1:
        raise ValueError("rate must be between 0 and 1")
    if severity <= 0:
        raise ValueError("severity must be positive")
    if not columns:
        raise ValueError("At least one column must be specified")

    result = df.copy(deep=True)
    rng = np.random.default_rng(seed)

    for column in columns:
        if column not in result.columns:
            raise ValueError(f"Unknown column: {column}")
        if not pd.api.types.is_numeric_dtype(result[column]):
            raise TypeError(f"Column must be numeric: {column}")

        result[column] = result[column].astype(float)
        values = result[column].to_numpy(dtype=float)
        valid_positions = np.flatnonzero(np.isfinite(values))
        n_corrupt = round(len(valid_positions) * rate)

        if n_corrupt == 0:
            continue

        positions = rng.choice(valid_positions, size=n_corrupt, replace=False)
        scale = np.std(values[valid_positions])
        if scale == 0:
            scale = max(abs(np.mean(values[valid_positions])), 1.0)

        directions = rng.choice([-1, 1], size=n_corrupt)
        result.iloc[positions, result.columns.get_loc(column)] = (
            values[positions] + directions * severity * scale
        )

    return result


def inject_label_noise(
    df: pd.DataFrame,
    target_column: str,
    rate: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """Flip a controlled fraction of nonmissing classification labels."""
    if not 0 <= rate <= 1:
        raise ValueError("rate must be between 0 and 1")
    if target_column not in df.columns:
        raise ValueError(f"Unknown target column: {target_column}")

    result = df.copy(deep=True)
    valid_positions = np.flatnonzero(result[target_column].notna().to_numpy())
    classes = pd.unique(result[target_column].dropna())

    if len(classes) < 2:
        raise ValueError("At least two distinct nonmissing classes are required")

    rng = np.random.default_rng(seed)
    n_corrupt = round(len(valid_positions) * rate)
    positions = rng.choice(valid_positions, size=n_corrupt, replace=False)

    column_index = result.columns.get_loc(target_column)

    for position in positions:
        original_label = result.iat[position, column_index]
        alternatives = classes[classes != original_label]
        result.iat[position, column_index] = rng.choice(alternatives)

    return result
