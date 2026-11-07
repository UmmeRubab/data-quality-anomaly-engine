"""Classification evaluation utilities for DataGuard AI."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)


def evaluate_classification(y_true, y_pred) -> dict:
    """Calculate reproducible classification performance metrics."""
    actual = np.asarray(y_true)
    predicted = np.asarray(y_pred)

    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("Labels must be one-dimensional")

    if len(actual) == 0:
        raise ValueError("Labels cannot be empty")

    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted labels must have equal lengths")

    precision, recall, f1, _ = precision_recall_fscore_support(
        actual,
        predicted,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "n_samples": int(len(actual)),
    }
