"""Tests for corruption utilities and the robustness experiment."""
import numpy as np
import pandas as pd
import pytest
from src.benchmarking.corruption import inject_missing_values, inject_outliers, inject_label_noise
from src.benchmarking.ml_evaluator import evaluate_classification
from src.benchmarking.robustness_experiment import run_label_noise_experiment, run_noise_sweep


def test_missing_values():
    original = pd.DataFrame({'amount': range(100)})
    changed = inject_missing_values(original, ['amount'], .2, seed=42)
    assert changed.amount.isna().sum() == 20
    assert original.amount.isna().sum() == 0
    pd.testing.assert_frame_equal(changed, inject_missing_values(original, ['amount'], .2, seed=42))


def test_outliers():
    original = pd.DataFrame({'amount': range(100)})
    changed = inject_outliers(original, ['amount'], .1, severity=10, seed=42)
    assert (changed.amount != original.amount).sum() == 10
    assert original.amount.iloc[0] == 0
    pd.testing.assert_frame_equal(changed, inject_outliers(original, ['amount'], .1, severity=10, seed=42))


def test_label_noise():
    original = pd.DataFrame({'label': [0, 1] * 50})
    changed = inject_label_noise(original, 'label', .2, seed=42)
    assert (changed.label != original.label).sum() == 20
    assert original.label.tolist() == [0, 1] * 50
    pd.testing.assert_frame_equal(changed, inject_label_noise(original, 'label', .2, seed=42))


def test_metrics():
    metrics = evaluate_classification([0, 0, 1, 1], [0, 1, 1, 1])
    assert metrics['accuracy'] == .75
    assert metrics['precision_macro'] == pytest.approx(5 / 6)
    assert metrics['f1_macro'] == pytest.approx(11 / 15)


def test_robustness_reproducibility():
    a = run_label_noise_experiment(.2, 42)
    b = run_label_noise_experiment(.2, 42)
    assert a == b
    assert a['clean']['n_samples'] == 250


def test_zero_noise():
    result = run_label_noise_experiment(0, 42)
    assert result['accuracy_change'] == pytest.approx(0)
    assert result['f1_macro_change'] == pytest.approx(0)


def test_sweep():
    raw, summary = run_noise_sweep(rates=[0, .2], seeds=[42, 43])
    assert len(raw) == 4
    assert len(summary) == 2
    assert np.isfinite(summary['accuracy_change_mean']).all()
