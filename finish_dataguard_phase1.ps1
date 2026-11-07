$ErrorActionPreference = 'Stop'
$root = (Get-Location).Path
if (-not (Test-Path 'src/benchmarking/robustness_experiment.py')) { throw 'Run this script from the data-quality-engine repository root.' }
New-Item -ItemType Directory -Force -Path 'src/benchmarking','tests','results/benchmarking' | Out-Null
@'
"""Repeatable experiments measuring sensitivity to training-label corruption."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.benchmarking.corruption import inject_label_noise
from src.benchmarking.ml_evaluator import evaluate_classification


def run_label_noise_experiment(noise_rate: float = 0.2, seed: int = 42) -> dict:
    """Use an identical clean holdout for the clean and corrupted training runs."""
    if not 0 <= noise_rate <= 1:
        raise ValueError('noise_rate must be between 0 and 1')
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                               n_redundant=5, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed)
    noisy = inject_label_noise(pd.DataFrame({'label': y_train}), 'label',
                               rate=noise_rate, seed=seed)['label'].to_numpy()

    def evaluate(labels):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
        model.fit(X_train, labels)
        return evaluate_classification(y_test, model.predict(X_test))

    clean = evaluate(y_train)
    corrupted = evaluate(noisy)
    return {
        'noise_rate': float(noise_rate), 'seed': int(seed),
        'clean': clean, 'corrupted': corrupted,
        'accuracy_change': corrupted['accuracy'] - clean['accuracy'],
        'f1_macro_change': corrupted['f1_macro'] - clean['f1_macro'],
    }


def run_noise_sweep(rates=None, seeds=None):
    """Run paired comparisons for each rate and seed; return raw and summary frames."""
    rates = [0, .05, .10, .20, .30, .40, .50] if rates is None else list(rates)
    seeds = list(range(42, 52)) if seeds is None else list(seeds)
    if not rates or not seeds:
        raise ValueError('At least one rate and seed are required')
    records = []
    for seed in seeds:
        for rate in rates:
            result = run_label_noise_experiment(rate, seed)
            records.append({
                'seed': seed, 'noise_rate': rate,
                'clean_accuracy': result['clean']['accuracy'],
                'corrupted_accuracy': result['corrupted']['accuracy'],
                'accuracy_change': result['accuracy_change'],
                'clean_f1_macro': result['clean']['f1_macro'],
                'corrupted_f1_macro': result['corrupted']['f1_macro'],
                'f1_macro_change': result['f1_macro_change'],
            })
    raw = pd.DataFrame.from_records(records)
    summary_rows = []
    for rate, group in raw.groupby('noise_rate', sort=True):
        row = {'noise_rate': float(rate), 'n_seeds': len(group)}
        for metric in ('accuracy_change', 'f1_macro_change', 'clean_accuracy',
                       'corrupted_accuracy', 'clean_f1_macro', 'corrupted_f1_macro'):
            values = group[metric].to_numpy(dtype=float)
            row[f'{metric}_mean'] = float(values.mean())
            row[f'{metric}_std'] = float(values.std(ddof=1)) if len(values) > 1 else float('nan')
            # Normal-approximation intervals across repeated independent synthetic seeds.
            # Interpret descriptively, not as evidence of real-world generalization.
            half_width = 1.96 * row[f'{metric}_std'] / np.sqrt(len(values)) if len(values) > 1 else float('nan')
            row[f'{metric}_ci95_low'] = row[f'{metric}_mean'] - half_width
            row[f'{metric}_ci95_high'] = row[f'{metric}_mean'] + half_width
        summary_rows.append(row)
    return raw, pd.DataFrame(summary_rows)


def save_results(raw, summary, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / 'label_noise_raw.csv', index=False)
    summary.to_csv(out / 'label_noise_summary.csv', index=False)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for metric, label in [('accuracy_change', 'Accuracy change'),
                              ('f1_macro_change', 'Macro F1 change')]:
            x = summary['noise_rate'].to_numpy(dtype=float)
            y = summary[f'{metric}_mean'].to_numpy(dtype=float)
            lo = summary[f'{metric}_ci95_low'].to_numpy(dtype=float)
            hi = summary[f'{metric}_ci95_high'].to_numpy(dtype=float)
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(x * 100, y, marker='o')
            if np.isfinite(lo).all() and np.isfinite(hi).all():
                ax.fill_between(x * 100, lo, hi, alpha=0.2)
            ax.axhline(0, linestyle='--', linewidth=1)
            ax.set(xlabel='Training labels corrupted (%)', ylabel=label,
                   title=f'{label} vs. symmetric label noise')
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(out / f'{metric}.png', dpi=180)
            plt.close(fig)
    except ImportError:
        print('matplotlib unavailable; CSV files were saved without plots.')
    metadata = {
        'dataset': 'sklearn.datasets.make_classification (synthetic)',
        'model': 'StandardScaler + LogisticRegression(max_iter=1000)',
        'split': 'stratified 75% train / 25% untouched test',
        'corruption': 'symmetric random label flipping on training labels only',
        'rates': sorted(raw['noise_rate'].unique().tolist()),
        'seeds': sorted(raw['seed'].unique().tolist()),
        'confidence_intervals': 'normal approximation, mean +/- 1.96 * sample_sd / sqrt(n_seeds)',
        'caveat': 'Synthetic-data benchmark; confidence intervals are descriptive and do not establish real-world generalization.'
    }
    (out / 'experiment_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Run DataGuard AI label-noise robustness sweep')
    parser.add_argument('--seeds', type=int, default=10, help='Number of independent seeds (default 10)')
    parser.add_argument('--output', default='results/benchmarking')
    args = parser.parse_args()
    if args.seeds < 2:
        parser.error('--seeds must be at least 2 for confidence intervals')
    raw, summary = run_noise_sweep(seeds=range(42, 42 + args.seeds))
    save_results(raw, summary, args.output)
    print(summary[['noise_rate', 'n_seeds', 'accuracy_change_mean', 'f1_macro_change_mean']].to_string(index=False))
    print(f'Files saved to: {Path(args.output).resolve()}')


if __name__ == '__main__':
    main()
'@ | Set-Content -Encoding utf8 'src/benchmarking/robustness_experiment.py'
@'
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
'@ | Set-Content -Encoding utf8 'tests/test_benchmarking.py'
@'
# DataGuard AI — Phase 1: ML Robustness Benchmark

## Purpose
Quantify how controlled training-label corruption changes downstream classification performance, using the same untouched holdout test set for each clean/noisy comparison.

## Run
From the repository root, with the project's Python environment active:

```powershell
python -m pytest tests/test_benchmarking.py -q
python -m src.benchmarking.robustness_experiment --seeds 10
```

Results are written to `results/benchmarking/`:
- `label_noise_raw.csv`: per-seed paired results
- `label_noise_summary.csv`: means, sample standard deviations, approximate 95% confidence intervals
- `accuracy_change.png`, `f1_macro_change.png`: visualizations (requires matplotlib)
- `experiment_metadata.json`: experiment setup and limitations

## Protocol
- Synthetic binary classification dataset generated independently for each seed.
- Stratified 75/25 training/test split; test set is never corrupted.
- StandardScaler and LogisticRegression fitted separately to clean and corrupted training data.
- Symmetric label noise is applied only to training labels.
- Report accuracy and macro F1; negative differences mean corrupted training performed worse.
- Repeated seeds characterize variability of this synthetic experiment, not external validity.

## Limitations and next research stages
This is a benchmark scaffold, **not** evidence of a new algorithm or a real-world validation study. Missing-value and outlier corruption are implemented as utilities but are **not yet** included in the ML sweep. Future work should add train-only imputation, robust preprocessing, remediation strategies, multiple datasets/models, explicit data leakage checks, and stronger statistical comparisons. The full Next.js/FastAPI research application is also a separate development phase.
'@ | Set-Content -Encoding utf8 'BENCHMARKING.md'
@'
results/benchmarking/
'@ | Add-Content -Encoding utf8 '.gitignore'
python -m pytest tests/test_benchmarking.py -q
if ($LASTEXITCODE -ne 0) { throw 'Tests failed; review output before proceeding.' }
python -m src.benchmarking.robustness_experiment --seeds 10
if ($LASTEXITCODE -ne 0) { throw 'Benchmark failed; review output before proceeding.' }
Write-Host 'Phase 1 benchmark completed successfully.'
