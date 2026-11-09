# DataGuard  â€” Phase 1: ML Robustness Benchmark

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
