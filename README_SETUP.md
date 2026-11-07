# DataGuard AI — Phase 2 starter

This package is an **additive starter**, not a completed production system. It assumes the Phase 1 `src/benchmarking/robustness_experiment.py` exists and exports `run_label_noise_experiment`.

## Where files go

Extract this ZIP somewhere convenient. In your existing repository at `E:\PhD USA\Projects\data-quality-engine`:

1. Copy the ZIP's `backend` directory into the repository root and **rename the copied directory to `phase2_backend`**.
2. Copy the ZIP's `frontend` directory into the repository root and **rename the copied directory to `phase2_frontend`**.
3. Add an empty file `phase2_backend/__init__.py` (already included in the ZIP).
4. Do not replace your existing `src`, `tests`, `data`, `.gitignore`, or `requirements.txt`.

Expected:
```
data-quality-engine/
  src/
  tests/
  results/
  phase2_backend/
    __init__.py
    main.py
    test_api.py
  phase2_frontend/
    app/
    package.json
    ...
```

## Backend (PowerShell, repository root, with .venv active)

```powershell
python -m pip install fastapi uvicorn httpx
python -m pytest -q phase2_backend/test_api.py
python -m uvicorn phase2_backend.main:app --reload --port 8000
```

Open http://127.0.0.1:8000/docs to test the API.

## Frontend (second VS Code terminal)

Install Node.js LTS if needed, then:

```powershell
cd phase2_frontend
npm install
Copy-Item .env.local.example .env.local
npm run dev
```

Open http://localhost:3000.

## Verify existing Phase 1 tests

From the repository root in a Python virtual environment:

```powershell
python -m pytest -q tests/test_benchmarking.py
```

## Git

Review first:

```powershell
git status --short
```

Do not commit `.venv`, `node_modules`, `.next`, or secrets. The frontend contains its own `.gitignore`. Do not push or merge without reviewing changes.

## Research limitations

The label-noise endpoint runs a synthetic benchmark in-process. For longer experiments, use background jobs and persistent storage. The preview endpoint reads only CSV files from the benchmark output directory. This starter has no authentication and is intended for local development only.
