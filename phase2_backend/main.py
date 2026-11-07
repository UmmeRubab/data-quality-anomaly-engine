"""DataGuard AI HTTP API.

Run from the repository root with:
    uvicorn phase2_backend.main:app --reload
"""
from __future__ import annotations

from pathlib import Path
import csv
import math

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.benchmarking.robustness_experiment import run_label_noise_experiment

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "benchmarking"
app = FastAPI(title="DataGuard AI API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ExperimentRequest(BaseModel):
    noise_rate: float = Field(default=0.2, ge=0, le=1)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "DataGuard AI"}


@app.post("/api/experiments/label-noise")
def label_noise_experiment(request: ExperimentRequest):
    try:
        return run_label_noise_experiment(noise_rate=request.noise_rate, seed=request.seed)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/results/files")
def list_result_files():
    if not RESULTS.is_dir():
        return {"files": [], "message": "Run the Phase 1 benchmark to generate results."}
    return {"files": sorted(p.name for p in RESULTS.glob("*.csv") if p.is_file())}


@app.get("/api/results/preview")
def preview_results(filename: str = Query(..., min_length=5, max_length=160), limit: int = Query(30, ge=1, le=200)):
    # Only allow a CSV basename, never a filesystem path.
    if Path(filename).name != filename or not filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Provide a CSV filename only")
    path = RESULTS / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Result file not found")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for i, row in enumerate(reader):
            if i >= limit:
                break
            rows.append(row)
    return {"filename": filename, "columns": reader.fieldnames or [], "rows": rows}
