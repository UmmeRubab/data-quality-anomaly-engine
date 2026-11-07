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

# DataGuard AI dataset intelligence endpoints (uploads processed in memory).
from fastapi import UploadFile, File
from phase2_backend.dataset_engine import parse_csv, profile, bank_marketing, run_experiment

class DatasetExperimentRequest(BaseModel):
    dataset: str = 'bank'
    target: str = 'y'
    model: str = 'logistic_regression'
    corruption: str = 'label_noise'
    rate: float = Field(default=0.2, ge=0, le=0.5)
    seed: int = Field(default=42, ge=0, le=2**32-1)
    drop_duration: bool = True

@app.get('/api/datasets/bank/profile')
def bank_profile():
    try: return profile(bank_marketing(),'UCI Bank Marketing (bank-full.csv)')
    except ValueError as exc: raise HTTPException(status_code=503,detail=str(exc)) from exc

@app.post('/api/datasets/upload/profile')
async def upload_profile(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=422,detail='Upload a .csv file')
    try:
        payload=await file.read(8*1024*1024+1)
        return profile(parse_csv(payload),file.filename)
    except ValueError as exc: raise HTTPException(status_code=422,detail=str(exc)) from exc
    finally: await file.close()

@app.post('/api/datasets/bank/experiment')
def bank_experiment(request: DatasetExperimentRequest):
    try:
        return run_experiment(bank_marketing(),request.target,request.model,request.corruption,request.rate,request.seed,request.drop_duration)
    except ValueError as exc:raise HTTPException(status_code=422,detail=str(exc)) from exc

@app.post('/api/datasets/upload/experiment')
async def upload_experiment(file: UploadFile = File(...),target: str = 'y',model: str='logistic_regression',corruption: str='label_noise',rate:float=0.2,seed:int=42):
    if not file.filename or not file.filename.lower().endswith('.csv'):raise HTTPException(status_code=422,detail='Upload a CSV file')
    try:
        df=parse_csv(await file.read(8*1024*1024+1))
        return run_experiment(df,target,model,corruption,rate,seed)
    except ValueError as exc:raise HTTPException(status_code=422,detail=str(exc)) from exc
    finally:await file.close()

# Public frontend origins must be explicitly configured at deployment.
import os
from starlette.middleware.cors import CORSMiddleware as _Cors
allowed=[o.strip().rstrip('/') for o in os.getenv('DATAGUARD_ALLOWED_ORIGINS','').split(',') if o.strip()]
if allowed:
    app.add_middleware(_Cors,allow_origins=allowed,allow_methods=['GET','POST'],allow_headers=['Content-Type'])
