"""
FastAPI backend for the Data Quality & Anomaly Detection Engine.
Exposes the detection and remediation logic as REST endpoints.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from detection.anomaly_detector import run_quality_check
from detection.remediation import build_remediation_report

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "transactions.csv"
)

app = FastAPI(title="Data Quality & Anomaly Detection Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for approval decisions (swap for PostgreSQL next)
approval_log = {}


class ApprovalRequest(BaseModel):
    transaction_id: int
    decision: str  # "approved" or "rejected"
    reviewer: Optional[str] = "Umme Rubab"


@app.get("/")
def root():
    return {"status": "ok", "service": "Data Quality & Anomaly Detection Engine"}


@app.get("/scorecard")
def get_scorecard():
    """Returns the data health scorecard: totals, missing values, duplicates, anomalies."""
    try:
        report = run_quality_check(DATA_PATH)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/suggestions")
def get_suggestions():
    """Returns all remediation suggestions with confidence levels."""
    try:
        suggestions, summary = build_remediation_report(DATA_PATH)
        return {"summary": summary, "suggestions": suggestions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/approve")
def approve_suggestion(request: ApprovalRequest):
    """Records a human reviewer's approve/reject decision for a suggestion."""
    if request.decision not in ("approved", "rejected"):
        raise HTTPException(status_code=400, detail="decision must be 'approved' or 'rejected'")

    approval_log[request.transaction_id] = {
        "decision": request.decision,
        "reviewer": request.reviewer,
    }
    return {"status": "recorded", "transaction_id": request.transaction_id, "decision": request.decision}


@app.get("/approvals")
def get_approvals():
    """Returns the full approval/rejection audit log."""
    return {"total_reviewed": len(approval_log), "log": approval_log}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
