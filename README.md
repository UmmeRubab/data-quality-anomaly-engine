# Data Quality & Anomaly Detection Engine

An AI-powered system that scans structured datasets, detects data quality 
issues, and generates human-review-ready remediation suggestions — designed 
around a "detect → suggest → human approves → apply" workflow rather than 
silent auto-correction.

## Overview
Given a transactions dataset, the system:
1. Detects missing values, duplicate rows, and statistical outliers 
   (Isolation Forest)
2. Computes a single Data Health Score (0–100)
3. Generates specific, confidence-tiered remediation suggestions for each 
   issue (e.g., "fill missing amount with category median", "remove exact 
   duplicate", "flag outlier for manual review")
4. Exposes everything through a FastAPI backend with an approve/reject 
   endpoint that logs reviewer decisions — an audit trail, not just a report

## Why detect-and-suggest, not auto-fix
Automatically modifying financial/transactional data without human review is 
risky in real governance settings. This system mirrors real-world data 
steward workflows: every suggestion carries a confidence level (High/Medium/Low) 
and nothing is applied without explicit approval.

## Tech Stack
- **Detection:** scikit-learn (Isolation Forest)
- **Remediation logic:** Python, pandas
- **Backend API:** FastAPI
- **Data:** Synthetic messy transactions dataset (generated with pandas/numpy, 
  simulating real-world issues: missing values, duplicates, outliers, drift)

## API Endpoints
- `GET /scorecard` — full data health scorecard
- `GET /suggestions` — remediation suggestions with confidence levels
- `POST /approve` — record a reviewer's approve/reject decision
  
## Example Output
Total rows scanned:        5,150
Duplicate rows found:      300
Anomalous rows detected:   148
Data Health Score:         90.7 / 100

## Project Structure
data-quality-anomaly-engine/
├── data/                   # Synthetic messy dataset
├── src/
│   ├── ingestion/           # Dataset generation
│   ├── detection/            # Anomaly detection + remediation engine
│   └── api/                    # FastAPI backend
└── requirements.txt

## Roadmap
- [ ] PostgreSQL for persistent approval history
- [ ] Streamlit dashboard for interactive review
- [ ] Docker Compose for one-command deployment
- [ ] Live deployment

## Author
Umme Rubab
- `GET /approvals` — audit trail of all review decisions

## Example Output
