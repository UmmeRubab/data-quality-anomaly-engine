# DataGuard AI
### Data Quality, Anomaly Detection & ML Robustness Platform

**DataGuard AI** is a Python-based application for detecting data quality issues, identifying statistical anomalies, generating human-review-ready remediation suggestions, and evaluating how data corruption affects machine learning models.

The project evolved from a transaction-focused data quality engine into an interactive platform combining **data governance, anomaly detection, human-in-the-loop review, and machine learning robustness evaluation**.

**Live Streamlit Application:** https://dataguardai.streamlit.app/

## 1. Project Overview

DataGuard AI addresses two connected challenges:

1. **Data Quality and Governance:** Identifying missing values, duplicates, and statistical anomalies while supporting human-reviewed remediation decisions.
2. **Machine Learning Reliability:** Evaluating how data quality problems, such as missing values, label noise, and numerical outliers, affect predictive model performance.

The project retains its original detect-and-suggest architecture while extending it with an interactive Streamlit dashboard and controlled ML experiments.

## 2. Key Features

### A. Data Quality Assessment

- Detect missing values and duplicate records.
- Identify statistical outliers using Isolation Forest.
- Generate column-level data quality profiles.
- Inspect dataset dimensions, column types, and summary statistics.
- Calculate a heuristic Data Health Score (0–100).
- Explore flagged records and export analysis results.

### B. Human-in-the-Loop Remediation

The original engine follows this workflow:

**Detect → Suggest → Human Reviews → Record Decision → Apply Separately**

The transaction-specific remediation engine can generate suggestions such as:

- Fill missing transaction amounts using the median for the relevant category.
- Identify exact or likely duplicate transactions.
- Flag suspicious numerical values for manual investigation.

The original remediation logic assigns confidence levels such as High, Medium, and Low.

The upgraded Streamlit interface additionally supports recording review decisions:

- Approve
- Reject
- Needs Investigation

**Important:** Streamlit review decisions are session-based. They do not automatically modify the uploaded dataset or create a permanent audit history.

### C. Machine Learning Robustness Evaluation

DataGuard AI includes controlled experiments to investigate how corrupted training data affects predictive performance.

**Supported models:**
- Logistic Regression
- Random Forest
- Histogram-Based Gradient Boosting

**Supported corruption scenarios:**
- Label noise
- Missing values
- Numerical outliers

**Evaluation metrics:**
- Accuracy
- Macro F1-score
- Balanced accuracy

The experiment engine compares clean and corrupted training conditions against an untouched held-out test set.

This functionality helps demonstrate the relationship between data quality and machine learning reliability.

### D. Interactive Streamlit Dashboard

The upgraded application provides:

- CSV dataset upload
- Dataset selection and exploration
- Data quality reports
- Missing-value and duplicate analysis
- Isolation Forest anomaly detection
- Human review of remediation suggestions
- ML robustness experiments
- Interactive charts and downloadable reports

**Live Application:** https://dataguardai.streamlit.app/

## 3. Supported Datasets

### Synthetic Transaction Dataset

The original project includes a synthetic transaction dataset containing deliberately introduced data quality problems.

These include missing values, duplicate transactions, and anomalous numerical observations.

### UCI Bank Marketing Dataset

The upgraded project supports the UCI Bank Marketing dataset for structured-data analysis and ML robustness experiments.

**Source:** https://archive.ics.uci.edu/dataset/222/bank+marketing

The experiment engine includes an option to exclude the `duration` variable when predicting the subscription target to reduce the risk of misleading predictive evaluation.

### Custom CSV Upload

Users can upload structured CSV files to explore:

- Missing data
- Duplicate records
- Column characteristics
- Numerical anomalies
- Model performance under selected corruption scenarios

Some ML experiments require a suitable classification target and sufficient labeled examples.

## 4. Why Detect and Suggest Instead of Automatically Fixing Data?

Automatically modifying financial or transactional records without review can introduce errors and compromise data integrity.

DataGuard AI therefore emphasizes a human-in-the-loop approach.

The original engine produces remediation recommendations with confidence levels, while the Streamlit dashboard allows users to review and record decisions.

This design demonstrates the importance of human oversight in data governance workflows.

## 5. Technology Stack

| Component | Technologies |
|---|---|
| Programming Language | Python |
| Interactive Dashboard | Streamlit |
| Original Backend API | FastAPI |
| Data Processing | Pandas, NumPy |
| Anomaly Detection | Scikit-learn, Isolation Forest |
| Machine Learning | Logistic Regression, Random Forest, Histogram-Based Gradient Boosting |
| Data Profiling | Pandas, custom validation logic |
| Visualization | Streamlit charts |
| Deployment | Streamlit Community Cloud |
| Version Control | Git, GitHub |

## 6. System Architecture

```text
                 DataGuard AI
                      |
            Dataset Input Layer
                      |
         +------------+------------+
         |            |            |
      CSV Upload   Synthetic    UCI Bank
                   Transactions Marketing
         |            |            |
         +------------+------------+
                      |
               Data Profiling
                      |
          +-----------+-----------+
          |                       |
          v                       v
    Data Quality Engine      ML Robustness Engine
          |                       |
    Missing Values            Clean Baseline
    Duplicate Detection       Data Corruption
    Isolation Forest          Model Training
    Health Score              Performance Evaluation
          |                       |
          v                       v
    Remediation Engine        Metric Comparison
          |                       |
    Human Review                  |
          |                       |
          +-----------+-----------+
                      |
              Streamlit Dashboard
                      |
              Reports & Exports
```

The repository also retains the original FastAPI implementation as a separate backend interface.

## 7. Original FastAPI Backend

The original project exposes data quality and remediation functionality through FastAPI.

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/scorecard` | Retrieve the data quality scorecard |
| GET | `/suggestions` | Retrieve remediation recommendations |
| POST | `/approve` | Record a reviewer's approval or rejection |
| GET | `/approvals` | Retrieve recorded review decisions |

These endpoints belong to the original backend implementation. They are separate from the public Streamlit interface.

The original API records reviewer decisions in memory rather than in a persistent production database.

## 8. Original Data Quality Example

An example from the original transaction-focused implementation:

| Metric | Example result |
|---|---:|
| Total rows scanned | 5,150 |
| Duplicate rows detected | 300 |
| Anomalous rows detected | 148 |
| Data Health Score | 90.7 / 100 |

These figures illustrate an earlier project run and are not guaranteed results for every uploaded dataset.

## 9. Project Structure

```text
DataGuard-AI/
|
|-- app.py
|-- requirements.txt
|-- README.md
|
|-- data/
|   |-- transactions.csv
|
|-- phase2_backend/
|   |-- dataset_engine.py
|   |-- main.py
|   |-- test_api.py
|   |-- test_dataset_engine.py
|
|-- phase2_frontend/
|   |-- [Original Next.js frontend files]
|
|-- src/
|   |-- ingestion/
|   |   |-- generate_data.py
|   |
|   |-- detection/
|   |   |-- anomaly_detector.py
|   |   |-- remediation.py
|   |
|   |-- benchmarking/
|   |   |-- corruption.py
|   |   |-- ml_evaluator.py
|   |   |-- robustness_experiment.py
|   |
|   |-- api/
|       |-- main.py
|
|-- tests/
|   |-- test_benchmarking.py
|
|-- BENCHMARKING.md
|-- DEPLOYMENT_AND_UPGRADE.md
|-- README_SETUP.md
|-- README_START_HERE.md
|-- render.yaml
```

The original Next.js and FastAPI components remain in the repository for reference and separate development. The public application uses Streamlit.

## 10. Installation and Local Execution

### Clone the Repository

```bash
git clone https://github.com/UmmeRubab/DataGuard-AI.git
cd DataGuard-AI
```

### Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Run the Streamlit Dashboard

```bash
python -m streamlit run app.py
```

Open the application at:

`http://localhost:8501`

### Original FastAPI Backend

If you wish to run the original backend separately, install its required dependencies and start the appropriate FastAPI application.

The Streamlit dashboard does not require a separately deployed FastAPI service.

## 11. Deployment

**Current deployment:** Streamlit Community Cloud

**Live URL:** https://dataguardai.streamlit.app/

The application runs from the repository's `app.py` entry point.

The earlier project architecture included a Next.js frontend and FastAPI backend, with deployment configurations for Vercel and Render. Those components are retained in the repository, but the current public dashboard uses Streamlit.

## 12. Limitations

- The Data Health Score is a heuristic measure.
- Statistical anomalies are not necessarily incorrect data.
- Review decisions in the Streamlit dashboard are not permanently stored.
- Approving a suggestion does not automatically apply a data correction.
- The original FastAPI approval log is not a persistent database-backed audit system.
- ML robustness results depend on the dataset, model, corruption type, and experimental settings.
- Single-split comparisons do not establish statistical significance.
- The application is an exploratory applied AI project rather than a production-certified data governance solution.

## 13. Future Improvements

- PostgreSQL integration for persistent approval history.
- Configurable validation rules and schema checks.
- More advanced anomaly detection algorithms.
- Automated, reversible remediation workflows with approval controls.
- Explainable anomaly detection.
- Additional corruption and distribution-shift experiments.
- Repeated-seed robustness evaluation.
- Improved data lineage and audit reporting.
- Docker-based deployment options.

## 14. Project Evolution

| Original implementation | Upgraded DataGuard AI |
|---|---|
| Transaction-focused quality analysis | General structured CSV exploration |
| Missing-value and duplicate detection | Extended column-level profiling |
| Isolation Forest outlier detection | Interactive anomaly analysis |
| Confidence-tiered remediation suggestions | Human review dashboard |
| FastAPI backend | Streamlit public application with retained FastAPI code |
| Transaction health score | Interactive data health reporting |
| Synthetic transactions | Synthetic, UCI Bank Marketing, and uploaded CSV datasets |
| Data quality focus | Data quality plus ML robustness evaluation |

## 15. Project Status

**Active — Streamlit Application Deployed**

DataGuard AI demonstrates practical experience in:

- Data quality engineering
- Data governance workflows
- Statistical anomaly detection
- Human-in-the-loop decision support
- Machine learning evaluation
- Data corruption and robustness experimentation
- Python application development

## Author

**Umm E Rubab**

**GitHub:** https://github.com/UmmeRubab

**Live Application:** https://dataguardai.streamlit.app/
