# DR-RiskPredict

Clinical data-based risk prediction model for diabetic retinopathy (DR), built with
**Scikit-learn** and served through **Flask**. Estimates DR risk from routine clinical
values (HbA1c, blood pressure, duration of diabetes, lipids, etc.) — no fundus image
required — to help clinics triage who needs an eye exam first.

## Project structure

```
DR-RiskPredict/
├── app.py                      # Flask application (routes, validation, prediction)
├── database.py                 # SQLite data layer (patients table)
├── train_model.py              # Week 2: data cleaning, train/test split, model training
├── generate_report_charts.py   # Week 4: ROC curve + confusion matrix charts
├── requirements.txt
├── data/
│   ├── generate_dataset.py     # Mock clinical dataset generator
│   ├── clinical_data.csv       # 180-row clinical dataset
│   ├── schema_mysql.sql        # MySQL DDL (production deployment)
│   └── dr_riskpredict.db       # SQLite DB (created at runtime)
├── models/                     # Saved model artifacts (.pkl) + metrics.json
├── reports/                     # Generated ROC curve & confusion matrix PNGs (reference docs only)
├── templates/                   # Jinja2 HTML templates (home, screen patient, dashboard, result)
└── docs/
    ├── user_manual.md
    └── technical_report.md
```

## Setup

```bash
pip install -r requirements.txt

# 1. Generate the clinical dataset
python3 data/generate_dataset.py

# 2. Train models (Logistic Regression + Random Forest) and select the best
python3 train_model.py

# 3. Run the app
python3 app.py
```

The app runs at **http://localhost:8080**.

> **If you see a model-loading error** (e.g. `'LogisticRegression' object has no attribute ...`),
> your installed scikit-learn version doesn't match the one the saved `.pkl` files were
> trained with. Fix it by retraining locally: `python3 train_model.py`, then restart the app.

## Routes

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Home / overview + live stats |
| `/predict-form` | GET | Patient screening form (with live risk gauge) |
| `/predict` | POST | Run prediction, save to DB, show result |
| `/api/predict` | POST | JSON API version (no DB write, no name required) |
| `/dashboard` | GET | Table of all screened patients |

A standalone technical report (model methodology, metrics, ROC curve, confusion
matrix) is in `docs/technical_report.md` — generate fresh chart images any time with
`python3 generate_report_charts.py` (writes to `reports/`).

## JSON API example

```bash
curl -X POST http://localhost:8080/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 58, "diabetes_type": "Type 2", "duration_years": 12,
    "hba1c": 9.2, "systolic_bp": 145, "diastolic_bp": 88,
    "cholesterol": 230, "triglycerides": 180, "bmi": 29.5, "smoking": "Yes"
  }'
```

Response:
```json
{"risk_label": "High Risk", "risk_probability": 0.93, "model_used": "Logistic Regression"}
```

## Switching to MySQL

This build uses SQLite for portability during development. To use MySQL instead:
1. Run `data/schema_mysql.sql` on your MySQL server.
2. Replace `database.py`'s `sqlite3` calls with a MySQL connector (e.g. `mysql-connector-python`
   or `PyMySQL`) using the same function signatures (`init_db`, `insert_prediction`,
   `get_all_records`, `get_stats`) — the rest of the app does not need to change.

## Important note on the dataset

The clinical dataset (`data/clinical_data.csv`) is **synthetically generated** for this
build (see `data/generate_dataset.py`) since no live data source was available. Risk
factors and their relationship to DR status follow established clinical patterns (e.g.
higher HbA1c, longer diabetes duration, and higher blood pressure increase risk), but
this is not real patient data. Before any clinical use, retrain on a real, ethically
sourced clinical dataset.
