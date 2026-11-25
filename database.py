"""
DR-RiskPredict — Database layer (v2: adds users, edit/delete, search, chart data)
"""

import sqlite3
import os
import hashlib
import secrets
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "dr_riskpredict.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    # Patients table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT NOT NULL,
            age INTEGER NOT NULL,
            sex TEXT NOT NULL DEFAULT 'Female',
            diabetes_type TEXT NOT NULL,
            duration_years REAL NOT NULL,
            hba1c REAL NOT NULL,
            systolic_bp REAL NOT NULL,
            diastolic_bp REAL NOT NULL,
            cholesterol REAL,
            triglycerides REAL,
            bmi REAL,
            smoking TEXT NOT NULL,
            risk_label TEXT NOT NULL,
            risk_probability REAL NOT NULL,
            model_used TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    # Migration guard: if an older DB (pre-"sex" column) already exists,
    # add the column in place rather than requiring a fresh database.
    cur.execute("PRAGMA table_info(patients)")
    existing_cols = {row["name"] for row in cur.fetchall()}
    if "sex" not in existing_cols:
        cur.execute("ALTER TABLE patients ADD COLUMN sex TEXT NOT NULL DEFAULT 'Female'")

    # Users table for authentication
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'staff',
            created_at TEXT NOT NULL
        )
    """)

    # Multi-disease risk predictions (one patient screening -> many disease rows)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS disease_risk_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            disease_key TEXT NOT NULL,
            disease_name TEXT NOT NULL,
            risk_score REAL NOT NULL,
            risk_level TEXT NOT NULL,
            explanation TEXT,
            contributing_factors TEXT,
            source TEXT NOT NULL DEFAULT 'rule_based',
            created_at TEXT NOT NULL,
            FOREIGN KEY (patient_id) REFERENCES patients (id) ON DELETE CASCADE
        )
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_disease_predictions_patient
        ON disease_risk_predictions (patient_id)
    """)

    # Create default admin user if no users exist
    cur.execute("SELECT COUNT(*) as c FROM users")
    if cur.fetchone()["c"] == 0:
        salt = secrets.token_hex(16)
        pw_hash = hashlib.sha256((salt + "admin123").encode()).hexdigest()
        cur.execute("""
            INSERT INTO users (username, password_hash, salt, role, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, ("admin", pw_hash, salt, "admin", datetime.now().isoformat(timespec="seconds")))

    conn.commit()
    conn.close()


# ---- Auth ----
def verify_user(username, password):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cur.fetchone()
    conn.close()
    if not user:
        return None
    pw_hash = hashlib.sha256((user["salt"] + password).encode()).hexdigest()
    if pw_hash == user["password_hash"]:
        return dict(user)
    return None


def get_user_by_id(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = cur.fetchone()
    conn.close()
    return dict(user) if user else None


# ---- Patients CRUD ----
def insert_prediction(record: dict):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO patients (
            patient_name, age, sex, diabetes_type, duration_years, hba1c,
            systolic_bp, diastolic_bp, cholesterol, triglycerides, bmi,
            smoking, risk_label, risk_probability, model_used, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        record["patient_name"], record["age"], record["sex"], record["diabetes_type"],
        record["duration_years"], record["hba1c"], record["systolic_bp"],
        record["diastolic_bp"], record.get("cholesterol"),
        record.get("triglycerides"), record.get("bmi"), record["smoking"],
        record["risk_label"], record["risk_probability"], record["model_used"],
        datetime.now().isoformat(timespec="seconds")
    ))
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def get_record_by_id(record_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM patients WHERE id = ?", (record_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def update_record(record_id, data):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE patients SET
            patient_name=?, age=?, sex=?, diabetes_type=?, duration_years=?, hba1c=?,
            systolic_bp=?, diastolic_bp=?, cholesterol=?, triglycerides=?, bmi=?,
            smoking=?
        WHERE id=?
    """, (
        data["patient_name"], data["age"], data["sex"], data["diabetes_type"],
        data["duration_years"], data["hba1c"], data["systolic_bp"],
        data["diastolic_bp"], data.get("cholesterol"), data.get("triglycerides"),
        data.get("bmi"), data["smoking"], record_id
    ))
    conn.commit()
    conn.close()


def delete_record(record_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM disease_risk_predictions WHERE patient_id = ?", (record_id,))
    cur.execute("DELETE FROM patients WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()


# ---- Multi-disease risk predictions ----
def insert_disease_predictions(patient_id, predictions: list):
    """
    Bulk-store the output of disease_rules.predict_all_diseases() for a
    given patient/screening record. Replaces any prior rows for that
    patient (keeps the table in sync with the latest screening values).
    """
    import json as _json
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM disease_risk_predictions WHERE patient_id = ?", (patient_id,))
    now = datetime.now().isoformat(timespec="seconds")
    for p in predictions:
        cur.execute("""
            INSERT INTO disease_risk_predictions (
                patient_id, disease_key, disease_name, risk_score, risk_level,
                explanation, contributing_factors, source, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            patient_id, p.get("key"), p.get("disease_name"), p.get("risk_score"),
            p.get("risk_level"), p.get("explanation"),
            _json.dumps(p.get("contributing_factors", [])),
            p.get("source", "rule_based"), now
        ))
    conn.commit()
    conn.close()


def get_disease_predictions(patient_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM disease_risk_predictions
        WHERE patient_id = ? ORDER BY risk_score DESC
    """, (patient_id,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_all_records(limit=500, search=None, risk_filter=None, sort="id_desc"):
    conn = get_connection()
    cur = conn.cursor()
    query = "SELECT * FROM patients WHERE 1=1"
    params = []
    if search:
        query += " AND patient_name LIKE ?"
        params.append(f"%{search}%")
    if risk_filter in ("High Risk", "Moderate Risk", "Low Risk"):
        query += " AND risk_label = ?"
        params.append(risk_filter)
    order = {
        "id_desc": "id DESC",
        "id_asc": "id ASC",
        "name_asc": "patient_name ASC",
        "risk_desc": "risk_probability DESC",
        "date_desc": "created_at DESC",
    }.get(sort, "id DESC")
    query += f" ORDER BY {order} LIMIT ?"
    params.append(limit)
    cur.execute(query, params)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_stats():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) as total FROM patients")
    total = cur.fetchone()["total"]
    cur.execute("SELECT COUNT(*) as c FROM patients WHERE risk_label = 'High Risk'")
    high_risk = cur.fetchone()["c"]
    cur.execute("SELECT COUNT(*) as c FROM patients WHERE risk_label = 'Moderate Risk'")
    moderate_risk = cur.fetchone()["c"]
    conn.close()
    return {
        "total": total,
        "high_risk": high_risk,
        "moderate_risk": moderate_risk,
        "low_risk": total - high_risk - moderate_risk,
    }


def get_chart_data():
    """Returns data for dashboard charts."""
    conn = get_connection()
    cur = conn.cursor()

    # Risk by diabetes type
    cur.execute("""
        SELECT diabetes_type,
               SUM(CASE WHEN risk_label='High Risk' THEN 1 ELSE 0 END) as high,
               SUM(CASE WHEN risk_label='Moderate Risk' THEN 1 ELSE 0 END) as moderate,
               SUM(CASE WHEN risk_label='Low Risk' THEN 1 ELSE 0 END) as low
        FROM patients GROUP BY diabetes_type
    """)
    risk_by_type = [dict(r) for r in cur.fetchall()]

    # Age group distribution
    cur.execute("""
        SELECT
          CASE
            WHEN age < 40 THEN '<40'
            WHEN age < 50 THEN '40-49'
            WHEN age < 60 THEN '50-59'
            WHEN age < 70 THEN '60-69'
            ELSE '70+'
          END as age_group,
          COUNT(*) as count,
          SUM(CASE WHEN risk_label='High Risk' THEN 1 ELSE 0 END) as high_risk,
          SUM(CASE WHEN risk_label='Moderate Risk' THEN 1 ELSE 0 END) as moderate_risk
        FROM patients GROUP BY age_group ORDER BY age_group
    """)
    age_groups = [dict(r) for r in cur.fetchall()]

    # Screening trend (last 10 screenings by date)
    cur.execute("""
        SELECT substr(created_at, 1, 10) as date, COUNT(*) as count,
               SUM(CASE WHEN risk_label='High Risk' THEN 1 ELSE 0 END) as high_risk
        FROM patients
        GROUP BY substr(created_at, 1, 10)
        ORDER BY date DESC LIMIT 10
    """)
    trend = list(reversed([dict(r) for r in cur.fetchall()]))

    # HbA1c distribution
    cur.execute("""
        SELECT
          CASE
            WHEN hba1c < 7 THEN '<7%'
            WHEN hba1c < 8 THEN '7-8%'
            WHEN hba1c < 9 THEN '8-9%'
            WHEN hba1c < 10 THEN '9-10%'
            ELSE '10%+'
          END as range,
          COUNT(*) as count
        FROM patients GROUP BY range
    """)
    hba1c_dist = [dict(r) for r in cur.fetchall()]

    conn.close()
    return {
        "risk_by_type": risk_by_type,
        "age_groups": age_groups,
        "trend": trend,
        "hba1c_dist": hba1c_dist,
    }


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
