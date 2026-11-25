"""
DR-RiskPredict v2 — Flask Web Application

Routes:
  GET/POST /login          -> login page
  GET      /logout         -> logout
  GET      /               -> home
  GET      /predict-form   -> screen patient form
  POST     /predict        -> run prediction + save
  GET      /result/<id>    -> result page
  GET      /dashboard      -> dashboard with charts, search, filter
  GET/POST /edit/<id>      -> edit patient record
  POST     /delete/<id>    -> delete patient record
  GET      /export/csv     -> export dashboard to CSV
  GET      /export/pdf     -> export dashboard to PDF
  GET      /result/<id>/print -> print-friendly result
  POST     /api/predict    -> JSON API (DR risk + multi-disease risk)
  POST     /api/predict-multi -> JSON API, multi-disease risk only
"""

from flask import (Flask, render_template, request, jsonify,
                   redirect, url_for, session, Response, make_response)
import joblib, json, os, csv, io
import numpy as np
from functools import wraps
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

import database as db
import disease_rules

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dr-riskpredict-secret-v2'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ---- Load model ----
try:
    best_model  = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    scaler      = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    imputer     = joblib.load(os.path.join(MODEL_DIR, "imputer.pkl"))
    encoders    = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
    with open(os.path.join(MODEL_DIR, "best_model_name.txt")) as f:
        best_model_name = f.read().strip()
    with open(os.path.join(MODEL_DIR, "feature_cols.json")) as f:
        feat_info = json.load(f)
    with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
        METRICS = json.load(f)
    MODEL_LOADED = True
except FileNotFoundError:
    MODEL_LOADED = False
    best_model_name = "N/A"
    feat_info = {"feature_cols": [], "numeric_cols": [], "categorical_cols": []}
    METRICS = None

FEATURE_COLS    = feat_info["feature_cols"]
NUMERIC_COLS    = feat_info["numeric_cols"]
CATEGORICAL_COLS = feat_info["categorical_cols"]

db.init_db()


# ---- Auth helpers ----
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login", next=request.path))
        return f(*args, **kwargs)
    return decorated


# ---- Validation & prediction ----
class ValidationError(Exception):
    pass


def validate_input(form, require_patient_name=True):
    errors = []
    required = ["age", "sex", "diabetes_type", "duration_years",
                "hba1c", "systolic_bp", "diastolic_bp", "smoking"]
    if require_patient_name:
        required = ["patient_name"] + required
    for field in required:
        if not form.get(field):
            errors.append(f"'{field}' is required.")

    def check_range(name, val, lo, hi):
        try:
            v = float(val)
            if not (lo <= v <= hi):
                errors.append(f"{name} must be between {lo} and {hi}.")
        except (TypeError, ValueError):
            errors.append(f"{name} must be a valid number.")

    checks = [
        ("Age", "age", 1, 120),
        ("Diabetes duration", "duration_years", 0, 100),
        ("HbA1c", "hba1c", 3, 20),
        ("Systolic BP", "systolic_bp", 60, 260),
        ("Diastolic BP", "diastolic_bp", 30, 160),
        ("Cholesterol", "cholesterol", 50, 500),
        ("Triglycerides", "triglycerides", 30, 1000),
        ("BMI", "bmi", 10, 70),
    ]
    for name, key, lo, hi in checks:
        if form.get(key):
            check_range(name, form.get(key), lo, hi)

    if form.get("diabetes_type") and form.get("diabetes_type") not in ("Type 1", "Type 2"):
        errors.append("Diabetes type must be 'Type 1' or 'Type 2'.")
    if form.get("sex") and form.get("sex") not in ("Male", "Female"):
        errors.append("Sex must be 'Male' or 'Female'.")
    if form.get("smoking") and form.get("smoking") not in ("Yes", "No"):
        errors.append("Smoking must be 'Yes' or 'No'.")
    if errors:
        raise ValidationError("; ".join(errors))


def run_prediction(data):
    import pandas as pd
    row = {
        "age": float(data["age"]),
        "sex": data["sex"],
        "diabetes_type": data["diabetes_type"],
        "duration_years": float(data["duration_years"]),
        "hba1c": float(data["hba1c"]),
        "systolic_bp": float(data["systolic_bp"]),
        "diastolic_bp": float(data["diastolic_bp"]),
        "cholesterol":   float(data["cholesterol"])   if data.get("cholesterol")   else np.nan,
        "triglycerides": float(data["triglycerides"]) if data.get("triglycerides") else np.nan,
        "bmi":           float(data["bmi"])           if data.get("bmi")           else np.nan,
        "smoking": data["smoking"],
    }
    X = pd.DataFrame([row])[FEATURE_COLS]
    X[NUMERIC_COLS] = imputer.transform(X[NUMERIC_COLS])
    for col in CATEGORICAL_COLS:
        X[col] = encoders[col].transform(X[col])
    X_m = X.copy()
    if best_model_name == "Logistic Regression":
        X_m[NUMERIC_COLS] = scaler.transform(X[NUMERIC_COLS])
    proba = best_model.predict_proba(X_m)[0][1]
    label = disease_rules.risk_label_from_proba(proba)
    return label, float(proba)


def run_multi_disease_prediction(data, dr_probability=None, dr_model_name=None):
    """
    Runs the modular multi-disease rule-based engine (disease_rules.py) on
    top of the same risk-factor inputs already collected for DR screening.
    Returns a list of disease-risk dicts sorted by risk score (desc).
    Never raises — failures degrade to an 'Unknown' entry so the primary
    DR prediction flow is never blocked by this secondary feature.
    """
    payload = dict(data)
    if dr_probability is not None:
        payload["dr_ml_probability"] = dr_probability
        payload["dr_ml_model_name"] = dr_model_name
    try:
        return disease_rules.predict_all_diseases(payload)
    except Exception as e:
        return [{
            "key": "error", "disease_name": "Multi-Disease Prediction",
            "risk_score": 0, "risk_level": "Unknown",
            "explanation": f"Multi-disease prediction is temporarily unavailable ({e}).",
            "contributing_factors": [], "prevention_tips": [],
            "lifestyle_recommendations": [], "diagnostic_tests": [], "source": "error",
        }]


# ================================================================
# AUTH ROUTES
# ================================================================
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        user = db.verify_user(request.form.get('username',''),
                              request.form.get('password',''))
        if user:
            session['user_id']   = user['id']
            session['username']  = user['username']
            session['role']      = user['role']
            next_url = request.args.get('next') or url_for('home')
            return redirect(next_url)
        error = "Invalid username or password."
    return render_template('login.html', error=error)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# ================================================================
# MAIN ROUTES
# ================================================================
@app.route('/')
@login_required
def home():
    stats = db.get_stats()
    return render_template('index.html', model_loaded=MODEL_LOADED,
                           best_model_name=best_model_name, stats=stats,
                           username=session.get('username'))


@app.route('/predict-form')
@login_required
def predict_form():
    return render_template('predict.html', error=None, form_data={})


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    form = request.form
    try:
        validate_input(form)
        label, proba = run_prediction(form)
        record = {
            "patient_name":    form.get("patient_name"),
            "age":             float(form.get("age")),
            "sex":             form.get("sex"),
            "diabetes_type":   form.get("diabetes_type"),
            "duration_years":  float(form.get("duration_years")),
            "hba1c":           float(form.get("hba1c")),
            "systolic_bp":     float(form.get("systolic_bp")),
            "diastolic_bp":    float(form.get("diastolic_bp")),
            "cholesterol":     float(form["cholesterol"])   if form.get("cholesterol")   else None,
            "triglycerides":   float(form["triglycerides"]) if form.get("triglycerides") else None,
            "bmi":             float(form["bmi"])           if form.get("bmi")           else None,
            "smoking":         form.get("smoking"),
            "risk_label":      label,
            "risk_probability": round(proba, 4),
            "model_used":      best_model_name,
        }
        record_id = db.insert_prediction(record)

        diseases = run_multi_disease_prediction(
            form, dr_probability=proba, dr_model_name=best_model_name)
        try:
            db.insert_disease_predictions(record_id, diseases)
        except Exception:
            pass  # audit-trail write is best-effort; never block the main result

        return redirect(url_for('result', record_id=record_id))
    except ValidationError as e:
        return render_template('predict.html', error=str(e), form_data=form), 400
    except Exception as e:
        return render_template('predict.html',
                               error=f"Prediction error: {e}", form_data=form), 500


@app.route('/result/<int:record_id>')
@login_required
def result(record_id):
    record = db.get_record_by_id(record_id)
    if not record:
        return render_template('error.html', code=404, message="Record not found."), 404
    diseases = run_multi_disease_prediction(
        record, dr_probability=record['risk_probability'], dr_model_name=record['model_used'])
    return render_template('result.html', record=record, record_id=record_id,
                           proba_pct=round(record['risk_probability'] * 100, 1),
                           diseases=diseases)


@app.route('/result/<int:record_id>/print')
@login_required
def result_print(record_id):
    record = db.get_record_by_id(record_id)
    if not record:
        return render_template('error.html', code=404, message="Record not found."), 404
    diseases = run_multi_disease_prediction(
        record, dr_probability=record['risk_probability'], dr_model_name=record['model_used'])
    return render_template('result_print.html', record=record, record_id=record_id,
                           proba_pct=round(record['risk_probability'] * 100, 1),
                           diseases=diseases)


# ================================================================
# DASHBOARD
# ================================================================
@app.route('/dashboard')
@login_required
def dashboard():
    search      = request.args.get('search', '').strip()
    risk_filter = request.args.get('risk', '')
    sort        = request.args.get('sort', 'id_desc')
    records     = db.get_all_records(search=search or None,
                                     risk_filter=risk_filter or None,
                                     sort=sort)
    stats       = db.get_stats()
    chart_data  = db.get_chart_data()
    return render_template('dashboard.html', records=records, stats=stats,
                           chart_data=json.dumps(chart_data),
                           search=search, risk_filter=risk_filter, sort=sort)


# ================================================================
# MODEL PERFORMANCE / ABOUT / HELP
# ================================================================
@app.route('/report')
@login_required
def report():
    if not MODEL_LOADED or METRICS is None:
        return render_template('error.html', code=503,
                               message="Model metrics are not available. Run train_model.py first."), 503
    return render_template('report.html',
                           metrics=METRICS["models"],
                           best_model_name=METRICS["best_model_name"],
                           dataset=METRICS["dataset"],
                           coefficients=METRICS["logistic_regression_coefficients"],
                           importances=METRICS["random_forest_importances"],
                           roc_data=json.dumps({
                               name: m["roc_curve"] for name, m in METRICS["models"].items()
                               if "roc_curve" in m
                           }))


@app.route('/about')
@login_required
def about():
    return render_template('about.html')


@app.route('/help')
@login_required
def help_desk():
    return render_template('help.html')


# ================================================================
# EDIT / DELETE
# ================================================================
@app.route('/edit/<int:record_id>', methods=['GET', 'POST'])
@login_required
def edit_record(record_id):
    record = db.get_record_by_id(record_id)
    if not record:
        return render_template('error.html', code=404, message="Record not found."), 404

    if request.method == 'POST':
        form = request.form
        try:
            validate_input(form)
            db.update_record(record_id, {
                "patient_name":  form.get("patient_name"),
                "age":           float(form.get("age")),
                "sex":           form.get("sex"),
                "diabetes_type": form.get("diabetes_type"),
                "duration_years": float(form.get("duration_years")),
                "hba1c":         float(form.get("hba1c")),
                "systolic_bp":   float(form.get("systolic_bp")),
                "diastolic_bp":  float(form.get("diastolic_bp")),
                "cholesterol":   float(form["cholesterol"])   if form.get("cholesterol")   else None,
                "triglycerides": float(form["triglycerides"]) if form.get("triglycerides") else None,
                "bmi":           float(form["bmi"])           if form.get("bmi")           else None,
                "smoking":       form.get("smoking"),
            })
            return redirect(url_for('dashboard'))
        except ValidationError as e:
            return render_template('edit.html', record=record, error=str(e)), 400

    return render_template('edit.html', record=record, error=None)


@app.route('/delete/<int:record_id>', methods=['POST'])
@login_required
def delete_record(record_id):
    db.delete_record(record_id)
    return redirect(url_for('dashboard'))


# ================================================================
# EXPORT
# ================================================================
@app.route('/export/csv')
@login_required
def export_csv():
    search      = request.args.get('search', '').strip()
    risk_filter = request.args.get('risk', '')
    records     = db.get_all_records(search=search or None,
                                     risk_filter=risk_filter or None)
    si = io.StringIO()
    writer = csv.writer(si)
    writer.writerow(["ID","Patient Name","Age","Sex","Diabetes Type","Duration (yrs)",
                     "HbA1c (%)","Systolic BP","Diastolic BP","Cholesterol",
                     "Triglycerides","BMI","Smoking","Risk Label",
                     "Risk Probability (%)","Model Used","Date"])
    for r in records:
        writer.writerow([
            r['id'], r['patient_name'], r['age'], r['sex'], r['diabetes_type'],
            r['duration_years'], r['hba1c'], r['systolic_bp'], r['diastolic_bp'],
            r['cholesterol'] or '', r['triglycerides'] or '', r['bmi'] or '',
            r['smoking'], r['risk_label'],
            round(r['risk_probability']*100, 1),
            r['model_used'], r['created_at']
        ])
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = \
        f"attachment; filename=dr_riskpredict_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    output.headers["Content-type"] = "text/csv"
    return output


@app.route('/export/pdf')
@login_required
def export_pdf():
    search      = request.args.get('search', '').strip()
    risk_filter = request.args.get('risk', '')
    records     = db.get_all_records(search=search or None,
                                     risk_filter=risk_filter or None)
    stats       = db.get_stats()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=36, rightMargin=36,
                            topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    story  = []

    # Title
    title_style = styles['Title']
    story.append(Paragraph("DR-RiskPredict — Patient Screening Report", title_style))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')} &nbsp;|&nbsp; "
        f"Total: {stats['total']} &nbsp;|&nbsp; "
        f"High Risk: {stats['high_risk']} &nbsp;|&nbsp; "
        f"Moderate Risk: {stats['moderate_risk']} &nbsp;|&nbsp; "
        f"Low Risk: {stats['low_risk']}",
        styles['Normal']
    ))
    story.append(Spacer(1, 16))

    # Table
    headers = ["ID","Patient","Age","Type","HbA1c","BP","Risk","Prob%","Date"]
    data = [headers]
    for r in records:
        data.append([
            str(r['id']), r['patient_name'], str(int(r['age'])),
            r['diabetes_type'], f"{r['hba1c']}%",
            f"{int(r['systolic_bp'])}/{int(r['diastolic_bp'])}",
            r['risk_label'],
            f"{round(r['risk_probability']*100,1)}%",
            r['created_at'][:10]
        ])

    tbl = Table(data, repeatRows=1)
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#16302B')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTSIZE',   (0,0), (-1,0), 8),
        ('FONTSIZE',   (0,1), (-1,-1), 7),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F5F3EE')]),
        ('GRID',       (0,0), (-1,-1), 0.4, colors.HexColor('#D8D3C4')),
        ('PADDING',    (0,0), (-1,-1), 4),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
    ]))

    # Color high/moderate-risk rows
    for i, r in enumerate(records, start=1):
        if r['risk_label'] == 'High Risk':
            tbl.setStyle(TableStyle([
                ('BACKGROUND', (0,i), (-1,i), colors.HexColor('#FBEAE6')),
                ('TEXTCOLOR',  (6,i), (6,i),  colors.HexColor('#A1402E')),
            ]))
        elif r['risk_label'] == 'Moderate Risk':
            tbl.setStyle(TableStyle([
                ('BACKGROUND', (0,i), (-1,i), colors.HexColor('#FBF3E1')),
                ('TEXTCOLOR',  (6,i), (6,i),  colors.HexColor('#A9701E')),
            ]))

    story.append(tbl)
    doc.build(story)
    buf.seek(0)

    response = make_response(buf.read())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = \
        f"attachment; filename=dr_riskpredict_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    return response


# ================================================================
# JSON API
# ================================================================
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400
        validate_input(data, require_patient_name=False)
        label, proba = run_prediction(data)
        diseases = run_multi_disease_prediction(
            data, dr_probability=proba, dr_model_name=best_model_name)
        return jsonify({"risk_label": label,
                        "risk_probability": round(proba, 4),
                        "model_used": best_model_name,
                        "diseases": diseases})
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/predict-multi', methods=['POST'])
def api_predict_multi():
    """
    Standalone JSON API for the multi-disease risk engine only (no DR model
    dependency required, no DB write). Same input contract as /api/predict.
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400
        validate_input(data, require_patient_name=False)
        diseases = run_multi_disease_prediction(data)
        return jsonify({"diseases": diseases})
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================================================================
# ERROR HANDLERS
# ================================================================
@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', code=404, message="Page not found."), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', code=500, message="Internal server error."), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
