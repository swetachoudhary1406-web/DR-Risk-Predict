"""
DR-RiskPredict — Week 4
Generate model performance report: confusion matrix + ROC curve charts
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "clinical_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
# NOTE: must be under static/ — Flask's url_for('static', ...) (used by
# templates/report.html's optional downloadable-image links) only serves
# files from here, not from a top-level /reports folder.
REPORT_DIR = os.path.join(BASE_DIR, "static", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# Reload everything (mirrors train_model.py split so test set matches)
df = pd.read_csv(DATA_PATH)
FEATURE_COLS = ["age", "sex", "diabetes_type", "duration_years", "hba1c",
                 "systolic_bp", "diastolic_bp", "cholesterol",
                 "triglycerides", "bmi", "smoking"]
numeric_cols = ["age", "duration_years", "hba1c", "systolic_bp",
                 "diastolic_bp", "cholesterol", "triglycerides", "bmi"]
categorical_cols = ["sex", "diabetes_type", "smoking"]

imputer = joblib.load(os.path.join(MODEL_DIR, "imputer.pkl"))
encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
log_reg = joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl"))
rf = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
with open(os.path.join(MODEL_DIR, "best_model_name.txt")) as f:
    best_model_name = f.read().strip()

df_model = df[FEATURE_COLS + ["dr_status"]].copy()
df_model[numeric_cols] = imputer.transform(df_model[numeric_cols])
for col in categorical_cols:
    df_model[col] = encoders[col].transform(df_model[col])

X = df_model[FEATURE_COLS]
y = df_model["dr_status"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_test_scaled = X_test.copy()
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

models = {
    "Logistic Regression": (log_reg, X_test_scaled),
    "Random Forest": (rf, X_test),
}

# ---------------------------------------------------------
# ROC Curve (both models on one chart)
# ---------------------------------------------------------
plt.figure(figsize=(7, 6))
colors = {"Logistic Regression": "#0F5460", "Random Forest": "#C0392B"}
for name, (model, X_t) in models.items():
    y_proba = model.predict_proba(X_t)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    style = "-" if name == best_model_name else "--"
    plt.plot(fpr, tpr, style, color=colors[name], linewidth=2,
              label=f"{name} (AUC = {roc_auc:.3f})" + (" [selected]" if name == best_model_name else ""))

plt.plot([0, 1], [0, 1], "k:", linewidth=1, label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("ROC Curve — DR Risk Prediction Models")
plt.legend(loc="lower right", fontsize=9)
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "roc_curve.png"), dpi=150)
plt.close()
print("Saved roc_curve.png")

# ---------------------------------------------------------
# Confusion matrices (side by side)
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, (name, (model, X_t)) in zip(axes, models.items()):
    y_pred = model.predict(X_t)
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["No DR", "DR"],
        cmap="Greens" if name == best_model_name else "Oranges", ax=ax, colorbar=False
    )
    title = name + (" [selected]" if name == best_model_name else "")
    ax.set_title(title, fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "confusion_matrices.png"), dpi=150)
plt.close()
print("Saved confusion_matrices.png")

print(f"\nReport assets saved to: {REPORT_DIR}")
