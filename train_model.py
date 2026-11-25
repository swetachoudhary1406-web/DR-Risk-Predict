"""
DR-RiskPredict — Week 2
- Clean data: handle missing values, encode categorical variables
- Split data: train (70%) / test (30%)
- Train models: Logistic Regression, Random Forest, compare accuracy
- Select best model based on accuracy + sensitivity
"""

import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "clinical_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} records, {df.shape[1]} columns")

FEATURE_COLS = [
    "age", "sex", "diabetes_type", "duration_years", "hba1c",
    "systolic_bp", "diastolic_bp", "cholesterol",
    "triglycerides", "bmi", "smoking"
]
TARGET_COL = "dr_status"

df_model = df[FEATURE_COLS + [TARGET_COL]].copy()

# ---------------------------------------------------------
# 2. Clean data: handle missing values
# ---------------------------------------------------------
numeric_cols = ["age", "duration_years", "hba1c", "systolic_bp",
                 "diastolic_bp", "cholesterol", "triglycerides", "bmi"]
categorical_cols = ["sex", "diabetes_type", "smoking"]

imputer = SimpleImputer(strategy="median")
df_model[numeric_cols] = imputer.fit_transform(df_model[numeric_cols])
print(f"Missing values after imputation: {df_model[numeric_cols].isnull().sum().sum()}")

# Encode categorical variables
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    encoders[col] = le
    print(f"Encoded '{col}': {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ---------------------------------------------------------
# 3. Train/test split (70/30)
# ---------------------------------------------------------
X = df_model[FEATURE_COLS]
y = df_model[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
print(f"\nTrain set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")

# Scale numeric features (helps Logistic Regression)
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

# ---------------------------------------------------------
# 4. Train models
# ---------------------------------------------------------
results = {}

# --- Logistic Regression ---
# Cross-validated regularization search: with 10 features and a modest
# training set, an untuned default C can let weak/noisy features (e.g. age,
# BMI) land on unstable — sometimes backwards — coefficients. 5-fold CV
# picks the C that generalizes best on held-out folds instead.
lr_grid = GridSearchCV(
    LogisticRegression(max_iter=2000, random_state=42),
    param_grid={"C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]},
    scoring="roc_auc", cv=5
)
lr_grid.fit(X_train_scaled, y_train)
log_reg = lr_grid.best_estimator_
print(f"Logistic Regression: best C = {lr_grid.best_params_['C']} "
      f"(CV ROC-AUC = {lr_grid.best_score_:.3f})")
y_pred_lr = log_reg.predict(X_test_scaled)
y_proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
results["Logistic Regression"] = {
    "accuracy": accuracy_score(y_test, y_pred_lr),
    "precision": precision_score(y_test, y_pred_lr),
    "recall_sensitivity": recall_score(y_test, y_pred_lr),
    "f1_score": f1_score(y_test, y_pred_lr),
    "roc_auc": roc_auc_score(y_test, y_proba_lr),
    "confusion_matrix": confusion_matrix(y_test, y_pred_lr).tolist(),
    "roc_curve": {"fpr": fpr_lr.tolist(), "tpr": tpr_lr.tolist()},
    "best_C": lr_grid.best_params_["C"],
}

# --- Random Forest ---
rf = RandomForestClassifier(
    n_estimators=200, max_depth=6, min_samples_leaf=4,
    random_state=42, class_weight="balanced"
)
rf.fit(X_train, y_train)  # tree models don't need scaling
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
results["Random Forest"] = {
    "accuracy": accuracy_score(y_test, y_pred_rf),
    "precision": precision_score(y_test, y_pred_rf),
    "recall_sensitivity": recall_score(y_test, y_pred_rf),
    "f1_score": f1_score(y_test, y_pred_rf),
    "roc_auc": roc_auc_score(y_test, y_proba_rf),
    "confusion_matrix": confusion_matrix(y_test, y_pred_rf).tolist(),
    "roc_curve": {"fpr": fpr_rf.tolist(), "tpr": tpr_rf.tolist()},
}

# ---------------------------------------------------------
# 5. Compare & select best model (accuracy + sensitivity)
# ---------------------------------------------------------
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)
for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"  Accuracy:    {metrics['accuracy']:.3f}")
    print(f"  Precision:   {metrics['precision']:.3f}")
    print(f"  Sensitivity: {metrics['recall_sensitivity']:.3f}")
    print(f"  F1-score:    {metrics['f1_score']:.3f}")
    print(f"  ROC-AUC:     {metrics['roc_auc']:.3f}")
    print(f"  Confusion Matrix: {metrics['confusion_matrix']}")

# Combined score: weight accuracy and sensitivity equally
# (sensitivity matters clinically — missing a DR case is costly)
for name, metrics in results.items():
    metrics["combined_score"] = 0.5 * metrics["accuracy"] + 0.5 * metrics["recall_sensitivity"]

best_model_name = max(results, key=lambda k: results[k]["combined_score"])
print(f"\n{'='*60}")
print(f"BEST MODEL: {best_model_name}")
print(f"{'='*60}")

# ---------------------------------------------------------
# 6. Save artifacts
# ---------------------------------------------------------
joblib.dump(log_reg, os.path.join(MODEL_DIR, "logistic_regression.pkl"))
joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(imputer, os.path.join(MODEL_DIR, "imputer.pkl"))
joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))

best_model = rf if best_model_name == "Random Forest" else log_reg
joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))

with open(os.path.join(MODEL_DIR, "best_model_name.txt"), "w") as f:
    f.write(best_model_name)

with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump({
        "models": results,
        "best_model_name": best_model_name,
        "feature_cols": FEATURE_COLS,
        "logistic_regression_coefficients": dict(zip(FEATURE_COLS, log_reg.coef_[0].tolist())),
        "random_forest_importances": dict(zip(FEATURE_COLS, rf.feature_importances_.tolist())),
        "dataset": {
            "total_rows": int(len(df)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "positive_rate": float(y.mean()),
        },
    }, f, indent=2)

with open(os.path.join(MODEL_DIR, "feature_cols.json"), "w") as f:
    json.dump({
        "feature_cols": FEATURE_COLS,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols
    }, f, indent=2)

print(f"\nAll model artifacts saved to: {MODEL_DIR}")
