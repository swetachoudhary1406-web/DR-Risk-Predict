"""
DR-RiskPredict — Multi-Disease Risk Prediction Engine
======================================================

Extends the existing diabetic-retinopathy screening flow into a broader,
multi-disease risk assessment, reusing the exact same clinical inputs
already captured on the screening form (age, diabetes type, duration,
HbA1c, blood pressure, cholesterol, triglycerides, BMI, smoking) plus the
diabetic-retinopathy probability already produced by the trained ML model.

Design
------
This module is intentionally **rule-based**, not ML-based, because no
labeled training data exists yet for heart disease / kidney disease /
hypertension / stroke in this project. It is built to the same "contract"
a future ML model would use, so any predictor here can be swapped for a
trained model later with no change to the caller (`app.py`):

    def predict_xxx(inputs: PatientInputs) -> DiseaseRisk

Every predictor:
  * accepts the same standardized `PatientInputs` dataclass
  * returns the same standardized `DiseaseRisk` dataclass
  * is registered in `DISEASE_REGISTRY` under a stable key

`predict_all_diseases()` runs the full registry, sorts results by risk
score (highest first), and isolates failures so one broken predictor
cannot take down the others or the request.

Risk score scale: 0-100 (%). Risk level bands:
  Low        0  - 29
  Moderate   30 - 59
  High       60 - 100
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


# ============================================================
# Standardized input / output contracts
# ============================================================

@dataclass
class PatientInputs:
    """Normalized, typed view of the risk factors already on the form."""
    age: float
    sex: str                      # "Male" | "Female"
    diabetes_type: str            # "Type 1" | "Type 2"
    duration_years: float
    hba1c: float
    systolic_bp: float
    diastolic_bp: float
    smoking: str                  # "Yes" | "No"
    cholesterol: Optional[float] = None
    triglycerides: Optional[float] = None
    bmi: Optional[float] = None
    # Optional pre-computed ML output, reused by the retinopathy predictor
    dr_ml_probability: Optional[float] = None
    dr_ml_model_name: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "PatientInputs":
        def _f(key):
            v = data.get(key)
            if v in (None, ""):
                return None
            return float(v)

        return cls(
            age=float(data["age"]),
            sex=str(data.get("sex", "Female")),
            diabetes_type=str(data["diabetes_type"]),
            duration_years=float(data["duration_years"]),
            hba1c=float(data["hba1c"]),
            systolic_bp=float(data["systolic_bp"]),
            diastolic_bp=float(data["diastolic_bp"]),
            smoking=str(data["smoking"]),
            cholesterol=_f("cholesterol"),
            triglycerides=_f("triglycerides"),
            bmi=_f("bmi"),
            dr_ml_probability=_f("dr_ml_probability"),
            dr_ml_model_name=data.get("dr_ml_model_name"),
        )


@dataclass
class DiagnosticTest:
    name: str
    purpose: str
    why_recommended: str
    essential: bool


@dataclass
class DiseaseRisk:
    key: str                       # stable id, e.g. "heart_disease"
    disease_name: str
    risk_score: float               # 0-100
    risk_level: str                  # Low | Moderate | High
    explanation: str
    contributing_factors: list = field(default_factory=list)
    prevention_tips: list = field(default_factory=list)
    lifestyle_recommendations: list = field(default_factory=list)
    diagnostic_tests: list = field(default_factory=list)  # list[DiagnosticTest]
    source: str = "rule_based"       # "rule_based" | "ml_model"
    flags: list = field(default_factory=list)  # visible caution notes, e.g. model/rule disagreement

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "disease_name": self.disease_name,
            "risk_score": round(self.risk_score, 1),
            "risk_level": self.risk_level,
            "explanation": self.explanation,
            "contributing_factors": self.contributing_factors,
            "prevention_tips": self.prevention_tips,
            "lifestyle_recommendations": self.lifestyle_recommendations,
            "diagnostic_tests": [t.__dict__ for t in self.diagnostic_tests],
            "source": self.source,
            "flags": self.flags,
        }


def _level(score: float) -> str:
    if score >= 60:
        return "High"
    if score >= 30:
        return "Moderate"
    return "Low"


def risk_label_from_proba(proba: float) -> str:
    """
    Single source of truth for risk banding, shared by the DR ML model
    (app.py's run_prediction) and every rule-based predictor below, so a
    given score always reads the same way everywhere in the app.
    Takes a 0-1 probability; returns a display label e.g. "High Risk".
    """
    return f"{_level(_clamp(proba * 100))} Risk"


def _clamp(score: float) -> float:
    return max(0.0, min(100.0, score))


# ============================================================
# Shared lifestyle tip pools
# ============================================================

_GENERAL_DIABETES_LIFESTYLE = [
    "Aim for at least 150 minutes/week of moderate aerobic activity (e.g. brisk walking).",
    "Follow a consistent, carbohydrate-aware meal pattern; consult a dietitian if unsure.",
    "Monitor blood glucose as advised and keep a log to share with your care team.",
    "Prioritize 7-9 hours of sleep; poor sleep worsens insulin resistance.",
]


def _smoking_prefix(inputs: PatientInputs, tips: list) -> list:
    """Prepend a personalized smoking-cessation tip when relevant."""
    if inputs.smoking == "Yes":
        return ["Quitting smoking is the single most impactful step you can take "
                "to lower this risk — ask your provider about cessation support."] + tips
    return tips


# ============================================================
# 1. Diabetes — Glycemic Control / Progression Risk
# ============================================================

def predict_diabetes_glycemic_risk(inputs: PatientInputs) -> DiseaseRisk:
    score = 0.0
    factors = []

    if inputs.hba1c >= 10:
        score += 60; factors.append(f"HbA1c very high ({inputs.hba1c}%)")
    elif inputs.hba1c >= 9:
        score += 45; factors.append(f"HbA1c high ({inputs.hba1c}%)")
    elif inputs.hba1c >= 8:
        score += 30; factors.append(f"HbA1c elevated ({inputs.hba1c}%)")
    elif inputs.hba1c >= 7:
        score += 15; factors.append(f"HbA1c above target ({inputs.hba1c}%)")

    if inputs.bmi is not None:
        if inputs.bmi >= 35:
            score += 20; factors.append(f"BMI in obese class II+ range ({inputs.bmi})")
        elif inputs.bmi >= 30:
            score += 12; factors.append(f"BMI in obese range ({inputs.bmi})")
        elif inputs.bmi >= 25:
            score += 6; factors.append(f"BMI overweight ({inputs.bmi})")

    if inputs.duration_years >= 15:
        score += 15; factors.append(f"Long diabetes duration ({inputs.duration_years} yrs)")
    elif inputs.duration_years >= 10:
        score += 10; factors.append(f"Diabetes duration {inputs.duration_years} yrs")
    elif inputs.duration_years >= 5:
        score += 5

    if inputs.age >= 45:
        score += 5; factors.append("Age 45 or older")

    score = _clamp(score)
    tests = [
        DiagnosticTest("HbA1c", "Measures average blood glucose over ~3 months.",
                        "Primary marker of long-term glycemic control and diabetes progression.", True),
        DiagnosticTest("Fasting Blood Sugar (FBS)", "Measures glucose after an 8-hour fast.",
                        "Confirms current glycemic status and day-to-day control.", True),
        DiagnosticTest("Random Blood Sugar (RBS)", "Spot-check glucose at any time of day.",
                        "Useful for quick monitoring between formal lab visits.", False),
        DiagnosticTest("Oral Glucose Tolerance Test (OGTT)", "Measures glucose response to a sugar load.",
                        "Helps clarify borderline or fluctuating control, especially if HbA1c is inconsistent with symptoms.", False),
    ]
    tips = [
        "Keep HbA1c below your individualized target (commonly <7%, discuss with your clinician).",
        "Take medications/insulin exactly as prescribed; don't skip doses.",
        "Review your care plan with your endocrinologist at least every 3-6 months.",
    ]
    lifestyle = _smoking_prefix(inputs, list(_GENERAL_DIABETES_LIFESTYLE))
    return DiseaseRisk(
        key="diabetes_glycemic",
        disease_name="Diabetes — Glycemic Control Risk",
        risk_score=score,
        risk_level=_level(score),
        explanation="Estimates the risk of poor ongoing glycemic control and disease progression, "
                    "based on HbA1c, BMI, disease duration, and age.",
        contributing_factors=factors or ["HbA1c, BMI, and diabetes duration are all currently in favorable ranges."],
        prevention_tips=tips,
        lifestyle_recommendations=lifestyle,
        diagnostic_tests=tests,
    )


# ============================================================
# 2. Diabetic Retinopathy — wraps the existing trained ML model
# ============================================================

def _retinopathy_heuristic_score(inputs: PatientInputs) -> float:
    """
    A transparent, clinical-threshold estimate of DR risk using the same
    full set of inputs the ML model sees. Used as (a) the fallback score
    if the ML model is unavailable, and (b) an always-on sanity check
    against the ML model's score, so a bad/overfit model prediction on an
    unusual input combination doesn't silently go unnoticed. Deliberately
    comprehensive (not just 3-4 factors) so it's a fair comparison rather
    than a crude strawman that disagrees with the model constantly.
    """
    score = 0.0
    score += max(0, (inputs.hba1c - 7) * 6.5)
    score += max(0, (inputs.duration_years - 5) * 1.7)
    score += max(0, (inputs.systolic_bp - 125) * 0.35)
    score += max(0, (inputs.diastolic_bp - 80) * 0.25)
    if inputs.cholesterol is not None:
        score += max(0, (inputs.cholesterol - 200) * 0.04)
    if inputs.triglycerides is not None:
        score += max(0, (inputs.triglycerides - 150) * 0.03)
    if inputs.bmi is not None:
        score += max(0, (inputs.bmi - 27) * 0.6)
    score += max(0, (inputs.age - 45) * 0.25)
    if inputs.smoking == "Yes":
        score += 10
    if inputs.diabetes_type == "Type 1":
        score += 8
    return _clamp(score)


def predict_retinopathy_risk(inputs: PatientInputs) -> DiseaseRisk:
    factors = []
    if inputs.hba1c >= 9:
        factors.append(f"HbA1c high ({inputs.hba1c}%)")
    if inputs.duration_years >= 10:
        factors.append(f"Diabetes duration {inputs.duration_years} yrs")
    if inputs.systolic_bp >= 140 or inputs.diastolic_bp >= 90:
        factors.append(f"Elevated blood pressure ({int(inputs.systolic_bp)}/{int(inputs.diastolic_bp)} mmHg)")
    if inputs.smoking == "Yes":
        factors.append("Current smoker")

    flags = []
    heuristic_score = _retinopathy_heuristic_score(inputs)

    if inputs.dr_ml_probability is not None:
        score = _clamp(inputs.dr_ml_probability * 100)
        source = "ml_model"
        explanation = (f"Predicted by the trained {inputs.dr_ml_model_name or 'ML'} model using the full "
                        "clinical feature set (age, diabetes type, duration, HbA1c, blood pressure, "
                        "lipids, BMI, smoking).")
        # Sanity check: the ML model is fit to a limited dataset and can, in
        # principle, land on an unstable prediction for an unusual input
        # combination. A one-tier disagreement (e.g. Moderate vs High) is
        # normal variation between two different modeling approaches and
        # not meaningful on its own — only a severe two-tier jump (Low vs
        # High) is actually alarming enough to surface.
        tier_index = {"Low": 0, "Moderate": 1, "High": 2}
        heuristic_level = _level(heuristic_score)
        ml_level = _level(score)
        if abs(tier_index[ml_level] - tier_index[heuristic_level]) >= 2:
            flags.append(
                f"This ML score ({score:.0f}%, {ml_level} Risk) sharply disagrees with a comprehensive "
                f"clinical threshold estimate ({heuristic_score:.0f}%, {heuristic_level} Risk) for the "
                "same values. Treat this prediction with extra caution and prioritize the diagnostic "
                "tests below."
            )
    else:
        # Fallback heuristic if the ML model failed to load — keeps the
        # feature degrading gracefully instead of disappearing.
        score = heuristic_score
        source = "rule_based"
        explanation = ("ML model unavailable — estimated using a simplified rule-based fallback on "
                        "HbA1c, duration, blood pressure, and smoking status.")

    tests = [
        DiagnosticTest("Dilated Fundus Examination", "Direct visual inspection of the retina.",
                        "Gold-standard screening test for detecting retinopathy changes.", True),
        DiagnosticTest("Optical Coherence Tomography (OCT)", "Cross-sectional imaging of retinal layers.",
                        "Detects macular edema and subtle structural changes not visible on exam alone.", False),
        DiagnosticTest("Fundus Photography", "Digital imaging of the retina for documentation.",
                        "Enables tracking of progression over time and remote/tele-screening.", False),
        DiagnosticTest("Visual Acuity Test", "Measures sharpness of vision.",
                        "Baseline functional check and quick indicator of vision-threatening disease.", True),
    ]
    tips = [
        "Get a dilated eye exam at least annually, or as advised by your ophthalmologist.",
        "Keep blood pressure and HbA1c within target — both directly slow retinopathy progression.",
        "Report any new floaters, blurred vision, or vision loss immediately.",
    ]
    lifestyle = _smoking_prefix(inputs, list(_GENERAL_DIABETES_LIFESTYLE))
    return DiseaseRisk(
        key="diabetic_retinopathy",
        disease_name="Diabetic Retinopathy",
        risk_score=score,
        risk_level=_level(score),
        explanation=explanation,
        contributing_factors=factors or ["No retinopathy-associated risk factors (glycemic control, blood pressure, smoking) are currently elevated."],
        prevention_tips=tips,
        lifestyle_recommendations=lifestyle,
        diagnostic_tests=tests,
        source=source,
        flags=flags,
    )


# ============================================================
# 3. Heart Disease (Cardiovascular Disease)
# ============================================================

def predict_heart_disease_risk(inputs: PatientInputs) -> DiseaseRisk:
    score = 0.0
    factors = []

    if inputs.systolic_bp >= 160:
        score += 35; factors.append(f"Systolic BP very high ({int(inputs.systolic_bp)} mmHg)")
    elif inputs.systolic_bp >= 140:
        score += 25; factors.append(f"Systolic BP high ({int(inputs.systolic_bp)} mmHg)")
    elif inputs.systolic_bp >= 130:
        score += 15; factors.append(f"Systolic BP elevated ({int(inputs.systolic_bp)} mmHg)")
    elif inputs.systolic_bp >= 120:
        score += 5

    if inputs.cholesterol is not None:
        if inputs.cholesterol >= 240:
            score += 20; factors.append(f"Total cholesterol high ({inputs.cholesterol} mg/dL)")
        elif inputs.cholesterol >= 200:
            score += 10; factors.append(f"Total cholesterol borderline ({inputs.cholesterol} mg/dL)")

    if inputs.triglycerides is not None:
        if inputs.triglycerides >= 500:
            score += 18; factors.append(f"Triglycerides very high ({inputs.triglycerides} mg/dL)")
        elif inputs.triglycerides >= 200:
            score += 12; factors.append(f"Triglycerides high ({inputs.triglycerides} mg/dL)")
        elif inputs.triglycerides >= 150:
            score += 5

    if inputs.smoking == "Yes":
        score += 20; factors.append("Current smoker")

    if inputs.duration_years >= 10:
        score += 10; factors.append(f"Diabetes duration {inputs.duration_years} yrs")

    if inputs.age >= 55:
        score += 10; factors.append("Age 55 or older")

    if inputs.bmi is not None and inputs.bmi >= 30:
        score += 8; factors.append(f"BMI in obese range ({inputs.bmi})")

    if inputs.sex == "Male":
        if inputs.age < 55:
            score += 8; factors.append("Male sex (higher CVD risk at this age)")
        else:
            score += 4  # gap narrows with age; not listed as a standout factor

    score = _clamp(score)
    tests = [
        DiagnosticTest("ECG (Electrocardiogram)", "Records the heart's electrical activity.",
                        "First-line test to detect arrhythmias, ischemia, or prior silent infarcts.", True),
        DiagnosticTest("Lipid Profile", "Measures total cholesterol, LDL, HDL, and triglycerides.",
                        "Directly quantifies atherosclerotic risk and guides statin therapy decisions.", True),
        DiagnosticTest("Echocardiogram", "Ultrasound imaging of heart structure and function.",
                        "Assesses ejection fraction and valve/wall motion abnormalities.", False),
        DiagnosticTest("Troponin Test", "Blood marker of heart muscle injury.",
                        "Recommended if chest pain or acute symptoms are present.", False),
        DiagnosticTest("Blood Pressure Monitoring", "Serial or ambulatory BP tracking.",
                        "Hypertension is the single largest modifiable driver of cardiovascular risk.", True),
    ]
    tips = [
        "Keep LDL cholesterol and blood pressure within targets set by your physician.",
        "Limit saturated fat, trans fat, and sodium intake.",
        "Discuss aspirin or statin therapy eligibility with your doctor if risk is elevated.",
    ]
    lifestyle = _smoking_prefix(inputs, [
        "Aim for 150+ minutes/week of moderate cardio exercise (walking, cycling, swimming).",
        "Adopt a heart-healthy diet (e.g. Mediterranean or DASH pattern).",
        "Limit alcohol intake and manage stress with relaxation techniques.",
        "Maintain a healthy waist circumference and BMI.",
    ])
    return DiseaseRisk(
        key="heart_disease",
        disease_name="Heart Disease (Cardiovascular Disease)",
        risk_score=score,
        risk_level=_level(score),
        explanation="Estimates cardiovascular risk from blood pressure, lipid values, smoking status, "
                    "diabetes duration, age, and BMI.",
        contributing_factors=factors or ["Blood pressure, lipids, and smoking status are all currently within healthy ranges."],
        prevention_tips=tips,
        lifestyle_recommendations=lifestyle,
        diagnostic_tests=tests,
    )


# ============================================================
# 4. Chronic Kidney Disease (Diabetic Nephropathy risk)
# ============================================================

def predict_kidney_disease_risk(inputs: PatientInputs) -> DiseaseRisk:
    score = 0.0
    factors = []

    if inputs.duration_years >= 20:
        score += 35; factors.append(f"Very long diabetes duration ({inputs.duration_years} yrs)")
    elif inputs.duration_years >= 10:
        score += 25; factors.append(f"Diabetes duration {inputs.duration_years} yrs")
    elif inputs.duration_years >= 5:
        score += 10

    if inputs.hba1c >= 9:
        score += 20; factors.append(f"HbA1c high ({inputs.hba1c}%)")
    elif inputs.hba1c >= 8:
        score += 12; factors.append(f"HbA1c elevated ({inputs.hba1c}%)")
    elif inputs.hba1c >= 7:
        score += 5

    if inputs.systolic_bp >= 140:
        score += 20; factors.append(f"Systolic BP high ({int(inputs.systolic_bp)} mmHg)")
    elif inputs.systolic_bp >= 130:
        score += 10; factors.append(f"Systolic BP elevated ({int(inputs.systolic_bp)} mmHg)")

    if inputs.age >= 60:
        score += 10; factors.append("Age 60 or older")

    if inputs.diabetes_type == "Type 1" and inputs.duration_years >= 15:
        score += 5; factors.append("Long-standing Type 1 diabetes")

    score = _clamp(score)
    tests = [
        DiagnosticTest("Serum Creatinine", "Blood marker used to estimate kidney filtration.",
                        "Baseline marker of kidney function and trend tracking.", True),
        DiagnosticTest("eGFR (estimated Glomerular Filtration Rate)", "Calculated from creatinine, age, and sex.",
                        "Standard staging measure for chronic kidney disease.", True),
        DiagnosticTest("Urine Albumin-to-Creatinine Ratio (ACR) / Urine Protein Test",
                        "Detects protein leakage into urine.",
                        "Microalbuminuria is often the earliest detectable sign of diabetic nephropathy.", True),
        DiagnosticTest("Kidney Ultrasound", "Imaging of kidney size and structure.",
                        "Rules out structural causes and obstruction when function is abnormal.", False),
    ]
    tips = [
        "Keep blood pressure below 130/80 mmHg or your clinician's target.",
        "Have urine albumin/creatinine ratio checked at least annually.",
        "Avoid NSAIDs and other nephrotoxic medications where possible.",
    ]
    lifestyle = _smoking_prefix(inputs, [
        "Moderate protein intake as advised by a renal dietitian if kidney function is reduced.",
        "Limit sodium intake to help control blood pressure.",
        "Stay well hydrated, but follow fluid guidance if kidney function is already impaired.",
        "Keep blood glucose in target range to slow nephropathy progression.",
    ])
    return DiseaseRisk(
        key="kidney_disease",
        disease_name="Chronic Kidney Disease (Diabetic Nephropathy)",
        risk_score=score,
        risk_level=_level(score),
        explanation="Estimates kidney disease risk from diabetes duration, HbA1c, blood pressure, and age — "
                    "the strongest known drivers of diabetic nephropathy.",
        contributing_factors=factors or ["Diabetes duration, HbA1c, and blood pressure are not yet in ranges associated with nephropathy."],
        prevention_tips=tips,
        lifestyle_recommendations=lifestyle,
        diagnostic_tests=tests,
    )


# ============================================================
# 5. Hypertension
# ============================================================

def predict_hypertension_risk(inputs: PatientInputs) -> DiseaseRisk:
    score = 0.0
    factors = []

    if inputs.systolic_bp >= 160:
        score += 60; factors.append(f"Systolic BP stage 2+ ({int(inputs.systolic_bp)} mmHg)")
    elif inputs.systolic_bp >= 140:
        score += 45; factors.append(f"Systolic BP stage 2 ({int(inputs.systolic_bp)} mmHg)")
    elif inputs.systolic_bp >= 130:
        score += 25; factors.append(f"Systolic BP stage 1 ({int(inputs.systolic_bp)} mmHg)")
    elif inputs.systolic_bp >= 120:
        score += 10; factors.append(f"Systolic BP elevated ({int(inputs.systolic_bp)} mmHg)")

    if inputs.diastolic_bp >= 90:
        score += 15; factors.append(f"Diastolic BP high ({int(inputs.diastolic_bp)} mmHg)")
    elif inputs.diastolic_bp >= 80:
        score += 8

    if inputs.bmi is not None and inputs.bmi >= 30:
        score += 10; factors.append(f"BMI in obese range ({inputs.bmi})")

    if inputs.smoking == "Yes":
        score += 5; factors.append("Current smoker")

    if inputs.age >= 50:
        score += 5

    score = _clamp(score)
    tests = [
        DiagnosticTest("Blood Pressure Monitoring", "Repeated clinic or home BP readings.",
                        "Required to confirm a hypertension diagnosis beyond a single elevated reading.", True),
        DiagnosticTest("Ambulatory BP Monitoring (24-hr ABPM)", "Continuous BP tracking over 24 hours.",
                        "Distinguishes true hypertension from white-coat effect.", False),
        DiagnosticTest("ECG (Electrocardiogram)", "Checks for heart strain from chronic high BP.",
                        "Detects left ventricular hypertrophy, a marker of long-standing hypertension.", False),
        DiagnosticTest("Renal Function Panel", "Creatinine and electrolytes.",
                        "Hypertension and kidney disease closely interact; baseline function guides treatment.", True),
    ]
    tips = [
        "Reduce sodium intake to under 2,300 mg/day (ideally ~1,500 mg/day).",
        "Take blood pressure medications consistently if prescribed.",
        "Monitor blood pressure at home and keep a log for your clinician.",
    ]
    lifestyle = _smoking_prefix(inputs, [
        "Engage in regular aerobic exercise; even brisk walking lowers BP meaningfully.",
        "Limit alcohol and caffeine intake.",
        "Manage stress through sleep, relaxation, or mindfulness practices.",
        "Maintain a healthy body weight.",
    ])
    return DiseaseRisk(
        key="hypertension",
        disease_name="Hypertension",
        risk_score=score,
        risk_level=_level(score),
        explanation="Estimates hypertension risk primarily from systolic/diastolic blood pressure readings, "
                    "with BMI, smoking, and age as secondary contributors.",
        contributing_factors=factors or ["Blood pressure readings are within normal limits."],
        prevention_tips=tips,
        lifestyle_recommendations=lifestyle,
        diagnostic_tests=tests,
    )


# ============================================================
# 6. Stroke
# ============================================================

def predict_stroke_risk(inputs: PatientInputs) -> DiseaseRisk:
    score = 0.0
    factors = []

    if inputs.systolic_bp >= 160:
        score += 30; factors.append(f"Systolic BP very high ({int(inputs.systolic_bp)} mmHg)")
    elif inputs.systolic_bp >= 140:
        score += 20; factors.append(f"Systolic BP high ({int(inputs.systolic_bp)} mmHg)")
    elif inputs.systolic_bp >= 130:
        score += 10

    if inputs.cholesterol is not None:
        if inputs.cholesterol >= 240:
            score += 15; factors.append(f"Total cholesterol high ({inputs.cholesterol} mg/dL)")
        elif inputs.cholesterol >= 200:
            score += 8

    if inputs.smoking == "Yes":
        score += 20; factors.append("Current smoker")

    if inputs.duration_years >= 10:
        score += 10; factors.append(f"Diabetes duration {inputs.duration_years} yrs")

    if inputs.age >= 65:
        score += 20; factors.append("Age 65 or older")
    elif inputs.age >= 55:
        score += 10; factors.append("Age 55 or older")

    if inputs.sex == "Male" and inputs.age < 65:
        score += 4  # modest — the male/female stroke-risk gap is smaller than for CHD

    score = _clamp(score)
    tests = [
        DiagnosticTest("Carotid Doppler Ultrasound", "Imaging of blood flow in neck arteries.",
                        "Detects carotid artery narrowing, a major stroke risk source.", False),
        DiagnosticTest("MRI / CT Brain", "Structural brain imaging.",
                        "Detects prior silent infarcts or structural abnormalities.", False),
        DiagnosticTest("Lipid Profile", "Cholesterol and triglyceride panel.",
                        "Elevated LDL contributes directly to atherosclerotic stroke risk.", True),
        DiagnosticTest("ECG / Holter Monitor", "Heart rhythm monitoring.",
                        "Detects atrial fibrillation, a leading cause of embolic stroke.", True),
    ]
    tips = [
        "Control blood pressure — it is the single biggest modifiable stroke risk factor.",
        "Know the signs of stroke (F.A.S.T.: Face drooping, Arm weakness, Speech difficulty, Time to call emergency services).",
        "Manage atrial fibrillation and cholesterol if present.",
    ]
    lifestyle = _smoking_prefix(inputs, [
        "Follow a diet low in saturated fat and rich in fruits/vegetables.",
        "Exercise regularly to support healthy blood pressure and weight.",
        "Limit alcohol intake.",
    ])
    return DiseaseRisk(
        key="stroke",
        disease_name="Stroke",
        risk_score=score,
        risk_level=_level(score),
        explanation="Estimates stroke risk from blood pressure, cholesterol, smoking status, diabetes "
                    "duration, and age.",
        contributing_factors=factors or ["No major stroke risk factors (blood pressure, cholesterol, smoking) are currently elevated."],
        prevention_tips=tips,
        lifestyle_recommendations=lifestyle,
        diagnostic_tests=tests,
    )


# ============================================================
# Registry + orchestration
# ============================================================

DISEASE_REGISTRY: dict[str, Callable[[PatientInputs], DiseaseRisk]] = {
    "diabetes_glycemic": predict_diabetes_glycemic_risk,
    "diabetic_retinopathy": predict_retinopathy_risk,
    "heart_disease": predict_heart_disease_risk,
    "kidney_disease": predict_kidney_disease_risk,
    "hypertension": predict_hypertension_risk,
    "stroke": predict_stroke_risk,
}


def predict_all_diseases(data: dict) -> list[dict]:
    """
    Run every registered disease predictor against the given input data
    (a plain dict, e.g. request.form or a JSON body — same shape the
    existing /predict route already validates).

    Returns a list of DiseaseRisk dicts, sorted by risk_score descending.
    A failure in any single predictor is caught and surfaced as a
    degraded ("error") entry rather than failing the whole request.
    """
    try:
        inputs = PatientInputs.from_dict(data)
    except (KeyError, ValueError, TypeError) as e:
        # Should not happen if validate_input() already ran, but fail safe.
        return [{
            "key": "error",
            "disease_name": "Multi-Disease Prediction",
            "risk_score": 0,
            "risk_level": "Unknown",
            "explanation": f"Could not run multi-disease prediction: invalid input ({e}).",
            "contributing_factors": [],
            "prevention_tips": [],
            "lifestyle_recommendations": [],
            "diagnostic_tests": [],
            "source": "error",
        }]

    results = []
    for key, predictor in DISEASE_REGISTRY.items():
        try:
            results.append(predictor(inputs).to_dict())
        except Exception as e:  # noqa: BLE001 - isolate per-disease failures
            results.append({
                "key": key,
                "disease_name": key.replace("_", " ").title(),
                "risk_score": 0,
                "risk_level": "Unknown",
                "explanation": f"This prediction could not be computed ({e}).",
                "contributing_factors": [],
                "prevention_tips": [],
                "lifestyle_recommendations": [],
                "diagnostic_tests": [],
                "source": "error",
            })

    results.sort(key=lambda r: r["risk_score"], reverse=True)
    return results
