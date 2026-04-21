from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib, re, httpx, os
import pandas as pd
import numpy as np
import shap

app = FastAPI(title="CARPIs System API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Models ──────────────────────────────────────────────────────────────
class ClinicalInput(BaseModel):
    Age: float
    Gender: int
    gender2: int
    Height_cm: float
    Weight_kg: float
    occupation_encoded: int
    Hypertension: int
    systolic_bp: float
    diastolic_bp: float
    total_cholesterol: float
    blood_sugar: float
    heart_rate: float
    diabetic_encoded: int
    smoking_encoded: int
    If_smoker_cigarettes_per_day: float
    alcohol_encoded: int
    physical_activity: float
    Average_Sleep_Duration: float
    Stress_Level_Self_Assessment: float
    diet_type: int
    Steps_in_a_day: float
    family_history: int
    Medical_Condition: int

class BPInput(BaseModel):
    systolic_bp: float
    diastolic_bp: float

class SugarInput(BaseModel):
    blood_sugar: float
    fasting: bool = True          # True = fasting, False = post-meal

class MedicineInput(BaseModel):
    medicine_name: str

# ── Load ML artifacts ───────────────────────────────────────────────────
try:
    best_model   = joblib.load('ml_core/best_model.pkl')
    scaler       = joblib.load('ml_core/scaler.pkl')
    explainer    = joblib.load('ml_core/explainer.pkl')
    with open('ml_core/best_model_info.txt') as f:
        best_model_name = f.read().strip()
except Exception as e:
    best_model = None
    best_model_name = "Not Loaded"
    print("Warning: Models not loaded:", e)

# ── Helpers ──────────────────────────────────────────────────────────────
def apply_feature_engineering_api(df):
    df_eng = df.copy()
    height_m = df_eng['Height_cm'] / 100
    df_eng['BMI'] = df_eng['Weight_kg'] / (height_m ** 2)
    df_eng['BP_Ratio'] = np.where(df_eng['diastolic_bp'] > 0,
                                   df_eng['systolic_bp'] / df_eng['diastolic_bp'], 0)
    sf    = df_eng['smoking_encoded']
    af    = df_eng['alcohol_encoded']
    act   = df_eng['physical_activity']
    stress= df_eng['Stress Level (Self-Assessment)']
    df_eng['Lifestyle_Score'] = sf + af + act + stress
    sleep = df_eng['Average Sleep Duration']
    df_eng['Sleep_Risk'] = np.where((sleep < 5) | (sleep > 9), 1, 0)
    df_eng = df_eng.fillna(0)
    return df_eng

# ── /predict ─────────────────────────────────────────────────────────────
@app.post("/predict")
def predict_risk(data: ClinicalInput):
    if not best_model:
        raise HTTPException(status_code=500, detail="Model not available.")

    df = pd.DataFrame([{
        'Age': data.Age, 'Gender': data.Gender, 'gender2': data.gender2,
        'Height_cm': data.Height_cm, 'Weight_kg': data.Weight_kg,
        'occupation_encoded': data.occupation_encoded, 'Hypertension': data.Hypertension,
        'systolic_bp': data.systolic_bp, 'diastolic_bp': data.diastolic_bp,
        'total_cholesterol': data.total_cholesterol, 'blood_sugar': data.blood_sugar,
        'heart_rate': data.heart_rate, 'diabetic_encoded': data.diabetic_encoded,
        'smoking_encoded': data.smoking_encoded,
        'If smoker, cigarettes per day': data.If_smoker_cigarettes_per_day,
        'alcohol_encoded': data.alcohol_encoded, 'physical_activity': data.physical_activity,
        'Average Sleep Duration': data.Average_Sleep_Duration,
        'Stress Level (Self-Assessment)': data.Stress_Level_Self_Assessment,
        'diet_type': data.diet_type, 'Steps in a day': data.Steps_in_a_day,
        'family_history': data.family_history, 'Medical Condition': data.Medical_Condition
    }])

    df_eng = apply_feature_engineering_api(df)
    df_eng.columns = [re.sub(r'[\[\]<>, ()-]', '_', col) for col in df_eng.columns]

    scale_models = ['Logistic Regression', 'KNN', 'SVM']
    X_pred = df_eng
    if best_model_name in scale_models:
        X_pred = pd.DataFrame(scaler.transform(df_eng), columns=df_eng.columns)

    try:
        pred = best_model.predict(X_pred)[0]
        prob = best_model.predict_proba(X_pred)[0][1] if hasattr(best_model, 'predict_proba') else float(pred)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # SHAP – only positive (risk-increasing) factors
    explanations = []
    try:
        shap_values = explainer.shap_values(X_pred)
        shap_output = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
        features = df_eng.columns
        pos = [(features[i], shap_output[i]) for i in range(len(features)) if shap_output[i] > 0]
        pos.sort(key=lambda x: x[1], reverse=True)
        for factor, _ in pos[:3]:
            explanations.append(factor.replace('_', ' ').title())
    except:
        pass

    recommendations = []
    if prob > 0.5:
        recommendations.append("High risk detected – please consult a cardiologist.")
        if data.smoking_encoded > 0:
            recommendations.append("Consider a smoking cessation programme.")
        recommendations.append("Monitor blood pressure and blood sugar regularly.")
    else:
        recommendations.append("Low risk. Maintain your current healthy lifestyle.")

    return {
        "prediction": int(pred),
        "risk_percentage": float(prob * 100),
        "explanation": explanations,
        "recommendations": recommendations,
        "disclaimer": "AI-based assessment – not a medical diagnosis."
    }

# ── /check-bp (SEPARATE) ─────────────────────────────────────────────────
@app.post("/check-bp")
def check_bp(data: BPInput):
    sys, dia = data.systolic_bp, data.diastolic_bp

    if sys > 180 or dia > 120:
        stage, color = "Hypertensive Crisis", "critical"
        advice = ["Seek emergency care immediately.",
                  "Do NOT exercise. Sit quietly and call emergency services.",
                  "Avoid any stimulants (caffeine, tobacco)."]
    elif sys >= 140 or dia >= 90:
        stage, color = "Hypertension Stage 2", "high"
        advice = ["Consult your doctor as soon as possible.",
                  "Significantly reduce sodium to < 1500 mg/day.",
                  "Start daily 30-min moderate walking if cleared by doctor."]
    elif sys >= 130 or dia >= 80:
        stage, color = "Hypertension Stage 1", "elevated"
        advice = ["Adopt DASH diet (fruits, vegetables, whole grains).",
                  "Limit alcohol and caffeine.",
                  "Monitor BP twice daily."]
    elif sys >= 120:
        stage, color = "Elevated BP (Prehypertension)", "warning"
        advice = ["Reduce sodium intake.",
                  "Exercise at least 150 min/week.",
                  "Manage stress through meditation or yoga."]
    else:
        stage, color = "Normal", "normal"
        advice = ["Great! Keep up healthy habits.",
                  "Stay hydrated and maintain regular physical activity."]

    return {
        "systolic": sys, "diastolic": dia,
        "stage": stage, "color": color,
        "advice": advice,
        "disclaimer": "AI-based assessment – consult a physician for clinical decisions."
    }

# ── /check-sugar (SEPARATE) ──────────────────────────────────────────────
@app.post("/check-sugar")
def check_sugar(data: SugarInput):
    val, fasting = data.blood_sugar, data.fasting

    if fasting:
        if val >= 126:
            stage, color = "Diabetic Range (Fasting)", "critical"
            advice = ["Consult your doctor immediately for HbA1c confirmation.",
                      "Avoid sugar, refined carbs, and sweetened beverages.",
                      "Monitor blood sugar daily."]
        elif val >= 100:
            stage, color = "Prediabetes (Fasting)", "elevated"
            advice = ["Follow a low-glycaemic diet.",
                      "Aim for 150 min of exercise per week.",
                      "Retest in 3 months."]
        else:
            stage, color = "Normal (Fasting)", "normal"
            advice = ["Excellent! Maintain a balanced diet.",
                      "Continue regular physical activity."]
    else:
        # Post-meal (2-hr postprandial)
        if val >= 200:
            stage, color = "Diabetic Range (Post-Meal)", "critical"
            advice = ["Seek immediate medical evaluation.",
                      "Avoid all simple sugars.",
                      "Check for symptoms: excessive thirst, frequent urination."]
        elif val >= 140:
            stage, color = "Prediabetes (Post-Meal)", "elevated"
            advice = ["Reduce portion sizes and avoid high-GI foods.",
                      "Walk 20 min after each meal.",
                      "Consider dietitian consultation."]
        else:
            stage, color = "Normal (Post-Meal)", "normal"
            advice = ["Blood sugar is well-controlled after meals.",
                      "Maintain healthy meal timing."]

    return {
        "blood_sugar": val, "fasting": fasting,
        "stage": stage, "color": color,
        "advice": advice,
        "disclaimer": "AI-based assessment – consult a physician for clinical decisions."
    }

# ── /medicine-info (OpenFDA) ─────────────────────────────────────────────
@app.post("/medicine-info")
async def medicine_info(data: MedicineInput):
    name = data.medicine_name.strip()

    # 1️⃣  Try OpenFDA drug label search
    fda_url = (
        f"https://api.fda.gov/drug/label.json"
        f"?search=openfda.brand_name:\"{name}\""
        f"&limit=1"
    )
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(fda_url)
            fda = r.json()
        results = fda.get("results", [])
        if not results:
            # Fallback: generic name search
            fda_url2 = (
                f"https://api.fda.gov/drug/label.json"
                f"?search=openfda.generic_name:\"{name}\""
                f"&limit=1"
            )
            r2 = await client.get(fda_url2) if False else await httpx.AsyncClient(timeout=8).get(fda_url2)
            fda = r2.json()
            results = fda.get("results", [])

        if results:
            r0 = results[0]

            def first(field):
                val = r0.get(field, [])
                return val[0][:600] if val else "Not specified."

            brand   = (r0.get("openfda", {}).get("brand_name") or [name.capitalize()])[0]
            generic = (r0.get("openfda", {}).get("generic_name") or ["—"])[0]

            return {
                "medicine_name":  brand,
                "generic_name":   generic,
                "what_it_does":   first("description") or first("purpose"),
                "indications":    first("indications_and_usage"),
                "dosage":         first("dosage_and_administration"),
                "when_to_take":   first("dosage_and_administration"),
                "precautions":    first("warnings_and_cautions") or first("warnings"),
                "side_effects":   first("adverse_reactions"),
                "contraindications": first("contraindications"),
                "storage":        first("how_supplied"),
                "warning":        "Always consult a licensed physician before taking any medicine.",
                "source":         "OpenFDA Drug Labels"
            }
    except Exception:
        pass

    # Fallback if FDA unavailable
    return {
        "medicine_name": name.capitalize(),
        "generic_name":  "—",
        "what_it_does":  "Information unavailable from FDA database at this time.",
        "indications":   "Please consult a pharmacist or doctor.",
        "dosage":        "Follow prescriber's instructions.",
        "when_to_take":  "As directed by your doctor.",
        "precautions":   "Consult your physician for individual precautions.",
        "side_effects":  "Varies – see package insert.",
        "contraindications": "Consult your doctor.",
        "storage":       "Store in a cool, dry place.",
        "warning":       "Always consult a licensed physician before taking any medicine.",
        "source":        "Fallback (FDA API unreachable)"
    }

# ── Static file serving (MUST be last — catch-all for frontend) ─────────
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
