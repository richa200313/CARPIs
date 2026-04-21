import requests
import time
import json

payload = {
    "Age": 45,
    "Gender": 1,
    "gender2": 1,
    "Height_cm": 175,
    "Weight_kg": 80,
    "occupation_encoded": 0,
    "Hypertension": 0,
    "systolic_bp": 130,
    "diastolic_bp": 85,
    "total_cholesterol": 200,
    "blood_sugar": 110,
    "heart_rate": 75,
    "diabetic_encoded": 0,
    "smoking_encoded": 0,
    "If_smoker_cigarettes_per_day": 0,
    "alcohol_encoded": 0,
    "physical_activity": 1,
    "Average_Sleep_Duration": 7,
    "Stress_Level_Self_Assessment": 5,
    "diet_type": 1,
    "Steps_in_a_day": 5000,
    "family_history": 0,
    "Medical_Condition": -1
}

def verify_predict():
    for _ in range(10):
        try:
            r = requests.post("http://127.0.0.1:8000/predict", json=payload)
            if r.status_code == 200:
                print("Predict API OK:")
                print(json.dumps(r.json(), indent=2))
                return True
            else:
                print("Server returned:", r.text)
        except Exception as e:
            time.sleep(1)
            
    print("Failed to reach server.")
    return False

if __name__ == '__main__':
    verify_predict()
