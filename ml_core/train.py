import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import joblib
import shap
import warnings
import os

warnings.filterwarnings('ignore')

os.makedirs('ml_core/viz', exist_ok=True)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(probability=True, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
    'CatBoost': CatBoostClassifier(random_state=42, verbose=0)
}

scale_models = ['Logistic Regression', 'KNN', 'SVM']

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    try:
        roc = roc_auc_score(y_test, probs)
    except:
        roc = 0.5
    return {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1 Score': f1, 'ROC-AUC': roc}

import re
def run_pipeline(data_path, is_engineered=False):
    df = pd.read_csv(data_path)
    X = df.drop(columns=['diagnosed_before'])
    X.columns = [re.sub(r'[\[\]<>, ()-]', '_', col) for col in X.columns]
    y = df['diagnosed_before'].astype(int)
    y = np.where(y > 0, 1, 0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    fitted_models = {}
    
    for name, m in models.items():
        if name in scale_models:
            m.fit(X_train_scaled, y_train)
            res = evaluate(m, X_test_scaled, y_test)
        else:
            m.fit(X_train, y_train)
            res = evaluate(m, X_test, y_test)
        results[name] = res
        fitted_models[name] = m
        
    return results, fitted_models, scaler, X_train, X_test, y_test

print("Training on BASE dataset...")
base_res, base_mods, _, _, _, _ = run_pipeline('ml_core/merged_data.csv', False)

print("Training on ENGINEERED dataset...")
eng_res, eng_mods, eng_scaler, X_train_eng, X_test_eng, y_test_eng = run_pipeline('ml_core/engineered_data.csv', True)

best_score = 0
best_model_name = ""
best_model = None

with open('ml_core/model_results.txt', 'w') as f:
    for name in models.keys():
        b_acc = base_res[name]['Accuracy']
        e_acc = eng_res[name]['Accuracy']
        b_f1 = base_res[name]['F1 Score']
        e_f1 = eng_res[name]['F1 Score']
        b_roc = base_res[name]['ROC-AUC']
        e_roc = eng_res[name]['ROC-AUC']
        
        score = (e_roc * 0.5) + (e_f1 * 0.3) + (e_acc * 0.2)
        if score > best_score:
            best_score = score
            best_model_name = name
            best_model = eng_mods[name]

        f.write(f"=== {name} ===\n")
        f.write(f"BASE -> Acc: {b_acc:.3f}, F1: {b_f1:.3f}, ROC: {b_roc:.3f}\n")
        f.write(f"ENG  -> Acc: {e_acc:.3f}, F1: {e_f1:.3f}, ROC: {e_roc:.3f}\n\n")

        fig, ax = plt.subplots(figsize=(8, 5))
        metrics = ['Accuracy', 'F1 Score', 'ROC-AUC']
        b_vals = [b_acc, b_f1, b_roc]
        e_vals = [e_acc, e_f1, e_roc]
        
        x = np.arange(len(metrics))
        width = 0.35
        ax.bar(x - width/2, b_vals, width, label='Before FE')
        ax.bar(x + width/2, e_vals, width, label='After FE')
        ax.set_ylabel('Scores')
        ax.set_title(f'{name} Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        plt.savefig(f'ml_core/viz/{name.replace(" ", "_").lower()}_comparison.png')
        plt.close()

print(f"Best Model: {best_model_name}")
joblib.dump(best_model, 'ml_core/best_model.pkl')
joblib.dump(eng_scaler, 'ml_core/scaler.pkl')

try:
    with open('ml_core/best_model_info.txt', 'w') as f:
        f.write(best_model_name)
    
    if best_model_name in scale_models:
        X_train_exp = pd.DataFrame(eng_scaler.transform(X_train_eng), columns=X_train_eng.columns)
    else:
        X_train_exp = X_train_eng
        
    explainer = shap.Explainer(best_model.predict, X_train_exp)
    joblib.dump(explainer, 'ml_core/explainer.pkl')
    print("SHAP explainer saved.")
except Exception as e:
    print("SHAP explainer failed:", e)

print("Training finished.")
