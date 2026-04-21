import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_and_merge_data():
    df1 = pd.read_excel('C:/Users/richa/OneDrive/Desktop/CARPIs/Cardiovascular_Data.xlsx')
    df2 = pd.read_excel('C:/Users/richa/OneDrive/Desktop/CARPIs/DR_PULKIT.xlsx')
    
    df2 = df2.loc[:, ~df2.columns.str.contains('^Unnamed')]
    
    rename_map = {
        'AGE': 'Age',
        'Sex': 'Gender',
        'BLOOD PRESSURE higher': 'systolic_bp',
        'blood pressure lower': 'diastolic_bp',
        'Heart Rate': 'heart_rate',
        'BLOOD SUGAR': 'blood_sugar',
        'TOTAL COLESTROL': 'total_cholesterol',
        'PHYSICAL ACTIVITY': 'physical_activity',
        'SMOKING': 'smoking_encoded',
        'ALCOHOL HISTORY': 'alcohol_encoded',
        'DIET TYPE': 'diet_type',
        'STRESS LEVEL': 'Stress Level (Self-Assessment)',
        'FAMILY HISTORY OF HEART DISEASE': 'family_history',
        'MEDICAL CONDITION (DM/HTN/HD)': 'Medical Condition'
    }
    df2 = df2.rename(columns=rename_map)
    
    df1['Medical Condition'] = -1
    df2['diagnosed_before'] = 1  
    
    df_merged = pd.concat([df1, df2], ignore_index=True)
    
    # Clean string numeric values
    for col in df_merged.columns:
        if df_merged[col].dtype == 'object':
            df_merged[col] = pd.to_numeric(df_merged[col].astype(str).str.extract(r'(\-?\d+\.?\d*)', expand=False), errors='coerce')
    
    df_merged = df_merged.fillna(-1)
    df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)
    return df_merged

if __name__ == '__main__':
    df = load_and_merge_data()
    os.makedirs('ml_core', exist_ok=True)
    df.to_csv('ml_core/merged_data.csv', index=False)
    print("Clean merged dataset created.")
