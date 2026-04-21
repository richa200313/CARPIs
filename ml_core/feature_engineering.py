import pandas as pd
import numpy as np

def perform_feature_engineering(df):
    df_eng = df.copy()
    
    def clean_num(col):
        # Extract first numeric sequence and convert.
        return pd.to_numeric(df_eng[col].astype(str).str.extract(r'(\-?\d+\.?\d*)', expand=False), errors='coerce')
    
    df_eng['Weight_kg'] = clean_num('Weight_kg').replace(-1, np.nan)
    df_eng['Height_cm'] = clean_num('Height_cm').replace(-1, np.nan)
    
    height_m = df_eng['Height_cm'] / 100
    df_eng['BMI'] = df_eng['Weight_kg'] / (height_m ** 2)
    df_eng['BMI'] = df_eng['BMI'].fillna(-1)
    
    df_eng['systolic_bp'] = clean_num('systolic_bp').replace(-1, np.nan)
    df_eng['diastolic_bp'] = clean_num('diastolic_bp').replace(-1, np.nan)
    df_eng['BP_Ratio'] = np.where(df_eng['diastolic_bp'] > 0, 
                                  df_eng['systolic_bp'] / df_eng['diastolic_bp'], 
                                  np.nan)
    df_eng['BP_Ratio'] = df_eng['BP_Ratio'].fillna(-1)
    
    sf = clean_num('smoking_encoded').replace(-1, 0).fillna(0)
    af = clean_num('alcohol_encoded').replace(-1, 0).fillna(0)
    act = clean_num('physical_activity').replace(-1, 0).fillna(0)
    stress = clean_num('Stress Level (Self-Assessment)').replace(-1, 0).fillna(0)
    df_eng['Lifestyle_Score'] = sf + af + act + stress
    
    sleep = clean_num('Average Sleep Duration').replace(-1, np.nan)
    df_eng['Sleep_Risk'] = np.where((sleep < 5) | (sleep > 9), 1, 0)
    df_eng['Sleep_Risk'] = np.where(sleep.isna(), -1, df_eng['Sleep_Risk'])
    
    df_eng = df_eng.fillna(-1)
    return df_eng

if __name__ == '__main__':
    df = pd.read_csv('ml_core/merged_data.csv')
    df_eng = perform_feature_engineering(df)
    df_eng.to_csv('ml_core/engineered_data.csv', index=False)
    print("Engineered dataset created.")
