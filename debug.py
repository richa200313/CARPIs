import pandas as pd
df1 = pd.read_excel('Cardiovascular_Data.xlsx')
df2 = pd.read_excel('DR_PULKIT.xlsx')

with open('debug.txt', 'w') as f:
    f.write("DF1 Columns:\n")
    for c in df1.columns:
        f.write(f"- {c}\n")
    f.write("\n------------------\n")
    f.write("DF2 Columns:\n")
    for c in df2.columns:
        f.write(f"- {c}\n")
    f.write("\n------------------\n")
    f.write(f"DF1 diagnosed_before types: {df1['diagnosed_before'].unique()}\n")
    f.write(f"DF2 Medical Condition types: {df2['MEDICAL CONDITION (DM/HTN/HD)'].unique()}\n")
