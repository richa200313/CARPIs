import pandas as pd
df1 = pd.read_excel('Cardiovascular_Data.xlsx')
df2 = pd.read_excel('DR_PULKIT.xlsx')
print('=== DF1 Unique Cols ===')
print(set(df1.columns) - set(df2.columns))
print('=== DF2 Unique Cols ===')
print(set(df2.columns) - set(df1.columns))
print('---')
print(df2[['Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15']].head())
