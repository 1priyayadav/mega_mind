import pandas as pd
from ucimlrepo import fetch_ucirepo 

# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 

# extract data (as pandas dataframes) 
X = heart_disease.data.features 
y = heart_disease.data.targets 

print(heart_disease.metadata) 
print(heart_disease.variables)

# Combine features and targets to save as a single CSV
df = pd.concat([X, y], axis=1)

# The UCI dataset typically has the target column named 'num', containing values from 0 to 4. 
# 0 indicates absence of heart disease, 1-4 indicate presence. We map this to binary (0/1) for binary classification
if 'num' in df.columns:
    df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
    df = df.drop(columns=['num'])

# Save to CSV
df.to_csv('heart_disease.csv', index=False)
print("Dataset downloaded and saved as 'heart_disease.csv'")
