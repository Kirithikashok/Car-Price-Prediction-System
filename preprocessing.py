import pandas as pd
import numpy as np

# ----------------------------
# 1. Load Dataset
# ----------------------------
df = pd.read_csv("cardekho_dataset.csv")

print("Initial Shape:", df.shape)

# ----------------------------
# 2. Basic Cleaning
# ----------------------------

# Drop index column if exists
if 'Unnamed: 0' in df.columns:
    df.drop('Unnamed: 0', axis=1, inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# ----------------------------
# 3. Handle Missing Values
# ----------------------------

# Check missing values
print(df.isnull().sum())

# Numerical columns
num_cols = ['vehicle_age', 'km_driven', 'mileage', 
            'engine', 'max_power', 'seats']

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col].fillna(df[col].median(), inplace=True)

# Categorical columns
cat_cols = ['seller_type', 'fuel_type', 'transmission_type']

for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# ----------------------------
# 4. Fix Inconsistent Values
# ----------------------------

# Standardize text
df['seller_type'] = df['seller_type'].str.strip()
df['fuel_type'] = df['fuel_type'].str.strip()
df['transmission_type'] = df['transmission_type'].str.strip()

# ----------------------------
# 5. Handle Outliers (IMPORTANT)
# ----------------------------

def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

for col in ['km_driven', 'selling_price', 'engine', 'max_power']:
    df = remove_outliers(df, col)

print("Shape after outlier removal:", df.shape)

# ----------------------------
# 6. Feature Selection
# ----------------------------

# Drop unnecessary columns
df_model = df.drop(['car_name', 'brand', 'model'], axis=1)

# ----------------------------
# 7. Encoding (VERY IMPORTANT)
# ----------------------------

df_model = pd.get_dummies(df_model, dtype=float)

# ----------------------------
# 8. Split Features & Target
# ----------------------------

X = df_model.drop('selling_price', axis=1)
y = df_model['selling_price']

# ----------------------------
# 9. (Optional) Scaling
# ----------------------------

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# 10. Final Output
# ----------------------------

print("Final Dataset Shape:", df_model.shape)
print("Features Shape:", X.shape)

# Save processed file (optional)
df_model.to_csv("processed_data.csv", index=False)

print("✅ Preprocessing Completed Successfully")