import pandas as pd
import numpy as np

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("processed_data.csv")

# ----------------------------
# Features & Target
# ----------------------------
X = df.drop("selling_price", axis=1)
y = df["selling_price"]

# ----------------------------
# Train-Test Split
# ----------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Train Models
# ----------------------------
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "DecisionTree": DecisionTreeRegressor(),
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42)
}

best_model = None
best_score = -1
best_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    score = r2_score(y_test, preds)
    print(f"{name} R2 Score: {score:.4f}")
    
    if score > best_score:
        best_score = score
        best_model = model
        best_name = name

# ----------------------------
# Best Model Output
# ----------------------------
print("\n✅ Best Model:", best_name)
print("✅ Best R2 Score:", best_score)

# ----------------------------
# Save Best Model
# ----------------------------
import pickle

with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("\n✅ Model saved successfully as best_model.pkl")