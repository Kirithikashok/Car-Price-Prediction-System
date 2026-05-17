import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("processed_data.csv")

# ----------------------------
# OPTIONAL: Remove Outliers (boost accuracy)
# ----------------------------
df = df[df["selling_price"] < df["selling_price"].quantile(0.99)]

# ----------------------------
# Features & Target
# ----------------------------
X = df.drop("selling_price", axis=1)
y = df["selling_price"]

# ----------------------------
# Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Hyperparameter Tuning
# ----------------------------
rf = RandomForestRegressor(random_state=42)

param_dist = {
    'n_estimators': [200, 300, 400, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=20,              # ✅ increase for better accuracy
    cv=5,
    scoring='r2',
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# ----------------------------
# Train Best Model
# ----------------------------
random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

# ----------------------------
# Evaluate
# ----------------------------
y_pred = best_model.predict(X_test)

r2 = r2_score(y_test, y_pred)

print("\n✅ Improved Model R2 Score:", r2)
print("✅ Best Parameters:", random_search.best_params_)

# ----------------------------
# Save Model
# ----------------------------
with open("best_model_F.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("\n✅ New Improved Model Saved as best_model.pkl")
