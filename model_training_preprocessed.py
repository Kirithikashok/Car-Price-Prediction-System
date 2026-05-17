import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("cardekho_dataset.csv")

# ----------------------------
# Feature Engineering
# ----------------------------

df_model = df.copy()

# Drop unnecessary features
df_model.drop(labels=['car_name','brand','model'], axis=1, inplace=True)

# Convert categorical -> numeric
df_model = pd.get_dummies(df_model, dtype=float)

# ----------------------------
# Feature split
# ----------------------------

X = df_model.drop('selling_price', axis=1)
y = df_model['selling_price']

# ----------------------------
# Train Test Split
# ----------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ----------------------------
# Feature Importance
# ----------------------------

from sklearn.ensemble import ExtraTreesRegressor

model = ExtraTreesRegressor()
model.fit(X, y)

# ----------------------------
# Model Building
# ----------------------------

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score

models = [
    LinearRegression,
    SVR,
    DecisionTreeRegressor,
    RandomForestRegressor,
    Ridge,
    Lasso
]

mse = []
rmse = []
evs = []
r_square_score = []

for model in models:
    regressor = model().fit(X_train, y_train)
    pred = regressor.predict(X_test)

    mse.append(mean_squared_error(y_test, pred))
    rmse.append(np.sqrt(mean_squared_error(y_test, pred)))
    evs.append(explained_variance_score(y_test, pred))
    r_square_score.append(r2_score(y_test, pred))

# ----------------------------
# Hyperparameter Tuning
# ----------------------------

from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    'n_estimators': [100,200,300,400,500],
    'max_depth': [None,10,20,30,40,50],
    'min_samples_split': [2,5,10],
    'min_samples_leaf': [1,2,4],
    'max_features': ['auto','sqrt','log2'],
    'bootstrap': [True, False]
}

rf = RandomForestRegressor()

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=100,
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

best_rf = random_search.best_estimator_

# ----------------------------
# Final evaluation
# ----------------------------

y_pred = best_rf.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred)
rmse_rf = np.sqrt(mse_rf)
evs_rf = explained_variance_score(y_test, y_pred)
r2_rf = r2_score(y_test, y_pred)

print("Random Forest R2:", r2_rf)

# ----------------------------
# Save Model
# ----------------------------

import pickle

with open("best_model1.pkl", "wb") as f:
    pickle.dump(best_rf, f)