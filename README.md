## 📌 Project Overview
This project is a Machine Learning-based Car Price Prediction System that predicts the selling price of a car based on key features such as mileage, engine power, fuel type, and more.
The system includes:

Data preprocessing pipeline
Model training and evaluation
Hyperparameter tuning for improved accuracy
Flask-based web application for real-time prediction


## 🎯 Objectives

Build a regression model to predict car prices accurately
Compare different machine learning models
Improve model performance using tuning techniques
Deploy the model with a simple web interface


## 🛠️ Tech Stack

Language: Python
Libraries:

Pandas
NumPy
Scikit-learn
Pickle


Framework: Flask
Frontend: HTML
IDE: VS Code


## 📂 Project Structure
#### car_price_project/
#### │
#### ├── app.py                          # Flask backend
#### ├── best_model.pkl                 # Final trained model
#### ├── best_model_f.pkl               # Alternate model version
#### ├── best_model1.pkl                # Old model version
#### │
#### ├── best_model_training.py         # Base training script
#### ├── model_training_preprocessed.py # Training with processed data
#### ├── model_training_processed.py    # Another training variation
#### │
#### ├── preprocessing.py               # Data preprocessing logic
#### ├── processed_data.csv             # Final cleaned dataset
#### ├── cardekho_dataset.csv           # Original dataset
#### │
#### └── templates/
####     └── index.html                 # Frontend UI


## ⚙️ Features Used for Prediction

Vehicle Age

Kilometers Driven

Mileage

Engine Capacity

Maximum Power

Number of Seats

Seller Type (Dealer / Individual / Trustmark Dealer)

Fuel Type (Petrol / Diesel / CNG / LPG)

Transmission Type (Manual / Automatic)


## 🤖 Machine Learning Models
The following models were used during experimentation:

Linear Regression

Ridge Regression

Lasso Regression

Decision Tree Regressor

Random Forest Regressor ✅ (Final Model)


## 📊 Model Performance
ModelR² ScoreLinear Regression~0.73Ridge~0.73Lasso~0.73Decision Tree~0.80Random Forest ✅~0.87
✅ After optimization & tuning:

Achieved ~0.90+ accuracy


## 🚀 How to Run the Project
1️⃣ Install Dependencies
pip install pandas numpy scikit-learn flask


2️⃣ Train Model (Optional)
Run improved training script:
python best_model_training.py

This generates:
best_model.pkl


3️⃣ Run Flask Application
python app.py

Open in browser:
http://127.0.0.1:8001
