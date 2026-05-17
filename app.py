from flask import Flask, render_template, request
import pickle
import numpy as np

# ✅ Load trained model
model = pickle.load(open('best_model_F.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.form

        # ✅ Numeric inputs
        vehicle_age = int(form_data['vehicle_age'])
        km_driven = int(form_data['km_driven'])
        mileage = float(form_data['mileage'])
        engine = int(form_data['engine'])
        max_power = float(form_data['max_power'])
        seats = int(form_data['seats'])

        # ✅ Categorical inputs
        seller_type = form_data['seller_type']
        fuel_type = form_data['fuel_type']
        transmission_type = form_data['transmission_type']

        # ✅ One-hot encoding (must match training data)
        seller_dict = {'Dealer': 0, 'Individual': 0, 'Trustmark Dealer': 0}
        seller_dict[seller_type] = 1

        fuel_dict = {'CNG': 0, 'Diesel': 0, 'LPG': 0, 'Petrol': 0}
        fuel_dict[fuel_type] = 1

        trans_dict = {'Automatic': 0, 'Manual': 0}
        trans_dict[transmission_type] = 1

        # ✅ Correct feature order (VERY IMPORTANT)
        input_features = [
            vehicle_age, km_driven, mileage, engine, max_power, seats,
            seller_dict['Dealer'], seller_dict['Individual'], seller_dict['Trustmark Dealer'],
            fuel_dict['CNG'], fuel_dict['Diesel'], fuel_dict['LPG'], fuel_dict['Petrol'],
            trans_dict['Automatic'], trans_dict['Manual']
        ]

        # ✅ Prediction
        prediction = model.predict([input_features])

        return render_template('index.html',
                               prediction_text=f"Predicted Price: ₹ {round(prediction[0], 2)}")

    except Exception as e:
        return render_template('index.html',
                               prediction_text=f"Error: {str(e)}")


# ✅ Run app
if __name__ == "__main__":
    app.run(debug=True, port=8001)