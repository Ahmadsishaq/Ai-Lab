from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load("rf_model.pkl")
features = joblib.load("features.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values
        input_data = [float(x) for x in request.form.values()]

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data], columns=features)

        # Prediction
        prediction = model.predict(input_df)[0]

        return render_template('index.html', prediction_text=f'Prediction: {prediction}')

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)