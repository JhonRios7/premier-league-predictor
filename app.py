from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Cargar los modelos
rf_model = joblib.load('models/rf_model.pkl')
nn_model = load_model('models/nn_model.h5')
scaler = StandardScaler()

# Cargar datos y entrenar el escalador (solo para este ejemplo, idealmente guardar el escalador)
matches = pd.read_csv('matches.csv', sep=',')
predictors = ["Venue_code", "Opp_code", "Hour", "Day_code"]
cols = ["GF", "GA", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]
new_cols = [f"{c}_rolling" for c in cols]

# Preprocesamiento
def preprocess(data):
    data = pd.DataFrame(data, index=[0])
    data[new_cols] = data[new_cols].fillna(data[new_cols].mean())
    scaled_data = scaler.transform(data[predictors + new_cols])
    return scaled_data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    processed_data = preprocess(data)
    
    # Predicciones
    rf_prediction = rf_model.predict(processed_data)
    nn_prediction = (nn_model.predict(processed_data) > 0.5).astype("int32")
    
    return jsonify({
        'rf_prediction': int(rf_prediction[0]),
        'nn_prediction': int(nn_prediction[0])
    })

if __name__ == "__main__":
    app.run(debug=True)
