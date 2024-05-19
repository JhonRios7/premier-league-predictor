from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Cargar los modelos preentrenados
with open('models/rf_model.pkl', 'rb') as rf_model_file:
    rf_model = pickle.load(rf_model_file)

nn_model = tf.keras.models.load_model('models/nn_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    
    # Predecir con Random Forest
    rf_prediction = rf_model.predict(features)
    
    # Predecir con Neural Network
    nn_prediction = nn_model.predict(features)
    
    return jsonify({
        'rf_prediction': rf_prediction[0],
        'nn_prediction': nn_prediction[0].tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
