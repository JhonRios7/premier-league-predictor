import numpy as np
from app.models.load import load_h5_model, load_pkl_model

class PremierLeaguePredictor:
    def __init__(self, model_path_h5, model_path_pkl):
        self.model_h5 = load_h5_model(model_path_h5)
        self.model_pkl = load_pkl_model(model_path_pkl)

    def predict_h5(self, input_data):
        features = np.array([input_data])
        prediction = self.model_h5.predict(features)
        # Asumiendo que el modelo H5 devuelve probabilidades
        win_probability = prediction[0][0]
        # Convirtiendo a predicción binaria (1 si prob > 0.5, 0 en caso contrario)
        binary_prediction = 1 if win_probability > 0.5 else 0
        return binary_prediction, win_probability

    def predict_pkl(self, input_data):
        features = np.array([input_data])
        prediction = self.model_pkl.predict(features)
        # Obteniendo la probabilidad de victoria
        win_probability = self.model_pkl.predict_proba(features)[0][1]
        return prediction[0], win_probability