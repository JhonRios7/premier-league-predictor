import tensorflow as tf
import pickle
from fastapi import FastAPI, HTTPException

app = FastAPI()

def load_h5_model(model_path: str):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading H5 model: {str(e)}")

def load_pkl_model(model_path: str):
    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading PKL model: {str(e)}")

# Cargar modelos al iniciar la aplicación
h5_model = load_h5_model('ruta/al/modelo.h5')
pkl_model = load_pkl_model('ruta/al/modelo.pkl')

@app.on_event("startup")
async def startup_event():
    global h5_model, pkl_model
    h5_model = load_h5_model('ruta/al/modelo.h5')
    pkl_model = load_pkl_model('ruta/al/modelo.pkl')