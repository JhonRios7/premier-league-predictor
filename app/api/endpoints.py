from fastapi import APIRouter, Depends, HTTPException
from app.models.predictor import PremierLeaguePredictor
from app.core.config import settings
from pydantic import BaseModel

router = APIRouter()

class PredictionInput(BaseModel):
    Venue_code: int
    Opp_code: int
    Hour: int
    Day_code: int
    GF_rolling: float
    GA_rolling: float
    Sh_rolling: float
    SoT_rolling: float
    Dist_rolling: float
    FK_rolling: float
    PK_rolling: float
    PKatt_rolling: float

class PredictionOutput(BaseModel):
    prediction: float
    win_probability: float

def get_predictor():
    return PremierLeaguePredictor(settings.MODEL_PATH_H5, settings.MODEL_PATH_PKL)

@router.get("/health")
async def health_check():
    return {"status": "healthy"}

@router.post("/predict/h5", response_model=PredictionOutput)
async def predict_h5(input_data: PredictionInput, predictor: PremierLeaguePredictor = Depends(get_predictor)):
    try:
        features = [
            input_data.Venue_code,
            input_data.Opp_code,
            input_data.Hour,
            input_data.Day_code,
            input_data.GF_rolling,
            input_data.GA_rolling,
            input_data.Sh_rolling,
            input_data.SoT_rolling,
            input_data.Dist_rolling,
            input_data.FK_rolling,
            input_data.PK_rolling,
            input_data.PKatt_rolling
        ]
        prediction, win_probability = predictor.predict_h5(features)
        return {"prediction": float(prediction), "win_probability": float(win_probability)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/pkl", response_model=PredictionOutput)
async def predict_pkl(input_data: PredictionInput, predictor: PremierLeaguePredictor = Depends(get_predictor)):
    try:
        features = [
            input_data.Venue_code,
            input_data.Opp_code,
            input_data.Hour,
            input_data.Day_code,
            input_data.GF_rolling,
            input_data.GA_rolling,
            input_data.Sh_rolling,
            input_data.SoT_rolling,
            input_data.Dist_rolling,
            input_data.FK_rolling,
            input_data.PK_rolling,
            input_data.PKatt_rolling
        ]
        prediction, win_probability = predictor.predict_pkl(features)
        return {"prediction": float(prediction), "win_probability": float(win_probability)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))