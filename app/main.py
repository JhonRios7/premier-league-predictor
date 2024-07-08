from fastapi import FastAPI
from app.api import endpoints
from app.models.load import load_h5_model, load_pkl_model
from app.core.config import settings

app = FastAPI(title="Premier League Predictor API")

app.include_router(endpoints.router)

@app.on_event("startup")
async def startup_event():
    app.state.h5_model = load_h5_model(settings.MODEL_PATH_H5)
    app.state.pkl_model = load_pkl_model(settings.MODEL_PATH_PKL)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)