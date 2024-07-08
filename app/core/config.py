from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_PATH_H5: str = "models/nn_model.h5"
    MODEL_PATH_PKL: str = "models/rf_model.pkl"

    class Config:
        env_file = ".env"

settings = Settings()