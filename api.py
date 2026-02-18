import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from typing import Dict


app = FastAPI(title="SpaceX Launch Success Predictor")

try:
    model = joblib.load("rocket_launch_predictor.pkl")

except FileNotFoundError:
     raise RuntimeError("Model file is not found. Either check the directory properly or train the model first.")

#the exact format the user must send when making a prediction request
class LaunchInput(BaseModel):
    month: int = Field(..., ge=1, le=12, description="Month of launch (1-12)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    year: int = Field(..., ge=2000, le=2030, description="Year of launch")
    wind_speed: float = Field(..., ge=0, description="Wind speed in m/s")
    temperature: float = Field(..., description="Temperature in Celsius")
    precipitation: float = Field(..., ge=0, description="Precipitation in mm")
    cloud_cover: float = Field(..., ge=0, le=100, description="Cloud cover percentage (0-100)")

    class Config:
         json_schema_extra = {
              "example" :{
                   "month" : 5,
                   "day_of_week": 3,
                   "year": 2026,
                   "wind_speed": 23.4,
                   "temperature": 28.0,
                   "precipitation": 0.0,
                   "cloud_cover": 20.0              
                   }
         }

class PredictionOutput(BaseModel):
     prediction: str
     probability_of_failure: float
     probability_of_success: float
     confidence: float
     risk_level: str
     input_features: Dict

@app.get("/")
def root():
     return {
          "message": "SpaceX Launch Success Prediction API",
          "endpoints": {
            "POST /predict": "Predict launch outcome",
        }
     }


@app.post("/predict", response_model=PredictionOutput)
def predict_launch_success(launch: LaunchInput):

       data = pd.DataFrame([[
        launch.month,
        launch.day_of_week,
        launch.year,
        launch.wind_speed,
        launch.temperature,
        launch.precipitation,
        launch.cloud_cover
    ]], columns=["month", "day_of_week", "year", "wind_speed", 
                 "temperature", "precipitation", "cloud_cover"])
       
       prediction = int(model.predict(data)[0])
       probability_failure = float(model.predict_proba(data)[0][1])
       probability_success = 1 - probability_failure
       confidence = max(probability_failure, probability_success)

       if probability_failure < 0.2:
            risk_level = "Low"
       elif probability_failure < 0.5:
            risk_level = "Medium"
       else:
            risk_level = "High"

       return PredictionOutput(  # Return as PredictionOutput object
            prediction="Failure" if prediction == 1 else "Success",
            probability_of_failure=round(probability_failure, 3),
            probability_of_success=round(probability_success, 3),
            confidence=round(confidence, 3),
            risk_level=risk_level,
            input_features=launch.dict()
        )

@app.get("/model-info")
def model_info():
    return {
        "model_type": "Decision Tree Classifier",
        "features": [
            "month", "day_of_week", "year", "wind_speed",
            "temperature", "precipitation", "cloud_cover"
        ],
        "training_data": "150+ SpaceX launches",
        "data_sources": ["Launch Library 2 API", "Open-Meteo Weather API"],
        "accuracy": "87%",
        "note": "Model trained on historical data with class imbalance (141 successes, 9 failures)"
    }

