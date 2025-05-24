from fastapi import FastAPI, logger
from pydantic import BaseModel

import main

app = FastAPI()
predictor = main.EnsembledPredictor()

class InputParameters(BaseModel):
    crop_name: str
    weather_type: str
    fertilizer_type: str
    images: list[str]

class OutputParameters(BaseModel):
    yield_production_estimate: float
    disease_prediction: str


@app.post("/predict", response_model=OutputParameters)
async def predict(input_parameters: InputParameters):
    # yield_production_estimate = predictor.predict_yield(input_parameters)
    disease_prediction = predictor.EnsembledPredictor.predict_disease(input_parameters.images)
    return OutputParameters(disease_prediction=disease_prediction)

@app.post("/predict_production", response_model=OutputParameters)
async def predict_production(input_parameters: InputParameters):
    yield_production_estimate = predictor.predict_production(input_parameters)
    return OutputParameters(yield_production_estimate=yield_production_estimate)


