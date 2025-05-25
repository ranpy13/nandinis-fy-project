from fastapi import FastAPI, logger
from pydantic import BaseModel

import main

import debugpy

# Allow other computers to attach (for Docker use 0.0.0.0)
debugpy.listen(("0.0.0.0", 5678))

print("⏳ Waiting for debugger attach...")
debugpy.wait_for_client()  # Optional: pause until debugger attaches
print("✅ Debugger is attached!")


app = FastAPI()
predictor = main.EnsembledPredictor()

class DiseaseInputParameters(BaseModel):
    crop_name: str
    weather_type: str
    fertilizer_type: str
    images: list[str]

class DiseaseOutputParameters(BaseModel):
    yield_production_estimate: float
    disease_prediction: str

class CropInputData(BaseModel):
    State_Name: str
    District_Name: str
    Crop_Year: int
    Season: str
    Crop: str
    Area: float

class CropOutputData(BaseModel):
    production: float


@app.post("/predict_disease", response_model=DiseaseOutputParameters)
async def predict(input_parameters: DiseaseInputParameters):
    # yield_production_estimate = predictor.predict_yield(input_parameters)
    disease_prediction = predictor.EnsembledPredictor.predict_disease(input_parameters.images)
    return DiseaseOutputParameters(disease_prediction=disease_prediction)

@app.post("/predict_production", response_model=CropOutputData)
async def predict_production(input_parameters: CropInputData):
    print(f"input parameters: {input_parameters}")
    yield_production_estimate = predictor.predict_production(input_parameters)
    return CropOutputData(production=yield_production_estimate)


