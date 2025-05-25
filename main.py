from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import pickle
from pydantic import BaseModel
import pandas as pd
import torchvision
from typing import List, Tuple
from utils.logger_util import setup_logger
from fertilizer_sub_model.fertilizer_sb import FertilizerPredictor, FertilizerType
from disease_sub_model.disease_sb import DiseaseAnalyzer, DiseaseClassifier, DiseaseClassifierConfig
from yield_sub_model.crop_yield_sb import YieldPredictor
from weather_sub_model.weather_sb import WeatherModelManager
from ensemble_model import EnsembleModel

logger = setup_logger(logger_name= __name__)

class WeatherType(Enum):
    SUNNY = "sunny"
    RAINY = "rainy"
    CLOUDY = "cloudy"
    FOGGY = "foggy"

class CropData(BaseModel):
    State_Name: str
    District_Name: str
    Crop_Year: int
    Season: str
    Crop: str
    Area: float

@dataclass
class FertilizerInput:
    name: FertilizerType
    quantity: float

@dataclass
class InputParameters:
    location: Tuple[float]
    crop_name: str
    images: List[str]
    weather_type: WeatherType
    fertilizer: FertilizerInput

@dataclass
class FrozenModel:
    disease_model: DiseaseClassifier
    ensembled_model: EnsembleModel

class EnsembledPredictor:
    def __init__(self) -> None:
        self.disease_analyser = DiseaseAnalyzer(
            DiseaseClassifierConfig(
                image_size=224,
                batch_size=64,
                epochs=5,
                model_save_path=r"./models/disease_prediction_model_frozen.pt"
        ))
        self.fertilizer_processor = FertilizerPredictor("sample_data.csv")
        self.yield_predictor = YieldPredictor()
        self.weather_processor = WeatherModelManager()
        self.frozen_models = FrozenModel(
            disease_model= DiseaseClassifier(DiseaseClassifierConfig(
                model_save_path=r"./models/disease_prediction_model_frozen.pt"
            )),
            ensembled_model= EnsembleModel({
                "fertilizer_data": "sample_data.csv",
                "weather_data": "sample_data.csv",
                "disease_model": "./models/disease_prediction_model_frozen.pt",
                "yield_data": "sample_data.csv"
            })
        )
        initialize_frozen_models(self.frozen_models)
        pass

    def get_yield_production_estimate(self, input_parameters: InputParameters) -> float:
        disease_factor = 1 - self.disease_analyser.process(input_parameters.images)
        weather_factor = 1 - self.weather_processor.predict(input_parameters.crop_name, input_parameters.weather_type)
        fertilizer_factor = 1 - self.fertilizer_processor.predict(input_parameters.fertilizer.name, input_parameters.crop_name)
        estimated_yield = 1 - self.yield_predictor.predict(input_parameters)

        return estimated_yield * disease_factor * weather_factor * fertilizer_factor
    
    def predict_disease(self, image: Path) -> str:
        return self.frozen_models.disease_model.predict(image)

    def predict_yield(self, input_parameters: InputParameters) -> float:
        return self.frozen_models.ensembled_model.predict(input_parameters)
    
    def predict_production(self, data: CropData):
        print(f"{label_encoders}")
        print(f"input data: {data}")
        input_dict = data.dict()
        print(input_dict)
        
        # Encode categorical inputs
        for col in ["State_Name", "District_Name", "Season", "Crop"]:
            encoder = label_encoders[col]
            print("got the encoders....\n\n")
            print(f"{encoder}")
            input_dict[col] = encoder.transform([input_dict[col]])[0]

        print("reaching here...")
        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])
        print(input_df.head())

        # Predict
        prediction = model.predict(input_df)[0]
        # return {"predicted_production": round(prediction, 2)}
        return prediction
    

def initialize_frozen_models(frozen_models: FrozenModel) -> None:
    global model, label_encoders
    frozen_models.disease_model.load_model(39)
    # frozen_models.disease_model.evaluate()
    
    # with open("models/ensembled_model_frozen.pkl", "rb") as ensembled_weights:
    #     frozen_models.ensembled_model = pickle.load(ensembled_weights)
    
    # Load model and encoders
    with open("models/crop_production_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("models/label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)