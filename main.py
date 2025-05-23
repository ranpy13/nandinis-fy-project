from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import pickle
import torch
import torch.nn as nn
from typing import List, Tuple
from utils.logger_util import setup_logger
from fertilizer_sub_model.fertilizer_sb import FertilizerPredictor, FertilizerType
from disease_sub_model.disease_sb import DiseaseAnalyzer, DiseaseClassifier
from yield_sub_model.crop_yield_sb import YieldPredictor
from weather_sub_model.weather_sb import WeatherModelManager
from ensemble_model import EnsembleModel

logger = setup_logger(__name__)

class WeatherType(Enum):
    SUNNY = "sunny"
    RAINY = "rainy"
    CLOUDY = "cloudy"
    FOGGY = "foggy"

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
        self.disease_analyser = DiseaseAnalyzer()
        self.fertilizer_processor = FertilizerPredictor()
        self.yield_predictor = YieldPredictor()
        self.weather_processor = WeatherModelManager()
        self.frozen_models = FrozenModel
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
    

def initialize_frozen_models(frozen_models: FrozenModel) -> None:
    frozen_models.disease_model.load_model(1)
    frozen_models.disease_model.evaluate()
    
    with open("models/ensembled_model_frozen.pkl", "rb") as ensembled_weights:
        frozen_models.ensembled_model = pickle.load(ensembled_weights)