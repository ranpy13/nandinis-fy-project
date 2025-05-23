"""
Ensemble Model for Crop Yield Prediction

This module combines multiple sub-models (fertilizer, weather, disease, and yield prediction)
to provide a comprehensive crop yield prediction based on various input parameters.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from fertilizer_sub_model.fertilizer_sb import FertilizerPredictor, FertilizerType, InputParameters as FertilizerInput
from weather_sub_model.weather_sb import WeatherModelManager, ModelName as WeatherModelName
from disease_sub_model.disease_sb import DiseaseClassifier, DiseaseClassifierConfig
from yield_sub_model.crop_yield_sb import YieldPredictor
from yield_sub_model.yield_sb import CropYieldPredictor, ModelType as YieldModelType

from utils.logger_util import setup_logger

logger = setup_logger(__name__)

@dataclass
class EnsembleInput:
    """Data class to store input parameters for ensemble prediction."""
    area: float  # Area of land in hectares
    crop_name: str  # Name of the crop
    location: Tuple[float, float]  # (latitude, longitude)
    soil_type: int  # Soil type index
    nitrogen: float  # Nitrogen content in soil
    potassium: float  # Potassium content in soil
    phosphorous: float  # Phosphorous content in soil

@dataclass
class EnsembleOutput:
    """Data class to store ensemble prediction results."""
    predicted_yield: float  # Predicted yield in tons
    confidence_score: float  # Confidence score of prediction (0-1)
    recommended_fertilizer: str  # Recommended fertilizer type
    disease_risk: float  # Disease risk percentage
    weather_suitability: float  # Weather suitability score

class EnsembleModel:
    """
    A class that combines multiple sub-models to provide comprehensive crop yield predictions.
    
    This ensemble model integrates predictions from:
    - Fertilizer recommendation model
    - Weather prediction model
    - Disease detection model
    - Crop yield prediction model
    
    Attributes:
        fertilizer_model (FertilizerPredictor): Model for fertilizer recommendations
        weather_model (WeatherModelManager): Model for weather predictions
        disease_model (DiseaseClassifier): Model for disease detection
        yield_model (YieldPredictor): Model for yield prediction
        scaler (StandardScaler): Scaler for normalizing input features
    """
    
    def __init__(self, model_paths: Dict[str, str]):
        """
        Initialize the ensemble model with all sub-models.
        
        Args:
            model_paths: Dictionary containing paths to model files and data
                Required keys:
                - fertilizer_data: Path to fertilizer training data
                - weather_data: Path to weather training data
                - disease_model: Path to disease model file
                - yield_data: Path to yield training data
        """
        try:
            # Initialize sub-models
            self.fertilizer_model = FertilizerPredictor(model_paths['fertilizer_data'])
            self.weather_model = WeatherModelManager(model_paths['weather_data'])
            self.disease_model = DiseaseClassifier(DiseaseClassifierConfig(
                model_save_path=model_paths['disease_model']
            ))
            self.yield_model = YieldPredictor()
            
            # Load and train models
            self._initialize_models()
            
            # Initialize scaler for feature normalization
            self.scaler = StandardScaler()
            
            logger.info("Ensemble model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ensemble model: {str(e)}")
            raise
    
    def _initialize_models(self) -> None:
        """Initialize and train all sub-models."""
        try:
            # Load and train fertilizer model
            self.fertilizer_model.load_data()
            self.fertilizer_model.train_models()
            
            # Load and train weather model
            self.weather_model.load_data()
            self.weather_model.train_all()
            
            # Load disease model
            self.disease_model.load_model(num_classes=38)  # Assuming 38 disease classes
            
            # Load and train yield model
            self.yield_model.load_data(self.model_paths['yield_data'])
            self.yield_model.train_model()
            
            logger.info("All sub-models initialized and trained successfully")
            
        except Exception as e:
            logger.error(f"Error initializing sub-models: {str(e)}")
            raise
    
    def _get_weather_data(self, location: Tuple[float, float]) -> Dict[str, float]:
        """
        Fetch weather data for the given location.
        
        Args:
            location: Tuple of (latitude, longitude)
            
        Returns:
            Dictionary containing weather parameters
        """
        try:
            weather_data = self.weather_model.fetch_weather(
                location[0], location[1],
                'temperature_2m', 'relative_humidity_2m', 'precipitation'
            )
            
            # Calculate average values
            return {
                'temperature': np.mean(weather_data['temperature_2m']),
                'humidity': np.mean(weather_data['relative_humidity_2m']),
                'rainfall': np.sum(weather_data['precipitation'])
            }
            
        except Exception as e:
            logger.error(f"Error fetching weather data: {str(e)}")
            raise
    
    def predict(self, input_data: EnsembleInput) -> EnsembleOutput:
        """
        Make a comprehensive prediction using all sub-models.
        
        Args:
            input_data: Input parameters for prediction
            
        Returns:
            EnsembleOutput containing all prediction results
            
        Raises:
            ValueError: If input parameters are invalid
            Exception: If prediction fails
        """
        try:
            # Get weather data
            weather_data = self._get_weather_data(input_data.location)
            
            # Prepare fertilizer input
            fertilizer_input = FertilizerInput(
                temperature=weather_data['temperature'],
                humidity=weather_data['humidity'],
                moisture=weather_data['rainfall'],
                soil_type=input_data.soil_type,
                crop_type=self._get_crop_type_index(input_data.crop_name),
                nitrogen=input_data.nitrogen,
                potassium=input_data.potassium,
                phosphorous=input_data.phosphorous
            )
            
            # Get fertilizer recommendation
            fertilizer = self.fertilizer_model.predict_best_fit(fertilizer_input)
            fertilizer_prob = self.fertilizer_model.predict(fertilizer, fertilizer_input)
            
            # Get weather suitability
            weather_suitability = self.weather_model.predict(
                input_data.crop_name,
                np.array([[
                    weather_data['temperature'],
                    weather_data['humidity'],
                    weather_data['rainfall']
                ]]),
                WeatherModelName.RANDOM_FOREST
            )
            
            # Prepare yield prediction input
            yield_input = np.array([[
                input_data.area,
                weather_data['temperature'],
                weather_data['rainfall'],
                weather_data['humidity'],
                input_data.nitrogen,
                input_data.potassium,
                input_data.phosphorous
            ]])
            
            # Get yield prediction
            predicted_yield = self.yield_model.predict(yield_input)[0]
            
            # Calculate confidence score
            confidence_score = np.mean([
                fertilizer_prob,
                weather_suitability,
                0.8  # Base confidence for yield prediction
            ])
            
            logger.info(f"Prediction completed successfully for {input_data.crop_name}")
            
            return EnsembleOutput(
                predicted_yield=predicted_yield,
                confidence_score=confidence_score,
                recommended_fertilizer=fertilizer.value if fertilizer else "Unknown",
                disease_risk=0.0,  # Placeholder - implement disease risk calculation
                weather_suitability=weather_suitability
            )
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def _get_crop_type_index(self, crop_name: str) -> int:
        """
        Convert crop name to index for fertilizer model.
        
        Args:
            crop_name: Name of the crop
            
        Returns:
            Integer index for the crop type
        """
        # This mapping should be maintained and updated based on the training data
        crop_mapping = {
            'rice': 0,
            'wheat': 1,
            'maize': 2,
            # Add more crops as needed
        }
        return crop_mapping.get(crop_name.lower(), 0)

def main():
    """Main function to demonstrate usage of the EnsembleModel class."""
    try:
        # Define model paths
        model_paths = {
            'fertilizer_data': 'FertilizerPrediction.csv',
            'weather_data': 'crop_recommendation.csv',
            'disease_model': 'plant_disease_model.pt',
            'yield_data': 'Burdwan_Crop.csv'
        }
        
        # Initialize ensemble model
        model = EnsembleModel(model_paths)
        
        # Example prediction
        input_data = EnsembleInput(
            area=1777.0,
            crop_name='rice',
            location=(22.8905, 87.7835),  # Burdwan coordinates
            soil_type=2,
            nitrogen=25.0,
            potassium=10.0,
            phosphorous=25.0
        )
        
        result = model.predict(input_data)
        
        # Log results
        logger.info("Prediction Results:")
        logger.info(f"Predicted Yield: {result.predicted_yield:.2f} tons")
        logger.info(f"Confidence Score: {result.confidence_score:.2f}")
        logger.info(f"Recommended Fertilizer: {result.recommended_fertilizer}")
        logger.info(f"Weather Suitability: {result.weather_suitability:.2f}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 