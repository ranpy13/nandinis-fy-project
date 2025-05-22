"""
Fertilizer Recommendation Module

This module provides functionality for recommending fertilizers based on various
environmental and soil parameters using machine learning models.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

from utils.logger_util import setup_logger

logger = setup_logger(__name__)

class FertilizerType(Enum):
    """Enumeration of available fertilizer types."""
    NPK_10_26_26 = "10-26-26"
    NPK_14_35_14 = "14-35-14"
    NPK_17_17_17 = "17-17-17"
    NPK_20_20 = "20-20"
    NPK_28_28 = "28-28"
    DAP = "DAP"
    UREA = "Urea"

@dataclass
class InputParameters:
    """Data class to store input parameters for fertilizer prediction."""
    temperature: float
    humidity: float
    moisture: float
    soil_type: int
    crop_type: int
    nitrogen: float
    potassium: float
    phosphorous: float

@dataclass
class ModelMetrics:
    """Data class to store model performance metrics."""
    accuracy: float
    classification_report: str
    training_accuracy: float

class FertilizerPredictor:
    """
    A class to handle fertilizer recommendations using machine learning models.
    
    This class provides functionality to train and evaluate multiple machine learning
    models for fertilizer recommendation based on environmental and soil parameters.
    
    Attributes:
        data_path (Path): Path to the training data file
        df (Optional[pd.DataFrame]): The loaded training dataset
        encoders (Dict[str, LabelEncoder]): Dictionary of label encoders
        models (Dict[str, object]): Dictionary of trained models
        metrics (Dict[str, ModelMetrics]): Dictionary of model metrics
    """
    
    def __init__(self, data_path: Union[str, Path]):
        """
        Initialize the FertilizerPredictor.
        
        Args:
            data_path: Path to the CSV file containing the training data
        """
        self.data_path = Path(data_path)
        self.df: Optional[pd.DataFrame] = None
        self.encoders: Dict[str, LabelEncoder] = {}
        self.models: Dict[str, object] = {}
        self.metrics: Dict[str, ModelMetrics] = {}
        self._is_trained: bool = False
        
    def load_data(self) -> None:
        """
        Load and preprocess the training data.
        
        Raises:
            FileNotFoundError: If the data file doesn't exist
            ValueError: If required columns are missing from the data
        """
        try:
            if not self.data_path.exists():
                raise FileNotFoundError(f"Data file not found at {self.data_path}")
                
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Successfully loaded data from {self.data_path}")
            
            # Log data statistics
            logger.debug(f"Data shape: {self.df.shape}")
            logger.debug(f"Data size: {self.df.size}")
            logger.debug(f"Data description:\n{self.df.describe()}")
            
            # Identify categorical columns
            categorical_columns = self.df.select_dtypes(include=['object']).columns
            logger.debug(f"Categorical columns: {categorical_columns.tolist()}")
            
            # Encode categorical variables
            for col in categorical_columns:
                self.encoders[col] = LabelEncoder()
                self.df[col] = self.encoders[col].fit_transform(self.df[col])
                logger.debug(f"Encoded {col} with {len(self.encoders[col].classes_)} classes")
                
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def train_models(self, test_size: float = 0.2, random_state: int = 10) -> None:
        """
        Train multiple models for fertilizer prediction.
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Raises:
            ValueError: If data hasn't been loaded yet
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        try:
            # Prepare features and target
            X = self.df.drop(["FertilizerName"], axis=1)
            y = self.df["FertilizerName"]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Train KNN model
            knn_model = KNeighborsClassifier(n_neighbors=10)
            knn_model.fit(X_train, y_train)
            self.models['knn'] = knn_model
            
            # Train Decision Tree model
            dt_model = DecisionTreeClassifier(criterion='entropy', random_state=0)
            dt_model.fit(X_train, y_train)
            self.models['decision_tree'] = dt_model
            
            # Calculate metrics for both models
            for model_name, model in self.models.items():
                y_pred = model.predict(X_test)
                self.metrics[model_name] = ModelMetrics(
                    accuracy=accuracy_score(y_test, y_pred),
                    classification_report=classification_report(y_test, y_pred),
                    training_accuracy=model.score(X_train, y_train)
                )
                
                logger.info(f"{model_name.upper()} Model Metrics:")
                logger.info(f"Training Accuracy: {self.metrics[model_name].training_accuracy:.2f}")
                logger.info(f"Testing Accuracy: {self.metrics[model_name].accuracy:.2f}")
                logger.debug(f"Classification Report:\n{self.metrics[model_name].classification_report}")
            
            self._is_trained = True
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            raise
    
    def validate_input(self, params: InputParameters) -> bool:
        """
        Validate input parameters for prediction.
        
        Args:
            params: Input parameters to validate
            
        Returns:
            bool: True if parameters are valid, False otherwise
        """
        try:
            return (
                (20 <= params.temperature <= 40) and
                (40 < params.humidity < 70) and
                (20 <= params.moisture <= 70) and
                (0 <= params.soil_type <= 4) and
                (0 <= params.crop_type <= 10) and
                (0 <= params.nitrogen <= 50) and
                (0 <= params.potassium <= 20) and
                (0 <= params.phosphorous <= 50)
            )
        except Exception as e:
            logger.error(f"Error validating input parameters: {str(e)}")
            return False
    
    def predict_best_fit(self, params: InputParameters, model_type: str = 'knn') -> Optional[FertilizerType]:
        """
        Predict fertilizer type based on input parameters.
        
        Args:
            params: Input parameters for prediction
            model_type: Type of model to use ('knn' or 'decision_tree')
            
        Returns:
            FertilizerType enum value or None if prediction fails
            
        Raises:
            ValueError: If model hasn't been trained or invalid model type
        """
        if not self._is_trained:
            raise ValueError("Models not trained. Call train_models() first.")
            
        if model_type not in self.models:
            raise ValueError(f"Invalid model type. Choose from: {list(self.models.keys())}")
            
        try:
            if not self.validate_input(params):
                logger.warning("Invalid input parameters")
                return None
                
            # Prepare input array
            input_array = np.array([[
                params.temperature, params.humidity, params.moisture,
                params.soil_type, params.crop_type, params.nitrogen,
                params.potassium, params.phosphorous
            ]])
            
            # Make prediction
            prediction = self.models[model_type].predict(input_array)[0]
            
            # Map prediction to FertilizerType enum
            fertilizer_map = {
                0: FertilizerType.NPK_10_26_26,
                1: FertilizerType.NPK_14_35_14,
                2: FertilizerType.NPK_17_17_17,
                3: FertilizerType.NPK_20_20,
                4: FertilizerType.NPK_28_28,
                5: FertilizerType.DAP,
                6: FertilizerType.UREA
            }
            
            result = fertilizer_map.get(prediction)
            if result:
                logger.info(f"Predicted fertilizer: {result.value}")
                return result
            else:
                logger.warning(f"Unknown prediction value: {prediction}")
                return None
                
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def predict(self, fertilizer_name: Union[str, FertilizerType], input_data: InputParameters, model_type: str = 'knn') -> float:
        """
        Predict the probability of a specific fertilizer being suitable for given input parameters.
        
        Args:
            fertilizer_name: Name or type of fertilizer to predict probability for
            input_data: Input parameters containing environmental and soil conditions
            model_type: Type of model to use for prediction ('knn' or 'decision_tree')
            
        Returns:
            float: Probability score between 0 and 1 indicating suitability of the fertilizer
            
        Raises:
            ValueError: If models haven't been trained or invalid model type is specified
            TypeError: If fertilizer_name is not a valid FertilizerType or string
            Exception: If prediction fails due to invalid input or model error
        """
        if not self._is_trained:
            logger.error("Models not trained. Call train_models() first.")
            raise ValueError("Models not trained. Call train_models() first.")
            
        if model_type not in self.models:
            logger.error(f"Invalid model type: {model_type}. Choose from: {list(self.models.keys())}")
            raise ValueError(f"Invalid model type. Choose from: {list(self.models.keys())}")
            
        try:
            if not self.validate_input(input_data):
                logger.warning("Invalid input parameters provided")
                return None
                
            # Convert string fertilizer name to enum if needed
            if isinstance(fertilizer_name, str):
                try:
                    fertilizer_name = FertilizerType(fertilizer_name)
                except ValueError:
                    logger.error(f"Invalid fertilizer name: {fertilizer_name}")
                    raise TypeError(f"Invalid fertilizer name: {fertilizer_name}")
                
            # Prepare input array
            input_array = np.array([[
                input_data.temperature, input_data.humidity, input_data.moisture,
                input_data.soil_type, input_data.crop_type, input_data.nitrogen,
                input_data.potassium, input_data.phosphorous
            ]])
            
            logger.debug(f"Making prediction using {model_type} model for {fertilizer_name.value}")

            prediction = self.models[model_type].predict_proba(input_array)[0]

            # Map prediction to FertilizerType enum
            fertilizer_map = {
                FertilizerType.NPK_10_26_26: 0,
                FertilizerType.NPK_14_35_14: 1,
                FertilizerType.NPK_17_17_17: 2,
                FertilizerType.NPK_20_20: 3,
                FertilizerType.NPK_28_28: 4,
                FertilizerType.DAP: 5,
                FertilizerType.UREA: 6
            }

            probability = prediction[fertilizer_map.get(fertilizer_name)]
            logger.info(f"Predicted probability for {fertilizer_name.value}: {probability:.2f}")
            return probability
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise Exception(f"Prediction failed: {str(e)}")
    
    def get_model_metrics(self, model_type: str) -> ModelMetrics:
        """
        Get metrics for a specific model.
        
        Args:
            model_type: Type of model to get metrics for
            
        Returns:
            ModelMetrics object containing the model's performance metrics
            
        Raises:
            ValueError: If model hasn't been trained or invalid model type
        """
        if not self._is_trained:
            raise ValueError("Models not trained. Call train_models() first.")
            
        if model_type not in self.metrics:
            raise ValueError(f"Invalid model type. Choose from: {list(self.metrics.keys())}")
            
        return self.metrics[model_type]

def main():
    """Main function to demonstrate usage of the FertilizerPredictor class."""
    try:
        # Initialize predictor
        predictor = FertilizerPredictor("FertilizerPrediction.csv")
        
        # Load data
        predictor.load_data()
        
        # Train models
        predictor.train_models()
        
        # Example prediction
        params = InputParameters(
            temperature=30.0,
            humidity=50.0,
            moisture=40.0,
            soil_type=2,
            crop_type=5,
            nitrogen=25.0,
            potassium=10.0,
            phosphorous=25.0
        )
        
        prediction = predictor.predict(params)
        if prediction:
            logger.info(f"Recommended fertilizer: {prediction.value}")
        else:
            logger.warning("Could not make a valid prediction")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()