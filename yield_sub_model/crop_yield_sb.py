"""
Crop Yield Prediction Module

This module provides functionality for predicting crop yields using linear regression
based on various environmental and agricultural parameters.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score

from utils.logger_util import setup_logger

logger = setup_logger(__name__)

@dataclass
class ModelMetrics:
    """Data class to store model performance metrics."""
    r2_score: float
    mean_squared_error: float
    cross_val_scores: np.ndarray

class YieldPredictor:
    """
    A class to handle crop yield prediction using linear regression.
    
    This class provides functionality to train and evaluate a linear regression model
    for crop yield prediction based on environmental and agricultural parameters.
    
    Attributes:
        training_data (Optional[pd.DataFrame]): The loaded training dataset
        features (Optional[pd.DataFrame]): Selected features for prediction
        target (Optional[pd.Series]): Target variable (Production)
        model (LinearRegression): The trained linear regression model
        metrics (Optional[ModelMetrics]): Model performance metrics
    """
    
    def __init__(self) -> None:
        """Initialize the YieldPredictor with default values."""
        self.training_data: Optional[pd.DataFrame] = None
        self.features: Optional[pd.DataFrame] = None
        self.target: Optional[pd.Series] = None
        self.model: LinearRegression = LinearRegression()
        self.metrics: Optional[ModelMetrics] = None
        self._is_trained: bool = False
        
    def load_data(self, data_path: Union[str, Path]) -> None:
        """
        Load and preprocess the training data.
        
        Args:
            data_path: Path to the CSV file containing the training data
            
        Raises:
            FileNotFoundError: If the data file doesn't exist
            ValueError: If required columns are missing from the data
        """
        try:
            data_path = Path(data_path)
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found at {data_path}")
                
            self.training_data = pd.read_csv(data_path)
            logger.info(f"Successfully loaded data from {data_path}")
            
            required_columns = ['Crop_Year', 'Area', 'Temperature', 
                              'Rainfall', 'Humidity', 'Sun hours', 'Production']
            
            missing_columns = [col for col in required_columns 
                             if col not in self.training_data.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            self.features = self.training_data[required_columns[:-1]]
            self.target = self.training_data['Production']
            
            logger.debug(f"Data shape: {self.training_data.shape}")
            logger.debug(f"Features: {self.features.columns.tolist()}")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def split_training_data(self, test_size: float = 0.2, 
                           random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, 
                                                          np.ndarray, np.ndarray]:
        """
        Split the data into training and testing sets.
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple containing X_train, X_test, y_train, y_test
            
        Raises:
            ValueError: If data hasn't been loaded yet
        """
        if self.features is None or self.target is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                self.features, self.target,
                test_size=test_size,
                random_state=random_state
            )
            
            logger.info(f"Data split into training ({len(X_train)} samples) and "
                       f"testing ({len(X_test)} samples) sets")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise
    
    def train_model(self, test_size: float = 0.2, random_state: int = 42) -> None:
        """
        Train the linear regression model.
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Raises:
            ValueError: If data hasn't been loaded yet
        """
        if self.features is None or self.target is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        try:
            X_train, X_test, y_train, y_test = self.split_training_data(
                test_size=test_size,
                random_state=random_state
            )
            
            logger.info("Training linear regression model...")
            self.model.fit(X_train, y_train)
            
            # Calculate metrics
            predictions = self.model.predict(X_test)
            r2 = r2_score(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            cv_scores = cross_val_score(self.model, self.features, self.target, cv=5)
            
            self.metrics = ModelMetrics(
                r2_score=r2,
                mean_squared_error=mse,
                cross_val_scores=cv_scores
            )
            
            self._is_trained = True
            
            logger.info("Model trained successfully")
            logger.info(f"R² Score: {r2:.2f}")
            logger.info(f"Mean Squared Error: {mse:.2f}")
            logger.debug(f"Cross-validation scores: {cv_scores}")
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            features: Input features for prediction
            
        Returns:
            Array of predictions
            
        Raises:
            ValueError: If model hasn't been trained yet
        """
        if not self._is_trained:
            raise ValueError("Model not trained. Call train_model() first.")
            
        try:
            predictions = self.model.predict(features)
            logger.info(f"Made predictions for {len(features)} samples")
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def get_model_metrics(self) -> ModelMetrics:
        """
        Get the model's performance metrics.
        
        Returns:
            ModelMetrics object containing the model's performance metrics
            
        Raises:
            ValueError: If model hasn't been trained yet
        """
        if not self._is_trained or self.metrics is None:
            raise ValueError("Model not trained. Call train_model() first.")
        return self.metrics

def main():
    """Main function to demonstrate usage of the YieldPredictor class."""
    try:
        # Initialize predictor
        predictor = YieldPredictor()
        
        # Load data
        predictor.load_data('./Burdwan_Crop.csv')
        
        # Train model
        predictor.train_model()
        
        # Example prediction
        sample_data = np.array([[2022, 1777, 31, 1100, 80, 9.3]])
        prediction = predictor.predict(sample_data)
        logger.info(f"Sample prediction: {prediction}")
        
        # Get metrics
        metrics = predictor.get_model_metrics()
        logger.info(f"Model R² Score: {metrics.r2_score:.2f}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
    


    