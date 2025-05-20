"""
Crop Yield Prediction Model Module

This module provides functionality for training and using machine learning models
to predict crop yields based on various environmental and agricultural parameters.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from utils.logger_util import setup_logger

logger = setup_logger(logger_name=__name__)

class ModelType(Enum):
    """Enumeration of available model types."""
    DECISION_TREE = "Decision Tree"
    NAIVE_BAYES = "Naive Bayes"
    SVM = "SVM"
    LOGISTIC_REGRESSION = "Logistic Regression"
    RANDOM_FOREST = "Random Forest"

@dataclass
class ModelMetrics:
    """Data class to store model performance metrics."""
    accuracy: float
    classification_report: str
    cross_val_scores: np.ndarray

class CropYieldPredictor:
    """
    A class to handle crop yield prediction using various machine learning models.
    
    This class provides functionality to train and evaluate multiple machine learning
    models for crop yield prediction based on environmental and agricultural parameters.
    """
    
    def __init__(self, data_path: Union[str, Path]):
        """
        Initialize the CropYieldPredictor.

        Args:
            data_path: Path to the CSV file containing the training data
        """
        self.data_path = Path(data_path)
        self.df: Optional[pd.DataFrame] = None
        self.features: Optional[pd.DataFrame] = None
        self.target: Optional[pd.Series] = None
        self.models: Dict[ModelType, object] = {}
        self.metrics: Dict[ModelType, ModelMetrics] = {}
        
    def load_data(self) -> None:
        """Load and preprocess the training data."""
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Successfully loaded data from {self.data_path}")
            
            # Extract features and target
            self.features = self.df[['Crop_Year', 'Area', 'Temparetue', 
                                   'Rainfall', 'Humidity', 'Sun hours']]
            self.target = self.df['Crop']
            
            logger.debug(f"Data shape: {self.df.shape}")
            logger.debug(f"Features: {self.features.columns.tolist()}")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def train_test_split(self, test_size: float = 0.2, 
                        random_state: int = 2) -> Tuple[np.ndarray, np.ndarray, 
                                                      np.ndarray, np.ndarray]:
        """
        Split the data into training and testing sets.

        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility

        Returns:
            Tuple containing X_train, X_test, y_train, y_test
        """
        if self.features is None or self.target is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        return train_test_split(
            self.features, self.target,
            test_size=test_size,
            random_state=random_state
        )

    def train_model(self, model_type: ModelType) -> None:
        """
        Train a specific model type.

        Args:
            model_type: Type of model to train
        """
        if self.features is None or self.target is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        X_train, X_test, y_train, y_test = self.train_test_split()
        
        model_map = {
            ModelType.DECISION_TREE: DecisionTreeClassifier(
                criterion="entropy",
                random_state=2,
                max_depth=5
            ),
            ModelType.NAIVE_BAYES: GaussianNB(),
            ModelType.SVM: SVC(gamma='auto'),
            ModelType.LOGISTIC_REGRESSION: LogisticRegression(random_state=2),
            ModelType.RANDOM_FOREST: RandomForestClassifier(
                n_estimators=20,
                random_state=0
            )
        }
        
        model = model_map[model_type]
        logger.info(f"Training {model_type.value}...")
        
        try:
            model.fit(X_train, y_train)
            self.models[model_type] = model
            
            # Calculate metrics
            predictions = model.predict(X_test)
            accuracy = model.score(X_test, y_test)
            report = classification_report(y_test, predictions)
            cv_scores = cross_val_score(model, self.features, self.target, cv=5)
            
            self.metrics[model_type] = ModelMetrics(
                accuracy=accuracy,
                classification_report=report,
                cross_val_scores=cv_scores
            )
            
            logger.info(f"{model_type.value} trained successfully")
            logger.info(f"Accuracy: {accuracy:.2f}")
            logger.debug(f"Cross-validation scores: {cv_scores}")
            
        except Exception as e:
            logger.error(f"Error training {model_type.value}: {str(e)}")
            raise

    def predict(self, features: np.ndarray, model_type: ModelType) -> np.ndarray:
        """
        Make predictions using a trained model.

        Args:
            features: Input features for prediction
            model_type: Type of model to use for prediction

        Returns:
            Array of predictions
        """
        if model_type not in self.models:
            raise ValueError(f"Model {model_type.value} not trained yet")
            
        try:
            predictions = self.models[model_type].predict(features)
            logger.info(f"Made predictions using {model_type.value}")
            return predictions
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise

    def get_model_metrics(self, model_type: ModelType) -> ModelMetrics:
        """
        Get metrics for a specific model.

        Args:
            model_type: Type of model to get metrics for

        Returns:
            ModelMetrics object containing the model's performance metrics
        """
        if model_type not in self.metrics:
            raise ValueError(f"Model {model_type.value} not trained yet")
        return self.metrics[model_type]

def main():
    """Main function to demonstrate usage of the CropYieldPredictor class."""
    try:
        # Initialize predictor
        predictor = CropYieldPredictor('./Burdwan_Crop.csv')
        predictor.load_data()
        
        # Train all models
        for model_type in ModelType:
            predictor.train_model(model_type)
        
        # Example prediction
        sample_data = np.array([[2022, 1777, 31, 1100, 80, 9.3]])
        prediction = predictor.predict(sample_data, ModelType.RANDOM_FOREST)
        logger.info(f"Sample prediction: {prediction}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()