from pathlib import Path
from typing import Optional
from numpy import np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score

from utils.logger_util import setup_logger
logger = setup_logger(__name__)

class YieldPredictor:
    def __init__(self) -> None:
        self.training_data: Optional[pd.DataFrame] = None
        self.x_train, self.y_train = None, None
        self.X_test, self.Y_test = None, None
        self.model = LinearRegression()
        self.features = None
        self.target = None
    
    def load_data(self, data_path: Path):
        self.training_data = pd.read_csv(data_path)
        logger.info(f"Successfully loaded data from {data_path}")

        self.features = self.training_data[['Crop_Year', 'Area', 'Temperature',
                                            'Rainfall', 'Humidity', 'Sun hours']]
        self.target = self.training_data['Production']

        logger.debug(f"Data Shape: {self.training_data.shape}")
        logger.debug(f"Features: {self.training_data.columns.tolist()}")

    def split_training_data(self, test_size: float = 0.2):
        self.x_train, self.y_train, self.X_test, self.Y_test = train_test_split(self.features, self.target, test_size= test_size)
    
    def train_model(self):
        self.model.fit(self.x_train, self.y_train)
        predictions = self.model.predict(self.X_test)
        accuracy = self.model.score(self.X_test, self.Y_test)
        report = classification_report(self.Y_test, predictions)
        cv_score = cross_val_score(self.model, self.features, self.target, cv = 5)

        logger.info("Model trained successfully")
        logger.info(f"Accuracy: {accuracy:.2f}")
        logger.debug(f"Cross-validation scores: {cv_score}")
    
    def predict(self, features: np.ndarray):
        predictions = self.model.predict(features)
        logger.info(f"Made predictions")
        return predictions
    


    