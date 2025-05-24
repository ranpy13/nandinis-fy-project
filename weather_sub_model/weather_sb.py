import os
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from enum import Enum
import pickle
import requests
from typing import Any, Dict, List, Optional, Tuple, Union
from utils.logger_util import setup_logger

logger = setup_logger(logger_name=__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
CROP_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "crop_recommendation.csv")

class ModelName(Enum):
    DECISION_TREE = "Decision Tree"
    NAIVE_BAYES = "Naive Bayes"
    SUPPORT_VECTOR_MACHINE = "Support Vector Machine"
    LOGISTIC_REGRESSION = "Logistic Regression"
    RANDOM_FOREST = "Random Forest"
    XG_BOOST = "XG Boost"

class WeatherModelManager:
    """
    Handles training, saving, loading, and prediction for crop recommendation models.
    """
    def __init__(self, data_path: str = CROP_DATA_PATH, models_dir: str = MODELS_DIR):
        self.data_path = data_path
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        self.models = {}
        self.accuracies = {}
        # self._load_data()

    def _load_data(self):
        self.df = pd.read_csv(self.data_path)
        self.features = self.df.drop(['label'], axis=1)
        self.target = self.df['label']
        self.Xtrain, self.Xtest, self.Ytrain, self.Ytest = train_test_split(
            self.features, self.target, test_size=0.2, random_state=2
        )

    def train_all(self):
        """Train all models and save them to disk."""
        self._train_decision_tree()
        self._train_naive_bayes()
        self._train_svm()
        self._train_logistic_regression()
        self._train_random_forest()
        self._train_xgboost()

    def _save_model(self, model: Any, filename: str):
        path = os.path.join(self.models_dir, filename)
        with open(path, 'wb'):
            pickle.dump(model, open(path, 'wb'))
        logger.debug(f"Saved model to {path}")

    def _train_decision_tree(self):
        model = DecisionTreeClassifier(criterion="entropy", random_state=42, max_depth=5)
        model.fit(self.Xtrain, self.Ytrain)
        self._evaluate_and_save(model, ModelName.DECISION_TREE, "DecisionTree.pkl")

    def _train_naive_bayes(self):
        model = GaussianNB()
        model.fit(self.Xtrain, self.Ytrain)
        self._evaluate_and_save(model, ModelName.NAIVE_BAYES, "NBClassifier.pkl")

    def _train_svm(self):
        model = SVC(gamma='auto')
        model.fit(self.Xtrain, self.Ytrain)
        self._evaluate_and_save(model, ModelName.SUPPORT_VECTOR_MACHINE, None)  # Not saved by default

    def _train_logistic_regression(self):
        model = LogisticRegression(random_state=42)
        model.fit(self.Xtrain, self.Ytrain)
        self._evaluate_and_save(model, ModelName.LOGISTIC_REGRESSION, "LogisticRegression.pkl")

    def _train_random_forest(self):
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(self.Xtrain, self.Ytrain)
        self._evaluate_and_save(model, ModelName.RANDOM_FOREST, "RandomForest.pkl")

    def _train_xgboost(self):
        model = xgb.XGBClassifier()
        model.fit(self.Xtrain, self.Ytrain)
        self._evaluate_and_save(model, ModelName.XG_BOOST, "XGBoost.pkl")

    def _evaluate_and_save(self, model: Any, model_name: ModelName, filename: Optional[str]):
        predicted = model.predict(self.Xtest)
        acc = accuracy_score(self.Ytest, predicted)
        self.models[model_name] = model
        self.accuracies[model_name] = acc
        logger.info(f"Accuracy for {model_name.value}: {acc}")
        logger.info(f"Classification Report for {model_name.value}:\n{classification_report(self.Ytest, predicted)}")
        if filename:
            self._save_model(model, filename)

    def load_model(self, model_name: ModelName) -> Any:
        filename = {
            ModelName.DECISION_TREE: "DecisionTree.pkl",
            ModelName.NAIVE_BAYES: "NBClassifier.pkl",
            ModelName.LOGISTIC_REGRESSION: "LogisticRegression.pkl",
            ModelName.RANDOM_FOREST: "RandomForest.pkl",
            ModelName.XG_BOOST: "XGBoost.pkl",
        }.get(model_name)
        if not filename:
            raise ValueError(f"Model {model_name.value} is not saved or not supported for loading.")
        path = os.path.join(self.models_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file {path} does not exist.")
        with open(path, 'rb') as f:
            model = pickle.load(f)
        logger.debug(f"Loaded model {model_name.value} from {path}")
        return model

    def predict_best_fit(self, data: np.ndarray, model_name: ModelName) -> np.ndarray:
        """
        Predict using the specified model. Loads from disk if not in memory.
        """
        if model_name in self.models:
            model = self.models[model_name]
        else:
            model = self.load_model(model_name)
        return model.predict(data)
    
    def predict(self, crop_name: str, input_data: np.ndarray, model_name: ModelName) -> float:
        """
        Predict the fit probability using the specified model. Loads from disk if not in memory.
        """
        if model_name in self.models:
            model = self.models[model_name]
        else:
            mode = self.load_model(model_name)
        
        fit_percentages = model.predict_proba([input_data])[0]
        crop_names = model.classes_
        crop_fit = dict(zip(crop_names, fit_percentages)).get(crop_name, 0.0)
        return crop_fit

    def get_accuracy(self, model_name: ModelName) -> float:
        return self.accuracies.get(model_name, None)

    def get_all_accuracies(self) -> Dict[ModelName, float]:
        return self.accuracies.copy()


def fetch_weather(lat: float, long: float, *args: str) -> Dict[str, List[Any]]:
    """
    Fetch and return hourly weather data (e.g., temperature, humidity) from Open-Meteo API.
    :param lat: Latitude of the location
    :param long: Longitude of the location
    :param args: Weather variable names to fetch (e.g., 'temperature_2m', 'relative_humidity_2m')
    :return: Dictionary with time and requested weather variables
    """
    base_url = "https://api.open-meteo.com/v1/forecast"
    weather_vars = ",".join(args)
    params = {
        "latitude": lat,
        "longitude": long,
        "hourly": weather_vars,
        "forecast_days": 7,
        "timezone": "auto"
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        result = {"time": data.get("hourly", {}).get("time", [])}
        for var in args:
            result[var] = data.get("hourly", {}).get(var, [])
        return result
    else:
        logger.error(f"API request failed with status code {response.status_code}: {response.text}")
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
