import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from __future__ import print_function
import pickle
import requests
import warnings
from enum import Enum
warnings.filterwarnings('ignore')

from utils.logger_util import setup_logger
logger = setup_logger(logger_name= __name__)


class ModelName(Enum):
    DECISION_TREE = "Decision Tree"
    NAIVE_BAYES = "Naive Bayes"
    SUPPORT_VECTOR_MACHINE = "Support Vector Machine"
    LOGISTIC_REGRESSION = "Logistic Regression"
    RANDOM_FOREST = "Random Forest"
    XG_BOOST = "XG Boost"


df = pd.read_csv('crop_recommendation.csv')
sns.heatmap(df.corr(), annot= True)


# Separating feature and target lables
features = df.drop(['label'])
target = df['label']
labels = df['label']

acc = []
model = []

Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, target, test_size= 0.2, random_state = 2)


# Decision Tree
DecisionTree = DecisionTreeClassifier(criterion= "entropy", random_state= 42, max_depth= 5)

DecisionTree.fit(Xtrain, Ytrain)

precicted_values = DecisionTree.predict(Xtest)
x = accuracy_score(Ytest, precicted_values)
logger.info(f"Accuracy with Decision Tree: {x}")

acc.append(x)
model.append(ModelName.DECISION_TREE)

logger.info(f"Classification Report: {classification_report(Ytest, precicted_values)}")

score = cross_val_score(DecisionTree, features, target, cv= 5)
logger.info(f"Cross Validation Score for Decision Tree: {score}")

# Saving trained Decision Tree model
logger.debug("Saving pickel file for Decision Tree classifier...\n")
DT_pkl_filename = "./models/DecisionTree.pkl"
DT_model_pkl = open(DT_pkl_filename)
pickle.dump(DecisionTree, DT_model_pkl)
DT_model_pkl.close()
logger.debug("Saved pickle file for Decision Tree Classifier.\n")


# Gaussain Naive Bayes

NaiveBayes = GaussianNB()
NaiveBayes.fit(Xtrain, Ytrain)

predicted_values = NaiveBayes.predict(Xtest)
x = accuracy_score(Ytest, predicted_values)
logger.info(f"Accuracy score for Naive Bayes: {x}")

acc.append(x)
model.append(ModelName.NAIVE_BAYES)

logger.info(f"Classfication Report for Naive Bayes: {classification_report(Ytest, precicted_values)}")

score = cross_val_score(NaiveBayes, features, target, cv= 5)
logger.info("Cross Validation Score for Naive Bayes: {}", score)

# Saving pickle file
logger.debug("Saving pickle file for Naive Bayes calssifier...\n")
NB_pkl_filename = "./models/NBClassifier.pkl"
NB_model_pkl = open(NB_pkl_filename, 'wb')
pickle.dump(NaiveBayes, NB_model_pkl)
NB_model_pkl.close()
logger.debug("Saving complete for Naive Bayes calssifer.\n")


# Support Vector Machines
SVM = SVC(gamma= 'auto')
SVM.fit(Xtrain, Ytrain)

predicted_values = SVM.predict(Xtest)
x = accuracy_score(Ytest, predicted_values)
logger.info(f"Accuracy Score for Support Vector Machine: {x}")

acc.append(x)
model.append('SVM')
logger.info(f"Classification Report for Support Vector Machine: {classification_report(Ytest, precicted_values)}")

score = cross_val_score(SVM, features, target, cv= 5)
logger.info(f"Cross Validation score for SVM: {score}")


# Logistic Regression
LogReg = LogisticRegression(random_state = 42)
LogReg.fit(Xtrain, Ytrain)

predicted_values = LogReg.predict(Xtest)
x = accuracy_score(Ytest, predicted_values)
logger.info(f"Accuracy Score for Logistic Regression: {x}")

acc.append(x)
model.append(ModelName.LOGISTIC_REGRESSION)
logger.info(f"Classfication Report for Logistic Regression: {classification_report(Ytest, predicted_values)}")

score = cross_val_score(LogReg, features, target, cv= 5)
logger.info(f"Cross Validation Score for Logistic Regression: {score}")

# Saving pickle file
logger.debug("Saving pickle file for Logistic Regression...\n")
LR_pkl_filename = "./models/LogisticRegression.pkl"
LR_model_pkl = open(LR_pkl_filename, 'wb')
pickle.dump(LogReg, LR_model_pkl)
LR_model_pkl.close()
logger.debug("Saved pickle file for Logistic Regression.\n")


# Random Forest
RandomForest = RandomForestClassifier(n_estimators= 20, random_state= 42)
RandomForest.fit(Xtrain, Ytrain)

predicted_values = RandomForest.predict(Xtest)

x = accuracy_score(Ytest, predicted_values)
logger.info(f"Accuracy Score for Random Forest: {x}")

acc.append(x)
model.append(ModelName.RANDOM_FOREST)

logger.info(f"Classification Report for Random Forest: {classification_report(Ytest, predicted_values)}")

# Saving to pickle file
logger.debug("Saving pickle fiel for Random Forest...\n")
RF_pkl_filename = "./models/RandomForest.pkl"
RF_model_pkl = open(RF_pkl_filename, 'wb')
pickle.dump(RandomForest, RF_model_pkl)
RF_model_pkl.close()
logger.debug("Saved pickle file for Random Forest Classifer.\n")



# XG Boost
XB = xgb.XBGClassifier()
XB.fit(Xtrain, Ytrain)

precicted_values = XB.predict(Xtest)

x = accuracy_score(Ytest, predicted_values)
logger.info(f"Accuracy Score for XG-Boost: {x}")

acc.append(x)
model.append(ModelName.XG_BOOST)

logger.info(f"Classification Report for XG Boost: {classification_report(Ytest, precicted_values)}")

score = cross_val_score(XB, features, target, cv= 5)
logger.info(f"Cross Validation Score for XG boost: {score}")

# Saving to pickle file
logger.debug("Saving pickle file for XG Boost...\n")
XB_pkl_filename = "./models/XGBoost.pkl"
XB_model_pkl = open(XB_pkl_filename, 'wb')
pickle.dump(XB, XB_model_pkl)
XB_model_pkl.close()
logger.debug("Saved pickle file for XG Boost.\n")


def make_prediction(data: np.array, model: ModelName):
    assert(data is not None and data.size() != 0, "Empty Data")
    
    match model:
        case ModelName.DECISION_TREE:
            return DecisionTree.predict(data)
        case ModelName.SUPPORT_VECTOR_MACHINE:
            return SVM.predict(data)
        case ModelName.LOGISTIC_REGRESSION:
            return LogReg.predict(data)
        case ModelName.NAIVE_BAYES:
            return NaiveBayes.predict(data)
        case ModelName.RANDOM_FOREST:
            return RandomForest.predict(data)
        case ModelName.XG_BOOST:
            return XB.predict(data)
        case _:
            logger.error("Undefine model!!")
    return
        

def fetch_weather(lat: float, long: float, *args):
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
        logger.error("API request failed!!")
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")


if __name__ == "__main__":
    plt.figure(figsize= (10, 5), dpi= 100)
    plt.title("Accuracy Comparision")
    plt.xlabel("Accuracy")
    plt.ylabel("Algorithm")
    sns.barplot(x= acc, y= model, palette= "dark")

    accuracy_models = dict(zip(model, acc))
