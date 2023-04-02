import pandas as pd
import pickle
import joblib
import numpy as np

import math

# Models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

#Tuning and Cross Validation
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score


def build_rf_model():
    rf = RandomForestClassifier()
    return rf


def tune_rf(x_train, y_train):
    rf_base = build_rf_model()
    param_grid = {
        'n_estimators' : [100, 200, 400],
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=3)
    grid_search.fit(x_train, y_train, verbose=0)

    tuned_rf = grid_search.best_estimator_
    tuned_rf.save('../../Data Files/Model Files/' + 'rf.pkl')

def get_pred(x_test):
    optimal_rf = pickle.load('../../Data Files/Model Files/' + 'rf.pkl')
    y_pred = optimal_rf.predict(x_test)

    return y_pred

def evaluate_rf(x_test, y_test):
    y_pred = get_pred(x_test)
    rmse_after_tuning = math.sqrt(mean_squared_error(y_test, y_pred))
    mae_after_tuning = mean_absolute_error(y_test, y_pred)

    return (rmse_after_tuning + mae_after_tuning)/2

