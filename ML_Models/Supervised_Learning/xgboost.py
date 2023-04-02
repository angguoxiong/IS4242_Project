import math

# Models
import xgboost as xgb

#Tuning and Cross Validation
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV

import warnings

from helper_functions import compress_pickle, decompress_pickle

warnings.filterwarnings("ignore")


def build_xgboost_model(verbosity):
    xgboost = xgb.XGBRegressor(verbosity=verbosity)
    return xgboost


def tune_xgboost_hyperparameter_with_cross_validation(x_train, y_train):
    xgb_tuning = build_xgboost_model(0)
    parameters = {'max_depth': [3, 18, 1],
              'gamma': [1,9],
              'reg_alpha' : [40,180,],
              'reg_lambda' : [0,1],
              'colsample_bytree' : [0.5,1],
              'min_child_weight' : [0, 10, 1],
             }

    estimator = GridSearchCV(xgb_tuning, parameters, cv=3)
    estimator.fit(x_train, y_train)
    tuned_xgb = estimator.best_estimator_
    compress_pickle('../../Data_Files/Model_Files/', 'xgb', tuned_xgb)

def evaluate_xgboost(x_test, y_test):
    optimal_xgb = decompress_pickle('../../Data_Files/Model_Files/', 'xgb')

    y_pred = optimal_xgb.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))

    return 0.5 * (mae + rmse)

