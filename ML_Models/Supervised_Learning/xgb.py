
# Models
import xgboost as xgb

# Tuning and Cross Validation
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Importing Helper Functions
from helper_functions import compress_pickle, decompress_pickle


def build_xgboost_model(verbosity):
    xgboost = xgb.XGBRegressor(verbosity)
    return xgboost


def tune_xgboost_with_cross_validation(x_train, y_train):
    xgb_tuning = build_xgboost_model(0)

    parameters = {  
        'max_depth': [3, 18, 1],
        'gamma': [1, 9],
        'reg_alpha' : [40, 180],
        'reg_lambda' : [0, 1],
        'colsample_bytree' : [0.5, 1],
        'min_child_weight' : [0, 10, 1],
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

    estimator = GridSearchCV(estimator=xgb_tuning, param_grid=parameters, cv=skf, scoring='neg_mean_absolute_error', verbose=3)
    estimator.fit(x_train, y_train, verbose=0)
    
    tuned_xgb = estimator.best_estimator_
    compress_pickle('Data_Files/Model_Files/', 'xgb', tuned_xgb)


def run_xgboost(x_test):
    optimal_xgb = decompress_pickle('Data_Files/Model_Files/' + 'xgb')  

    y_pred_xgb = optimal_xgb.predict(x_test)

    return y_pred_xgb

