import pandas as pd
import math

# Models
import xgboost as xgb

# Tuning and Cross Validation
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Importing Helper Functions
from helper_functions import compress_pickle, decompress_pickle



def build_xgboost_model(v):
    xgboost = xgb.XGBRegressor(verbosity=v)
    return xgboost


def tune_xgboost_with_cross_validation(x_train, y_train, x_test, y_test):
    xgb_tuning = build_xgboost_model(1)

    parameters = {  
        'n_estimators': [50, 100, 300],
        'learning_rate': [0.05, 0.1, 0.3],
        'max_depth': [6, 12, 24], 
        'gamma': [1, 10],
        'reg_lambda' : [0.1, 1, 10],
        'colsample_bytree' : [0.25, 0.5, 1],
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

    grid_search = GridSearchCV(estimator=xgb_tuning, param_grid=parameters, cv=skf, scoring='neg_mean_absolute_error', verbose=3)
    grid_search.fit(x_train, y_train, verbose=1)

    print("################# Tuned XGBoost Parameters #################")
    print(grid_search.best_params_)

    scores_df = pd.DataFrame(grid_search.cv_results_['params'])
    scores_df['mean_test_score'] = -grid_search.cv_results_['mean_test_score']
    scores_df['std_test_score'] = grid_search.cv_results_['std_test_score']
    scores_df['mean_fit_time'] = grid_search.cv_results_['mean_fit_time']

    scores_df.to_csv('Data_Files/Model_Files/' + 'grid_search_results_xgb.csv')
    
    tuned_xgb = grid_search.best_estimator_
    compress_pickle('Data_Files/Model_Files/', 'xgb', tuned_xgb)

    evaluate_xgboost(tuned_xgb, x_test, y_test)



def evaluate_xgboost(model, x_test, y_test):
    y_pred_xgb = model.predict(x_test)

    mae_xgb = mean_absolute_error(y_pred_xgb, y_test)
    rmse_xgb = math.sqrt(mean_squared_error(y_pred_xgb, y_test))

    error_xgb = (mae_xgb + rmse_xgb) / 2


    model_perf = pd.read_csv('Data_Files/Model_Files/model_performance.csv')

    row = model_perf[model_perf['Model'] == 'XGBoost']
    row.loc[:, 'MAE'] = mae_xgb
    row.loc[:, 'RMSE'] = rmse_xgb
    row.loc[:, 'combined_error'] = error_xgb
    model_perf.update(row)
    
    model_perf.to_csv('Data_Files/Model_Files/model_performance.csv', index=False)



def run_xgboost(x_test):
    optimal_xgb = decompress_pickle('Data_Files/Model_Files/', 'xgb')  

    data = x_test.drop(['UserID', 'Title'], axis=1)
    y_pred_xgb = optimal_xgb.predict(data)

    results_xgb = pd.DataFrame(y_pred_xgb).set_index([x_test['UserID'], x_test['Title']])

    return results_xgb

