import pandas as pd
import math

# Models
from sklearn.ensemble import RandomForestRegressor

#Tuning and Cross Validation
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Importing Helper Functions
from helper_functions import compress_pickle, decompress_pickle



def build_rf_model():
    rf = RandomForestRegressor()
    return rf


def tune_random_forest_with_cross_validation(x_train, y_train, x_test, y_test):
    rf_base = build_rf_model()

    param_grid = {
        'n_estimators' : [100, 200, 400],
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

    grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=skf, verbose=3)
    grid_search.fit(x_train, y_train)

    print("################# Tuned RandomForest Parameters #################")
    print(grid_search.best_params_)

    scores_df = pd.DataFrame(grid_search.cv_results_['params'])
    scores_df['mean_test_score'] = -grid_search.cv_results_['mean_test_score']
    scores_df['std_test_score'] = grid_search.cv_results_['std_test_score']
    scores_df['mean_fit_time'] = grid_search.cv_results_['mean_fit_time']

    scores_df.to_csv('Data_Files/Model_Files/' + 'grid_search_results_rf.csv')

    tuned_rf = grid_search.best_estimator_
    compress_pickle('Data_Files/Model_Files/', 'rf', tuned_rf)
    
    evaluate_random_forest(tuned_rf, x_test, y_test)



def evaluate_random_forest(model, x_test, y_test):
    y_pred_rf = model.predict(x_test)

    mae_rf = mean_absolute_error(y_pred_rf, y_test)
    rmse_rf = math.sqrt(mean_squared_error(y_pred_rf, y_test))

    error_rf = (mae_rf + rmse_rf) / 2


    model_perf = pd.read_csv('Data_Files/Model_Files/model_performance.csv')

    row = model_perf[model_perf['Model'] == 'RandomForest']
    row.loc[:, 'MAE'] = mae_rf
    row.loc[:, 'RMSE'] = rmse_rf
    row.loc[:, 'combined_error'] = error_rf
    model_perf.update(row)
    
    model_perf.to_csv('Data_Files/Model_Files/model_performance.csv', index=False)



def run_random_forest(x_test):
    optimal_rf = decompress_pickle('Data_Files/Model_Files/', 'rf')  

    data = x_test.drop(['UserID', 'Title'], axis=1)
    y_pred_rf = optimal_rf.predict(data)

    results_rf = pd.DataFrame(y_pred_rf).set_index([x_test['UserID'], x_test['Title']])

    return results_rf
