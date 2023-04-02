
# Models
from sklearn.ensemble import RandomForestRegressor

#Tuning and Cross Validation
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Importing Helper Functions
from helper_functions import compress_pickle, decompress_pickle



def build_rf_model():
    rf = RandomForestRegressor()
    return rf


def tune_random_forest_with_cross_validation(x_train, y_train):
    rf_base = build_rf_model()

    param_grid = {
        'n_estimators' : [100, 200, 400],
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

    grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=skf, verbose=3)
    grid_search.fit(x_train, y_train, verbose=0)

    tuned_rf = grid_search.best_estimator_
    compress_pickle('Data_Files/Model_Files/', 'rf', tuned_rf)


def run_random_forest(x_test):
    optimal_rf = decompress_pickle('Data_Files/Model_Files/' + 'rf')  

    y_pred_rf = optimal_rf.predict(x_test)

    return y_pred_rf

