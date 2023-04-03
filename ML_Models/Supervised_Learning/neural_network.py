import pandas as pd
import math

# Models
from keras.models import Sequential, load_model
from keras.layers import Dense

# Tuning and Cross Validation
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error




def build_neural_network_model(num_neurons_1, num_neurons_2, input_dimensions, activation_fn, optimizer_fn):
    nn = Sequential()
    nn.add(Dense(num_neurons_1, input_shape=input_dimensions, activation=activation_fn))
    nn.add(Dense(num_neurons_2, input_shape=input_dimensions, activation=activation_fn))
    nn.add(Dense(1, activation='linear'))
    nn.compile(loss='mean_absolute_error', optimizer=optimizer_fn, metrics=['mean_absolute_error']) 
    return nn



def tune_neural_network_with_cross_validation(x_train, y_train, x_test, y_test):
    model = KerasRegressor(build_fn=build_neural_network_model)

    param_grid = {  
        'num_neurons_1': [30, 50],
        'num_neurons_2': [10, 30, 50],
        'activation_fn': ['tanh', 'softplus', 'relu'],
        'optimizer_fn': ['adam', 'sgd'],
        'input_dimensions': [(len(x_train.columns),)],
        'batch_size': [16, 64],
        'epochs': [10]
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=skf, verbose=3)
    grid_search.fit(x_train, y_train, verbose=0)

    print("################# Tuned Neural Network Parameters #################")
    print(grid_search.best_params_)

    tuned_nn = grid_search.best_estimator_
    tuned_nn.model.save('Data_Files/Model_Files/' + 'nn.h5')

    evaluate_neural_network(tuned_nn, x_test, y_test)



def evaluate_neural_network(model, x_test, y_test):
    y_pred_nn = model.predict(x_test)

    mae_nn = mean_absolute_error(y_pred_nn, y_test)
    rmse_nn = math.sqrt(mean_squared_error(y_pred_nn, y_test))

    error_nn = (mae_nn + rmse_nn) / 2


    model_perf = pd.read_csv('Data_Files/Model_Files/model_performance.csv')

    row = model_perf[model_perf['Model'] == 'NeuralNetwork']
    row.loc[:, 'MAE'] = mae_nn
    row.loc[:, 'RMSE'] = rmse_nn
    row.loc[:, 'combined_error'] = error_nn
    model_perf.update(row)
    
    model_perf.to_csv('Data_Files/Model_Files/model_performance.csv', index=False)



def run_neural_network(x_test):
    optimal_nn = load_model('Data_Files/Model_Files/' + 'nn.h5')
    
    data = x_test.drop(['UserID', 'Title'], axis=1)
    y_pred_nn = optimal_nn.predict(data)

    results_nn = pd.DataFrame(y_pred_nn).set_index([x_test['UserID'], x_test['Title']])

    return results_nn
