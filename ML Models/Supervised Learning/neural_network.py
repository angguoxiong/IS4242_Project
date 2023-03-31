import math

# Models
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense

#Tuning and Cross Validation
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error



def build_neural_network_model(num_neurons_1, num_neurons_2, activation_fn, optimizer_fn, input_dimensions):
    nn = Sequential()
    nn.add(Dense(num_neurons_1, input_shape=input_dimensions, activation=activation_fn))
    nn.add(Dense(num_neurons_2, input_shape=input_dimensions, activation=activation_fn))
    nn.add(Dense(1, activation='linear'))
    nn.compile(loss='mean_absolute_error', optimizer=optimizer_fn, metrics=['mean_absolute_error']) 
    return nn


def tune_neural_network_hyperparameter_with_cross_validation(x_train, y_train):
    model = KerasRegressor(build_fn=build_neural_network_model)

    param_grid = {'num_neurons_1': [30, 50],
                'num_neurons_2': [10, 30, 50],
                'activation_fn': ['tanh', 'softplus', 'relu'],
                'optimizer_fn': ['adam', 'sgd'],
                'input_dimensions': [len(x_train.columns),],
                'batch_size': [16, 64],
                'epochs': [10]}

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=skf, verbose=3)
    grid_search.fit(x_train, y_train, verbose=0)

    tuned_nn = grid_search.best_estimator_
    tuned_nn.save('../../Data Files/Model Files/' + 'nn.h5')


def evaluate_neural_network(x_test, y_test):
    optimal_nn = load_model('../../Data Files/Model Files/' + 'nn.h5')  

    y_pred = optimal_nn.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))

    return 0.5 * (mae + rmse)

