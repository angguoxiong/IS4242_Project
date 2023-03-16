import pandas as pd
import numpy as np

# Models
from keras.models import Sequential
from keras.layers import Dense

#Tuning
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score

x_train = pd.DataFrame()    # genre, description, duration, directors, stars, votes, user rating, movie rating
y_train = pd.DataFrame()    # ratings
x_test = pd.DataFrame()
y_test = pd.DataFrame()

def build_nn_model(num_neurons_1, num_neurons_2):
    nn = Sequential()
    nn.add(Dense(num_neurons_1, input_shape=[len(x_train.columns),], activation="relu"))
    nn.add(Dense(num_neurons_2, input_shape=[len(x_train.columns),], activation="relu"))
    nn.add(Dense(len(np.unique(y_train.values)), activation="softmax"))
    nn.compile(loss='sparse_categorical_crossentropy', optimizer='adam') # use sparse_categorical_crossentropy as the products are label-encoded as integers
    return nn


nn = build_nn_model(30, 30)
nn_history = nn.fit(x_train, y_train, batch_size=5, epochs=5, verbose=1)
y_pred_nn_prob = nn.predict(x_test)
y_pred_nn = np.argmax(y_pred_nn_prob, axis=1)


### tuning for hyperparameters
model = KerasClassifier(build_fn=build_nn_model)

param_grid = {'activation': ['relu', 'softmax'],
              'optimizer': ['adam', 'sgd'],
              'batch_size': [5, 10, 20],
              'epochs': [5, 10, 20]}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_search.fit(x_train, y_train)

optimal_clf = grid_search.best_estimator_

Y_pred = optimal_clf.predict(x_test)
acc_after_tuning = accuracy_score(y_test, Y_pred)
print("test accuracy = {:.2%}".format(acc_after_tuning))