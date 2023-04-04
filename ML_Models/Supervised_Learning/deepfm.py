import pandas as pd
import math

from sklearn.preprocessing import LabelEncoder
from deepctr_torch.inputs import SparseFeat, get_feature_names

# Models
import torch
from deepctr_torch.models import DeepFM

# Tuning 
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import mean_absolute_error, mean_squared_error
from helper_functions import compress_pickle, decompress_pickle



def build_deepfm_model(linear_feature_columns, dnn_feature_columns,
                       dnn_hidden_units, dnn_dropout):
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    model = DeepFM(linear_feature_columns, dnn_feature_columns,
                   dnn_hidden_units=dnn_hidden_units, dnn_dropout=dnn_dropout,
                   task='regression', device=device)

    model.compile('adam', 'mse', metrics=['mse'])
    return model


def process_x(data):
    for x in data:
        if x.find("."):
            old_column = x
            new_column = x.replace(".", "_")
            data.rename(columns={old_column: new_column}, inplace = True)
    
    sparse_features = data.columns.tolist()

    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
        
    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique()) for feat in sparse_features]
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    model_input = {name: data[name] for name in feature_names}
    
    return model_input


def label_encode_for_deepfm(x_train, x_test, y_train, y_test):

    x_data = pd.concat([x_train, x_test])

    sparse_data = x_data.iloc[:, 14:]

    for column in sparse_data:
        if column.find(".") > -1:
            old_column = column
            new_column = column.replace(".", "_")
            sparse_data.rename(columns={old_column: new_column}, inplace = True)

    for x in x_train:
        if x.find("."):
            old_column = x
            new_column = x.replace(".", "_")
            x_train.rename(columns={old_column: new_column}, inplace = True)

    for x in x_test:
        if x.find("."):
            old_column = x
            new_column = x.replace(".", "_")
            x_test.rename(columns={old_column: new_column}, inplace = True)

    target = ['User_Rating']
    
    sparse_features = sparse_data.columns.tolist()
    
    for feat in sparse_features:
        lbe = LabelEncoder()
        sparse_data[feat] = lbe.fit_transform(sparse_data[feat])
        
    fixlen_feature_columns = [SparseFeat(feat, sparse_data[feat].nunique()) for feat in sparse_features]
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    
    train_model_input = {name: x_train[name] for name in feature_names}
    test_model_input = {name: x_test[name] for name in feature_names}
    
    return train_model_input, test_model_input, y_train, y_test, linear_feature_columns, dnn_feature_columns


def tune_deepfm_with_cross_validation(x_train, y_train, x_test, y_test):
    encoded_x_train, encoded_x_test, y_train, y_test, lfc, dfc = label_encode_for_deepfm(x_train, x_test, y_train, y_test)

    def objective_function(param_space):
        dnn_hidden_units = param_space["dnn_hidden_units"]
        dnn_dropout = param_space["dnn_dropout"]

        model = build_deepfm_model(lfc, dfc, dnn_hidden_units, dnn_dropout)

        pred_ans = model.predict(encoded_x_test, 256)
        mse = round(mean_squared_error(y_test['User_Rating'].values, pred_ans), 4)

        return {
            "loss": mse,
            "status": STATUS_OK,
            "dnn_hidden_units": dnn_hidden_units,
            "dnn_dropout": dnn_dropout
        }

    trials = Trials()
    param_space = {
        "dnn_hidden_units": hp.choice("dnn_hidden_units", [(128, 128), (256, 256)]),
        "dnn_dropout": hp.choice("dnn_dropout", [0, 0.1,])}

    best = fmin(fn=objective_function, space=param_space,
                algo=tpe.suggest, max_evals=20, trials=trials)

    best_estimator = build_deepfm_model(lfc, dfc, [best['dnn_hidden_units']], best['dnn_dropout'])
    compress_pickle('Data_Files/Model_Files/', 'deepfm', best_estimator)

    evaluate_deepfm(best_estimator, encoded_x_test, y_test)



def evaluate_deepfm(model, x_test, y_test):
    y_pred_dfm = model.predict(x_test)

    mae_dfm = mean_absolute_error(y_pred_dfm, y_test)
    rmse_dfm = math.sqrt(mean_squared_error(y_pred_dfm, y_test))

    error_dfm = (mae_dfm + rmse_dfm) / 2

    model_perf = pd.read_csv('Data_Files/Model_Files/model_performance.csv')

    row = model_perf[model_perf['Model'] == 'DeepFM']
    row.loc[:, 'MAE'] = mae_dfm
    row.loc[:, 'RMSE'] = rmse_dfm
    row.loc[:, 'combined_error'] = error_dfm
    model_perf.update(row)
    
    model_perf.to_csv('Data_Files/Model_Files/model_performance.csv', index=False)
    

def run_deepfm(x_test):
    optimal_deepfm = decompress_pickle('Data_Files/Model_Files/', 'deepfm')

    data = x_test.drop(['UserID', 'Title'], axis=1)
    
    data = process_x(data)
    
    y_pred_dfm = optimal_deepfm.predict(data)

    results_dfm = pd.DataFrame(y_pred_dfm).set_index([x_test['UserID'], x_test['Title']])

    return results_dfm

