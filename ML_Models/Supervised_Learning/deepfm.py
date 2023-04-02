
from sklearn.preprocessing import LabelEncoder
from deepctr_torch.inputs import SparseFeat, get_feature_names

# Models
import torch
from deepctr_torch.models import DeepFM

# Tuning 
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
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


def label_encode_for_deepfm(x_train, x_test, y_train, y_test):
    data = x_train.append(x_test)
    sparse_data = data.iloc[:, 5:]

    sparse_data = sparse_data.loc[:, sparse_data.columns != 'children']
    sparse_data = sparse_data.loc[:, sparse_data.columns != 'clear']
    sparse_data = sparse_data.loc[:, sparse_data.columns != 'forward']
    sparse_data = sparse_data.loc[:, sparse_data.columns != 'half']
    sparse_data = sparse_data.loc[:, sparse_data.columns != 'pop']
    sparse_data = sparse_data.loc[:, sparse_data.columns != 'train']

    sparse_features = list(sparse_data.columns)
    target = ['User_Rating']
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    fixlen_feature_columns = [SparseFeat(feat,
                                         data[feat].nunique()) for feat in sparse_features]
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    lbe_y_train = LabelEncoder()
    y_train['User_Rating'] = lbe_y_train.fit_transform(y_train[target])
    lbe_y_test = LabelEncoder()
    y_test['User_Rating'] = lbe_y_test.fit_transform(y_test[target])

    x_train = {name: x_train[name] for name in feature_names}

    x_test = {name: x_test[name] for name in feature_names}
    return x_train, x_test, y_train, y_test, linear_feature_columns, dnn_feature_columns


def tune_deepfm_with_cross_validation(x_train, y_train, x_test, y_test):
    encoded_x_train, encoded_x_test, encoded_y_train, encoded_y_test, lfc, dfc = label_encode_for_deepfm(x_train,
                                                                                                         x_test,
                                                                                                         y_train,
                                                                                                         y_test)

    def objective_function(param_space):
        dnn_hidden_units = param_space["dnn_hidden_units"]
        dnn_dropout = param_space["dnn_dropout"]

        model = build_deepfm_model(lfc, dfc, dnn_hidden_units, dnn_dropout)

        history = model.fit(x_train, y_train['User_Rating'].values,
                            batch_size=32, epochs=10, verbose=2,
                            validation_split=0.2)
        pred_ans = model.predict(x_test, 256)
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
        "dnn_dropout": hp.choice("dnn_dropout", [0, 0.1, 0.5, 1])}

    best = fmin(fn=objective_function, space=param_space,
                algo=tpe.suggest, max_evals=20, trials=trials)
    best_estimator = build_deepfm_model(lfc, dfc, best['dnn_hidden_units'], best['dnn_dropout'])
    compress_pickle('../../Data_Files/Model_Files/', 'deepfm', best_estimator)


def run_deepfm(x_test):
    optimal_deepfm = decompress_pickle('../../Data_Files/Model_Files/', 'deepfm')

    y_pred_dfm = optimal_deepfm.predict(x_test)

    return y_pred_dfm
