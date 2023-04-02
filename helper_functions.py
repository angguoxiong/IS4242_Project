import math
import bz2
import _pickle as cPickle
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ML_Models.Supervised_Learning.neural_network import tune_neural_network_hyperparameter_with_cross_validation



def compress_pickle(save_path, title, data):
    with bz2.BZ2File(save_path + title + '.pbz2', 'w') as f: 
        cPickle.dump(data, f)

def decompress_pickle(save_path, file):
    data = bz2.BZ2File(save_path + file + '.pbz2', 'rb')
    data = cPickle.load(data)
    return data



def ensemble_supervised(y_pred_dfm, y_pred_nn, y_pred_rf, y_pred_xgb, y_test):
    mae_dfm = mean_absolute_error(y_pred_dfm, y_test)
    mae_nn = mean_absolute_error(y_pred_nn, y_test)
    mae_rf = mean_absolute_error(y_pred_rf, y_test)
    mae_xgb = mean_absolute_error(y_pred_xgb, y_test)

    rmse_dfm = math.sqrt(mean_squared_error(y_pred_dfm, y_test))
    rmse_nn = math.sqrt(mean_squared_error(y_pred_nn, y_test))
    rmse_rf = math.sqrt(mean_squared_error(y_pred_rf, y_test))
    rmse_xgb = math.sqrt(mean_squared_error(y_pred_xgb, y_test))

    error_dfm = (mae_dfm + rmse_dfm) / 2
    error_nn = (mae_nn + rmse_nn) / 2
    error_rf = (mae_rf + rmse_rf) / 2
    error_xgb = (mae_xgb + rmse_xgb) / 2

    ranked_error = [error_dfm, error_nn, error_rf, error_xgb].sort()

    dict_mapping = {error_dfm: y_pred_dfm,
                    error_nn: y_pred_nn,
                    error_rf: y_pred_rf,
                    error_xgb: y_pred_xgb}

    weights = 0.4
    combined_y_pred = 0
    for i in ranked_error:
        combined_y_pred = weights * dict_mapping.get(i)
        weights -= 0.1

    return combined_y_pred
    


def ensemble_unsupervised(reco_set_1, reco_set_2, reco_set_3):
    to_reco = 10
    combined_reco = {}
    common_users = reco_set_1.keys() & reco_set_2.keys() & reco_set_3.keys()
    unique_users = reco_set_1.keys() ^ reco_set_2.keys() ^ reco_set_3.keys()

    if common_users:
        for user in common_users:
            combined_list = reco_set_1[user] + reco_set_2[user] + reco_set_3[user]
            ranked_list = pd.Series(combined_list).value_counts().index.tolist()
            combined_reco[user] = ranked_list[:to_reco]
    
    if unique_users:    
        combined_dict = {key: reco_set_1.get(key, None) or reco_set_2.get(key, None) or reco_set_3.get(key, None) for key in unique_users}
        combined_reco.update(combined_dict)

    return combined_reco


# def ensemble_unsupervised(recos_knn, recos_mf):
#     to_reco = 10
#     combined_reco = {}
#     common_users = recos_knn.keys() & recos_mf.keys()
#     unique_users = recos_knn.keys() ^ recos_mf.keys()

#     print("common")
#     print(common_users)

#     if common_users:
#         for user in common_users:
#             combined_list = recos_knn[user] + recos_mf[user]
#             ranked_list = pd.Series(combined_list).value_counts().index.tolist()
#             print("ranked list")
#             print(pd.Series(combined_list).value_counts())
#             combined_reco[user] = ranked_list[:to_reco]
    
#     print("unique")
#     print(unique_users)

#     if unique_users:    
#         combined_dict = {key: recos_knn.get(key, None) or recos_mf.get(key, None) for key in unique_users}
#         combined_reco.update(combined_dict)

#     return combined_reco