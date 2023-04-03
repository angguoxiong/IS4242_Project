import math
import bz2
import _pickle as cPickle
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error



def compress_pickle(save_path, title, data):
    with bz2.BZ2File(save_path + title + '.pbz2', 'w') as f: 
        cPickle.dump(data, f)

def decompress_pickle(save_path, file):
    data = bz2.BZ2File(save_path + file + '.pbz2', 'rb')
    data = cPickle.load(data)
    return data



def ensemble_supervised(y_pred_dfm, y_pred_nn, y_pred_rf, y_pred_xgb):

    model_perf = pd.read_csv('Data_Files/Model_Files/model_performance.csv')
    model_rankings = model_perf.sort_values(by='combined_error').Model.tolist()
    
    model_mapping = {
        'DeepFM': y_pred_dfm,
        'NeuralNetwork': y_pred_nn,
        'RandomForest': y_pred_rf,
        'XGBoost': y_pred_xgb
        }

    weights = 0.4
    combined_y_pred = 0
    for i in model_rankings:
        combined_y_pred += (weights * model_mapping.get(i))
        weights -= 0.1

    combined_ratings = combined_y_pred.reset_index()
    ratings_df = combined_ratings.rename(columns={0: 'User_Rating'})

    return ratings_df
    


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