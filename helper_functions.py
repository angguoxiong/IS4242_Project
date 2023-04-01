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


def construct_pivot_table(df):
    df_filtered = df[['UserID', 'Title', 'User_Rating']]
    df_rating = df_filtered.copy()

    # df_rating['combined'] = df_rating['UserID'] + df_rating['Title']
    # counts = df_rating['combined'].value_counts()
    # unique_counts = counts[counts == 1]
    # df_rating = df_rating[df_rating['combined'].isin(unique_counts.index)]

    pivoted_table = df_rating.pivot(index='UserID',columns='Title',values='User_Rating').fillna(0)

    return pivoted_table


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
    

def ensemble_unsupervised(recos_knn, recos_mf):
    to_reco = 10
    combined_reco = {}
    common_users = recos_knn.keys() & recos_mf.keys()
    unique_users = recos_knn.keys() ^ recos_mf.keys()

    if common_users:
        for user in common_users:
            combined_list = recos_knn[user] + recos_mf[user]
            ranked_list = pd.Series(combined_list).value_counts().index.tolist()
            combined_reco[user] = ranked_list[:to_reco]
    
    if unique_users:    
        combined_dict = {key: recos_knn.get(key, None) or recos_mf.get(key, None) for key in unique_users}
        combined_reco.update(combined_dict)

    return combined_reco
