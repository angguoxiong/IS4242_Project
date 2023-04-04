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

    # Loop over the dictionaries and combine the recommended movies for the same user
    for d in [reco_set_1, reco_set_2, reco_set_3]:
        for key, value in d.items():
            if key not in combined_reco:
                combined_reco[key] = []
            combined_reco[key].extend(value)

    # Find the most recommended movies 
    for user, movies in combined_reco.items():
        ranked_list = pd.Series(movies).value_counts().index.tolist()
        combined_reco[user] = ranked_list[:to_reco]

    return combined_reco
