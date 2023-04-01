import pandas as pd


def construct_pivot_table(df):
    df_filtered = df[['UserID', 'Title', 'User_Rating']]
    df_rating = df_filtered.copy()

    # df_rating['combined'] = df_rating['UserID'] + df_rating['Title']
    # counts = df_rating['combined'].value_counts()
    # unique_counts = counts[counts == 1]
    # df_rating = df_rating[df_rating['combined'].isin(unique_counts.index)]

    pivoted_table = df_rating.pivot(index='UserID',columns='Title',values='User_Rating').fillna(0)

    return pivoted_table


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
            
