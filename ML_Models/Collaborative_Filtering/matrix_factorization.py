import pandas as pd
import numpy as np
from sklearn.decomposition import NMF



def construct_pivot_table(df):
    df_filtered = df[['UserID', 'Title', 'User_Rating']]
    df_rating = df_filtered.copy()

    df_rating['combined'] = df_rating['UserID'] + df_rating['Title']
    counts = df_rating['combined'].value_counts()
    unique_counts = counts[counts == 1]
    df_rating = df_rating[df_rating['combined'].isin(unique_counts.index)]

    pivoted_table = df_rating.pivot(index='UserID', columns='Title', values='User_Rating').fillna(0)

    return pivoted_table


def run_user_based_mf_CF(ratings_df):

    pivoted_table = construct_pivot_table(ratings_df)
    
    model = NMF(init='nndsvd',                      # using Nonnegative Double Singular Value Decomposition (NNDSVD) initialization - better for sparseness
                n_components=5,                     
                solver='cd',                        
                l1_ratio=0.5,                       # using a combination of L1 and L2 regularization
                max_iter=500,                       # setting high number to ensure convergence 
                random_state=0)
    
    users_matrix = model.fit_transform(pivoted_table)
    ratings_matrix = model.components_
    predictions = np.dot(users_matrix, ratings_matrix)

    threshold = 1
    recommendations = abs(pivoted_table - predictions)
    filtered = (recommendations[recommendations>threshold]).dropna(how='all')

    all_recos = {}
    to_reco = 30

    for index, row in filtered.iterrows():
        recos = row[row>0].sort_values(ascending=False)[:to_reco].index.tolist()
        all_recos[index] = recos

    return all_recos