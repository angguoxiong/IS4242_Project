import pandas as pd
import numpy as np
from sklearn.decomposition import NMF


def run_user_based_mf_collab_filtering(pivoted_table):
    
    model = NMF(init='nndsvd',                      # using Nonnegative Double Singular Value Decomposition (NNDSVD) initialization - better for sparseness
                n_components=len(pivoted_table),    # setting to the highest between min(n_samples, n_features)
                solver='mu',                        # using Multiplicative Update solver provides much higher predicted ratings as compared to Coordinate Descent solver.
                l1_ratio=0.5,                       # using a combination of L1 and L2 regularization
                max_iter=500,                       # setting high number to ensure convergence 
                random_state=0)
    
    users_matrix = model.fit_transform(pivoted_table)
    ratings_matrix = model.components_
    predictions = np.dot(users_matrix, ratings_matrix)

    threshold = 0.001
    recommendations = abs(pivoted_table - predictions)
    filtered = (recommendations[recommendations>threshold]).dropna(how='all')

    all_recos = {}
    to_reco = 10

    for index, row in filtered.iterrows():
        recos = row[row>0].sort_values(ascending=False)[:to_reco].index.tolist()
        all_recos[index] = recos

    return all_recos