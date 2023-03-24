import pandas as pd
import numpy as np
from sklearn.decomposition import NMF



pivoted_table = pd.DataFrame()      #TODO: to take in combined output from the ML models

model = NMF(init='nndsvd', n_components=pivoted_table.columns.size, solver='cd', l1_ratio=0.0, max_iter=500, random_state=0)
users_matrix = model.fit_transform(pivoted_table)
ratings_matrix = model.components_
predictions = np.dot(users_matrix, ratings_matrix)


mse = ((pivoted_table - predictions) ** 2).mean()
rmse = np.sqrt(mse)
print("RMSE:", rmse)