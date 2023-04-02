import pandas as pd
from surprise import KNNBaseline, Reader, Dataset



def run_item_based_knn_CF(ratings_df):

    reader = Reader(rating_scale=(0, 10))
    data = Dataset.load_from_df(ratings_df[['UserID', 'Title', 'User_Rating']], reader)

    # Use item-based k-NN with cosine similarity
    sim_options = {'name': 'pearson_baseline', 'user_based': False}
    algo = KNNBaseline(k=20, sim_options=sim_options)

    # Train the algorithm on the dataset
    trainset = data.build_full_trainset()
    algo.fit(trainset)

    # Get the list of all item IDs
    all_item_ids = ratings_df['Title'].value_counts().index.tolist()

    # Generate recommendations for each item
    item_recommendations = {}
    for item_id in all_item_ids:
        item_inner_id = trainset.to_inner_iid(item_id)
        item_neighbors = algo.get_neighbors(item_inner_id, k=10)
        item_neighbors = [trainset.to_raw_iid(inner_id) for inner_id in item_neighbors]
        item_recommendations[item_id] = item_neighbors

    # Get the list of all item IDs
    all_user_ids = ratings_df['UserID'].value_counts().index.tolist()

    user_recommendations = {}
    for user in all_user_ids:
        user_reco = []
        user_movies = ratings_df[ratings_df['UserID'] == user]
        top_rated_movies = user_movies.sort_values('User_Rating', ascending=False).head(10)['Title'].tolist()
        for movie in top_rated_movies:
            user_reco.extend(item_recommendations[movie])
        user_recommendations[user] = user_reco

    return user_recommendations




def run_user_based_knn_CF(ratings_df):

    reader = Reader(rating_scale=(0, 10))
    data = Dataset.load_from_df(ratings_df[['UserID', 'Title', 'User_Rating']], reader)

    # Use item-based k-NN with cosine similarity
    sim_options = {'name': 'pearson_baseline', 'user_based': True}
    algo = KNNBaseline(k=10, sim_options=sim_options)

    # Train the algorithm on the dataset
    trainset = data.build_full_trainset()
    algo.fit(trainset)


    # Get the list of all item IDs
    all_user_ids = ratings_df['UserID'].value_counts().index.tolist()

    user_recommendations = {}
    for user_id in all_user_ids:
        user_inner_id = trainset.to_inner_uid(user_id)
        user_neighbors = algo.get_neighbors(user_inner_id, k=5)
        similar_users = [trainset.to_raw_uid(inner_id) for inner_id in user_neighbors]
        
        # Find the movies that similar users have rated highly
        movie_recommendations = []
        for similar_user in similar_users:
            similar_user_ratings = ratings_df[ratings_df['UserID'] == similar_user]
            high_rated_movies = similar_user_ratings.sort_values('User_Rating', ascending=False).head(int(len(similar_user_ratings) * 0.01))['Title'].tolist()
            movie_recommendations.extend(high_rated_movies)
        
        # Remove the movies that the user has already rated
        user_ratings = ratings_df[ratings_df['UserID'] == user_id]
        user_rated_movies = user_ratings['Title'].tolist()
        movie_recommendations = list(set(movie_recommendations) - set(user_rated_movies))
        
        user_recommendations[user_id] = movie_recommendations

    return user_recommendations