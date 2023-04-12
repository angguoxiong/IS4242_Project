# IS4242_Project

## Project Structure
```
.
├── Data Files 
│   ├── Model Files                 <- storing trained model compressed pickle files and gridsearch/model results
│   ├── Raw Data                    <- storing raw data to be processed 
│   └── Training Data               <- storing processed data for model training
│
├── Data Pre-Processing             <- preprocess raw data and conduct feature engineering
│   ├── eda.ipynb
|   ├── feature_engineering.ipynb
|   ├── textual_processing.ipynb
|   ├── train_test_split.ipynb
│   └── unseen_movies_processing.ipynb
│
├── ML Models                       <- training machine learning models
│   │
│   ├── Colloborative Filtering     <- to recommend other movies based on output from supervised learning methods
│   │   ├── k_nearest_neighbour.py
│   │   └── matrix_factorization.py
│   │
│   └── Supervised Learning         <- to predict user ratings for unseen movies
│       ├── deepfm.py
│       ├── neural_network.py
│       ├── random_forest.py
│       └── xgb.py
│
├── Web Scraping                    <- extracting data from IMDB
│
│
├── helper_functions.py             <- methods to ensemble outputs from supervised and collaborative filtering methods
└── main.ipynb                      <- containing the pipeline and logic of the recommender system 

```

## Running the Movie Recommender System

1. Install the necessary packages needed
    ```bash
    pip install -r requirements.txt
    ```
    
2. Train and tune the models in main.ipynb
    ```bash
    train_and_tune_ML_Models(x_train, y_train, x_test, y_test)
    ```
     > __Note__ The trained model files are already uploaded in the repository, this step can be safely skipped. The intensive GridSearchCV process will take hours to complete.
    
3. Run the recommender system in main.ipynb
    ```bash
    intelligent_recommender(to_predict, user_ratings)
    ```
4. Observe the recommendation results in the printed output.
5. Model results are recorded in Data_Files/Model_Files.
