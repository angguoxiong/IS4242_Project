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
│   │   ├── k_nearest_neighbour.ipynb
│   │   └── matrix_factorization.ipynb
│   │
│   └── Supervised Learning         <- to predict user ratings for unseen movies
│       ├── deepfm.ipynb
│       ├── neural_network.ipynb
│       ├── random_forest.ipynb
│       └── xgb.ipynb
│
└── Web Scraping                    <- extracting data from IMDB

```
