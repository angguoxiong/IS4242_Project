# IS4242_Project

## Project Structure
```
.
├── Data Files 
│   │
│   ├── Model Files                 <- storing trained model pickle files
│   │   ├── dfm.pkl
│   │   ├── nn.h5
│   │   ├── rf.pkl
│   │   └── xgb.pkl 
│   │
│   ├── Raw Data                    <- storing raw data to be processed 
│   │    ├── users_ratings.csv
│   │    └── users_reviews.csv
│   │
│   └── Training Data               <- storing processed data for model training
│
├── Data Pre-Processing             <- preprocess raw data and conduct feature engineering
│   ├── data_preprocessing.ipynb
│   └── model_ensemble.ipynb
│
├── ML Models                       <- training machine learning models
│   │
│   ├── Colloborative Filtering     <- to recommend other movies based on output from supervised learning methods
│   │   ├── knn_clustering.ipynb
│   │   └── matrix_factorization.ipynb
│   │
│   └── Supervised Learning         <- to predict user ratings 
│       ├── deepfm.ipynb
│       ├── neural_network.ipynb
│       ├── random_forest.ipynb
│       └── xgboost.ipynb
│
└── Web Scraping                    <- extracting data from IMDB
    ├── scraping IMDB user details.ipynb
    └── scraping IMDB user reviews.ipynb

```
