# IS4242_Project

## Project Structure
```
.
├── Data Files                                  
│   ├── Model Files                       <- storing trained model pickle files
│   │   ├── deepfm.pkl
│   │   ├── neural_network.pkl
│   │   ├── random_forest.pkl
│   │   └── xgboost.pkl 
│   │
│   └── Raw Data                          <- storing raw data for model training 
│       ├── users_ratings.csv
│       └── users_reviews.csv                                
│
├── Data Pre-Processing                   <- preprocess raw data and conduct feature engineering
│   ├── data_preprocessing.ipynb
│   └── model_ensemble.ipynb
│
├── ML Models                             <- training machine learning models
│   ├── Colloborative Filtering           <- to recommend other movies based on output from supervised learning methods
│   │   ├── knn_clustering.ipynb
│   │   └── matrix_factorization.ipynb
│   │
│   └── Supervised Learning               <- to predict user ratings 
│       ├── deepfm.ipynb
│       ├── neural_network.ipynb
│       ├── random_forest.ipynb
│       └── xgboost.ipynb
│
└── Web Scraping                          <- extracting data from IMDB
    ├── scraping IMDB user details.ipynb
    └── scraping IMDB user reviews.ipynb

```
