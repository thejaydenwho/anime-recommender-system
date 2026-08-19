# anime-recommender-system

A content-based hybrid anime recommendation engine built in Python. This project combines categorical metadata, synopsis text vectorization, and community popularity metrics to generate personalized recommendations.

## Features

* **Multi-Feature Representation:** Combines weighted metadata including genres, themes, demographics, content ratings, media types, release years, and episode counts.
* **Text Analysis:** Uses TF-IDF vectorization to capture plot similarity across synopses.
* **Hybrid Quality Boost:** Blends content cosine similarity with global popularity, member counts, and user scores.
* **Multi-Tier Search Engine:** Features exact lookup, substring search, and fuzzy matching via RapidFuzz to resolve title queries and typos gracefully.

## Tech Stack

* **Python 3.x**
* **Pandas** & **NumPy** – Data cleaning and array manipulation
* **SciPy** – Sparse matrix construction
* **Scikit-Learn** – Vectorization (TF-IDF, CountVectorizer), scaling, and cosine similarity
* **RapidFuzz** – String matching and title search

## Usage

1. Install dependencies:
   `pip install pandas scipy numpy matplotlib scikit-learn rapidfuzz`
2. Run the recommender:
   `python recommender.py`


## Data Source & Acknowledgements

This project uses the Comprehensive MyAnimeList (MAL) Dataset 2026 by nafiulislam490, available on [Kaggle](https://www.kaggle.com/datasets/nafiulislam490/comprehensive-myanimelist-mal-dataset-2026) under the CC BY-SA 4.0 license.