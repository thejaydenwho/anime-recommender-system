# anime-recommender-system

A content-based hybrid anime recommendation engine built in Python. This project combines categorical metadata, dense semantic text vectorization, and community popularity metrics to generate personalized recommendations.

## Features

* **Multi-Feature Late Fusion:** Integrates weighted metadata including genres, themes, demographics, content ratings, media types, release years, and episode counts.
* **Semantic & Text Analysis:** Combines dense neural embeddings from SentenceTransformers with TF-IDF vectorization to capture deep plot themes and specific jargon.
* **Hybrid Quality Boost:** Blends content cosine similarity with global popularity, member counts, favorites, and user scores.
* **Multi-Tier Search Engine:** Features exact lookup, substring search, and fuzzy matching via RapidFuzz to resolve title queries and typos gracefully.

## Tech Stack

* **Python 3.x**
* **Pandas** & **NumPy** – Data cleaning and array manipulation
* **SciPy** – Sparse matrix construction and vector operations
* **SentenceTransformers (PyTorch / ROCm / CUDA)** – Dense vector embeddings via `mixedbread-ai/mxbai-embed-large-v1`
* **Scikit-Learn** – Vectorization (TF-IDF, CountVectorizer), scaling, and cosine similarity
* **RapidFuzz** – String matching and title search

## Usage

1. Install dependencies:
   `pip install pandas scipy numpy matplotlib scikit-learn rapidfuzz`

2. Run the recommender:
   * **Pre-computed Embeddings:** The `data/synopsis_embeddings.npy` file is included in the repository, so you can run recommendations immediately without downloading external model weights.
   * **Local Regeneration:** If you delete the `.npy` file or modify the dataset, `recommender.py` will automatically load `mixedbread-ai/mxbai-embed-large-v1` to regenerate the embeddings locally.

3. Run the recommender:
   `python recommender.py`


## Data Source & Acknowledgements

This project uses the Comprehensive MyAnimeList (MAL) Dataset 2026 by nafiulislam490, available for download on [Kaggle](https://www.kaggle.com/datasets/nafiulislam490/comprehensive-myanimelist-mal-dataset-2026) under the CC BY-SA 4.0 license.
