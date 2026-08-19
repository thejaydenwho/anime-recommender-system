import pandas as pd
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz

pd.set_option('display.max_columns', None)

tfidf = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),       # Capture both single words and two-word phrases
    min_df=2,                 # Ignore terms that appear fewer than 2 times across the dataset
    max_features=10000,       # Keep top 10,000 n-grams
    sublinear_tf=True         # Dampen repeated words in short texts
)

scaler = MinMaxScaler()

# Process CSV file
df_raw = pd.read_csv("data/anime_info.csv", on_bad_lines="skip", quotechar='"', skipinitialspace=True)
df_clean = df_raw.copy()
df_clean = df_clean.map(lambda x: x.strip() if isinstance(x,str) else x)
df_clean.columns = df_clean.columns.str.strip()

# Filter for valid series types
valid_types = ['TV', 'Movie', 'OVA', 'ONA', 'TV Special', 'Special']
type_mask = df_clean['type'].isin(valid_types)
df_clean = df_clean[type_mask].reset_index(drop=True)

# Remove series with inadequate information
df_clean = df_clean.dropna(subset=['score', 'synopsis', 'genres', 'aired_from'])
missing_synopsis = "No synopsis has been added for this series yet.  Click here to update this information."
df_clean = df_clean[~df_clean['synopsis'].str.contains(missing_synopsis, case=False, na=False)].reset_index(drop=True)

# Add year to series missing one
df_clean['aired_from_dt'] = pd.to_datetime(df_clean['aired_from'], errors='coerce')
df_clean['year'] = df_clean['year'].fillna(df_clean['aired_from_dt'].dt.year)
# Turn years into int instead of float
df_clean['year'] = df_clean['year'].astype(int)

# Filling in missing episode counts
# 1. Calculate exact fractional years elapsed from air date to current date
current_date = pd.Timestamp('2026-08-17')
elapsed_years_exact = (current_date - df_clean['aired_from_dt']).dt.days / 365.25

# Fallback to the 'year' column if 'aired_from' fails to parse
fallback_years = (2026 - df_clean['year']).clip(lower=1)
elapsed_years = elapsed_years_exact.fillna(fallback_years).clip(lower=0.25)

# 2. Estimate episodes using exact elapsed time * 35 multiplier
estimated_ongoing_eps = elapsed_years * 35

# 3. Fill missing or zero values in 'episodes'
df_clean['episodes_clean'] = df_clean['episodes'].replace(0, np.nan)
df_clean['episodes_clean'] = df_clean['episodes_clean'].fillna(estimated_ongoing_eps)

# 4. Apply log-transform and scale to [0, 1] for matrix integration
df_clean['log_episodes'] = np.log1p(df_clean['episodes_clean'])


# One-hot encoded media type
type_encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
type_matrix = type_encoder.fit_transform(df_clean[['type']])

# Define content rating hierarchy

# 1. Standardize string formatting and handle missing data
df_clean['content_rating'] = df_clean['content_rating'].fillna('Unknown').astype(str)

# 2. Extract just the prefix code safely using a space-split
df_clean['content_abbreviation'] = df_clean['content_rating'].apply(lambda x: x.split(' ')[0].strip())

# 3. Dictionary mapping each content rating to a score
rating_order = {
    'G': 1,
    'PG': 2,
    'PG-13': 3,
    'R': 4,
    'R+': 5,
    'Rx': 6
}

# 4. Map the clean prefixes to ranks and scale
df_clean['content_rating_rank'] = df_clean['content_abbreviation'].map(rating_order).fillna(0)
rating_matrix = scaler.fit_transform(df_clean[['content_rating_rank']])

# Treat genre names separated by '|' as distinct tokens
df_clean['genres_list'] =  df_clean['genres'].fillna('').str.split('|')
genre_vec = CountVectorizer(analyzer=lambda x: x)
genre_matrix = genre_vec.fit_transform(df_clean['genres_list'])
# Get the feature names if needed
unique_genres = genre_vec.get_feature_names_out()

# Repeat same process for demographics and themes
df_clean['demographics_list'] =  df_clean['demographics'].fillna('').str.split('|')
demographics_vec = CountVectorizer(analyzer=lambda x: x)
demographics_matrix = demographics_vec.fit_transform(df_clean['demographics_list'])
unique_demographics = demographics_vec.get_feature_names_out()

df_clean['themes_list'] =  df_clean['themes'].fillna('').str.split('|')
themes_vec = CountVectorizer(analyzer=lambda x: x)
themes_matrix = themes_vec.fit_transform(df_clean['themes_list'])
unique_themes = themes_vec.get_feature_names_out()

# Synopsis matrix
tfidf_matrix = tfidf.fit_transform(df_clean['synopsis'].fillna('')) 

# Numeric matrices
episodes_scaled = scaler.fit_transform(df_clean[['log_episodes']])
year_scaled = scaler.fit_transform(df_clean[['year']])

# 1. Fill NaNs (for unranked/obscure entries, set to maximum rank/lowest priority)
max_rank = df_clean['rank'].max()
max_pop = df_clean['popularity'].max()

df_clean['rank'] = df_clean['rank'].fillna(max_rank)
df_clean['popularity'] = df_clean['popularity'].fillna(max_pop)

# 2. Prepare features (log transform heavy right-skewed counts)
df_clean['log_members'] = np.log1p(df_clean['members'].fillna(0))
df_clean['log_favorites'] = np.log1p(df_clean['favorites'].fillna(0))
df_clean['score'] = df_clean['score'].fillna(df_clean['score'].median())

# 3. Scale raw metrics (0 to 1)
metrics = df_clean[['score', 'log_members', 'log_favorites', 'rank', 'popularity']]
scaled_metrics = scaler.fit_transform(metrics)

# 4. Extract and INVERT rank and popularity so 1.0 represents top tier
score_s      = scaled_metrics[:, 0]
members_s    = scaled_metrics[:, 1]
favorites_s  = scaled_metrics[:, 2]
rank_s       = 1.0 - scaled_metrics[:, 3]        # Inverted: Rank #1 becomes 1.0
popularity_s = 1.0 - scaled_metrics[:, 4]        # Inverted: Popularity #1 becomes 1.0

# 5. Calculate final unified quality boost
df_clean['quality_boost'] = (
    score_s      * 0.75 +
    favorites_s  * 0.05 +
    rank_s       * 0.15 +
    popularity_s * 0.05
)

# 1. Feature Weight Configuration
w_genre  = 3.0
w_theme  = 1.5
w_demo   = 0.8
w_tfidf  = 1.3
w_rating = 0.5  # Ordinal content rating rank
w_type   = 0.5  # One-hot media type
w_year   = 0.2  # Scaled premiere year
w_eps    = 0.1  # Log-scaled estimated episode count

# 2. Ensure 1D/2D dense arrays are explicit CSR sparse matrices
year_sparse = sp.csr_matrix(year_scaled)
episodes_sparse = sp.csr_matrix(episodes_scaled)

# 3. Stack all feature matrices horizontally
final_feature_matrix = sp.hstack([
    genre_matrix * w_genre,
    themes_matrix * w_theme,
    demographics_matrix * w_demo,
    tfidf_matrix * w_tfidf,
    rating_matrix * w_rating,
    type_matrix * w_type,
    year_sparse * w_year,
    episodes_sparse * w_eps
]).tocsr()  # Convert to Compressed Sparse Row format for fast vector ops

def build_title_lookup(df):
    title_to_index = {}
    
    for idx, row in df.iterrows():
        candidates = []
        
        # Primary, English, and Japanese titles
        for field in ['title', 'title_english', 'title_japanese']:
            val = row.get(field)
            if pd.notna(val) and str(val).strip():
                candidates.append(str(val).strip())
        
        # Synonyms (pipe-separated)
        synonyms = row.get('title_synonyms')
        if pd.notna(synonyms) and isinstance(synonyms, str):
            # Splitting by '|' handles your synonym format
            split_syns = [s.strip() for s in synonyms.split('|') if s.strip()]
            candidates.extend(split_syns)

        # Map all variants in lowercase to row index
        for title in candidates:
            clean_title = title.lower()
            if clean_title not in title_to_index:
                title_to_index[clean_title] = idx
                
    return title_to_index

# Generate mapping
title_to_index = build_title_lookup(df_clean)


def get_anime_index_robust(title_query, title_map, score_cutoff=75):
    if not isinstance(title_query, str):
        return None, None

    clean_query = title_query.strip().lower()

    # Tier 1: Exact O(1) Match
    if clean_query in title_map:
        return title_map[clean_query], clean_query

    # Tier 2: Safe Substring Search
    # Only check if query is inside titles, and ignore ultra-short single-letter title keys
    substring_matches = []
    for title, idx in title_map.items():
        if len(title) >= 3 and clean_query in title:
            substring_matches.append((title, idx))
            
    if substring_matches:
        # Sort by shortest matching title length
        substring_matches.sort(key=lambda x: len(x[0]))
        best_title, best_idx = substring_matches[0]
        print(f"Substring match applied: '{title_query}' -> '{best_title}'")
        return best_idx, best_title

    # Tier 3: Fuzzy Matching Fallback
    match, score, _ = process.extractOne(
        clean_query, 
        title_map.keys(), 
        scorer=fuzz.WRatio
    )

    if score >= score_cutoff:
        print(f"Fuzzy match applied: '{title_query}' -> '{match}' (Score: {score:.1f}%)")
        return title_map[match], match

    return None, None

def recommend_anime(title_query, df, feature_matrix, title_map, top_n=10, boost_weight=0.15):
    # Use robust lookup
    idx, matched_title = get_anime_index_robust(title_query, title_map)
    
    if idx is None:
        return f"Error: Anime '{title_query}' not found."
    
    # 1. Calculate Content Similarity
    query_vec = feature_matrix[idx]
    content_sim = cosine_similarity(query_vec, feature_matrix).flatten()
    
    # 2. Apply Post-Similarity Quality Boost
    quality_boost = df['quality_boost'].values
    final_scores = (1 - boost_weight) * content_sim + boost_weight * quality_boost
    
    # 3. Rank Top Results
    top_indices = np.argsort(final_scores)[::-1]
    top_indices = [i for i in top_indices][:top_n]
    
    # 4. Format Results
    results = df.iloc[top_indices][['title', 'title_english', 'score']].copy()
    results['similarity'] = content_sim[top_indices].round(4)
    results['boosted_similarity'] = final_scores[top_indices].round(4)
    
    return results.reset_index(drop=True)

recommendations = recommend_anime(
    title_query="Hunter x Hunter (2011)",
    df=df_clean,
    feature_matrix=final_feature_matrix,
    title_map=title_to_index,
    top_n=20,
    boost_weight=0.1
)

print(recommendations)