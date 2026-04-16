# ==============================
# CONTENT-BASED FILTERING
# ==============================

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Sample Movie Dataset
data = {
    'Movie': ['Movie1', 'Movie2', 'Movie3', 'Movie4'],
    'Action': [1, 1, 0, 0],
    'Comedy': [0, 1, 1, 0],
    'Sci-Fi': [1, 0, 1, 0]
}

df = pd.DataFrame(data)
df.set_index('Movie', inplace=True)

# Compute Similarity
similarity = cosine_similarity(df)

similarity_df = pd.DataFrame(similarity, index=df.index, columns=df.index)

print("\n--- Content-Based Similarity Matrix ---")
print(similarity_df)

# Recommendation Function
def recommend_content(movie_name):
    print(f"\nRecommendations for {movie_name}:")
    similar_movies = similarity_df[movie_name].sort_values(ascending=False)[1:3]
    print(similar_movies)

# Example
recommend_content("Movie1")