import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ==============================
# Step 1: User-Item Matrix
# ==============================
data = {
    'Movie1': [5, 4, 0, 1],
    'Movie2': [4, 0, 0, 1],
    'Movie3': [1, 1, 0, 5],
    'Movie4': [0, 0, 5, 4],
}

df = pd.DataFrame(data, index=['User1', 'User2', 'User3', 'User4'])

print("\n--- User-Item Matrix ---")
print(df)

# ==============================
# Step 2: User Similarity
# ==============================
user_similarity = cosine_similarity(df)

user_sim_df = pd.DataFrame(user_similarity, index=df.index, columns=df.index)

print("\n--- User Similarity Matrix ---")
print(user_sim_df)

# ==============================
# Step 3: Recommendation Function
# ==============================
def recommend_collaborative(user_name):
    print(f"\nRecommendations for {user_name}:")

    if user_name not in df.index:
        print("User not found!")
        return

    similar_users = user_sim_df[user_name].drop(user_name)
    user_ratings = df.loc[user_name]

    scores = {}

    for other_user in similar_users.index:
        similarity = similar_users[other_user]
        other_ratings = df.loc[other_user]

        for movie in df.columns:
            if user_ratings[movie] == 0:  # not watched
                scores[movie] = scores.get(movie, 0) + similarity * other_ratings[movie]

    if len(scores) == 0:
        print("No recommendations available")
        return

    recommendations = pd.Series(scores).sort_values(ascending=False)

    print(recommendations.head(2))


# ==============================
# Step 4: Run
# ==============================
recommend_collaborative("User1")