
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
ratings = pd.read_csv("data/u.data", sep='\t', encoding='latin-1', header=None,
                      names=['user_id', 'item_id', 'rating', 'timestamp'])
items = pd.read_csv("data/u.item", sep='|', encoding='latin-1', header=None,
                    names=['item_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
                           'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
                           'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                           'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])

# Merge ratings with movie titles
data = pd.merge(ratings, items, on="item_id")
user_item_matrix = data.pivot_table(index='user_id', columns='title', values='rating')

def recommend_movies(user_id, top_n=5):
    filled_matrix = user_item_matrix.fillna(0)
    similarity = cosine_similarity(filled_matrix)
    similarity_df = pd.DataFrame(similarity, index=filled_matrix.index, columns=filled_matrix.index)

    similar_users = similarity_df[user_id].sort_values(ascending=False)[1:]
    weighted_ratings = np.dot(similar_users.values, filled_matrix.loc[similar_users.index])
    similarity_sum = np.sum(similar_users.values)

    if similarity_sum == 0:
        return []

    weighted_avg = weighted_ratings / similarity_sum
    recommendations = pd.Series(weighted_avg, index=filled_matrix.columns)

    watched_movies = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id].notna()].index
    recommendations = recommendations.drop(watched_movies, errors='ignore')

    top_recommendations = recommendations.sort_values(ascending=False).head(top_n)
    return list(top_recommendations.index)

# Example
if __name__ == "__main__":
    user_id = 10
    print(f"Recommended Movies for User {user_id}:")
    for movie in recommend_movies(user_id, top_n=5):
        print(f" - {movie}")
