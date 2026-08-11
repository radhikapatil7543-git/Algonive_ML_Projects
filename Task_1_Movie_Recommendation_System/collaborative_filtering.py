import pandas as pd

# Load datasets
movies = pd.read_csv("dataset/movies.csv")
ratings = pd.read_csv("dataset/ratings.csv")

# Merge movies and ratings
movie_ratings = ratings.merge(movies, on="movieId")

# Create user-movie matrix
user_movie_matrix = movie_ratings.pivot_table(index="userId", columns="title", values="rating")

# Fill missing values with 0
user_movie_matrix = user_movie_matrix.fillna(0)
print("User-Movie Matrix:")
print(user_movie_matrix.head())

# Ask user for a movie name
movie_name = input("\nEnter a movie name: ")

# Check if movie exists
if movie_name in user_movie_matrix.columns:

# Find correlation with all other movies
 similar_movies = user_movie_matrix.corrwith(user_movie_matrix[movie_name])

# Remove missing values
 similar_movies = similar_movies.dropna()

# Sort recommendations
 recommendations = similar_movies.sort_values(ascending=False)
 print("\nTop 10 Recommended Movies:")
 print(recommendations.head(10))
else:
 print("Movie not found in dataset.")