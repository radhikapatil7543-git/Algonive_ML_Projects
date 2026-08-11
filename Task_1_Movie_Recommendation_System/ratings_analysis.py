import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
movies = pd.read_csv("dataset/movies.csv")
ratings = pd.read_csv("dataset/ratings.csv")

# Merge datasets
movie_ratings = pd.merge(ratings, movies, on="movieId")

# Display first 5 rows
print("Movie Ratings Dataset:")
print(movie_ratings.head())

# Basic statistics
print("\nRatings Statistics:")
print(movie_ratings["rating"].describe())

# Count ratings
rating_counts = movie_ratings["rating"].value_counts().sort_index()
print("\nRating Counts:")
print(rating_counts)

# Plot rating distribution
plt.figure(figsize=(8,5))
rating_counts.plot(kind="bar")
plt.title("Distribution of Movie Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Ratings")
plt.grid(axis="y")
plt.show()

# Top 10 most rated movies
top_movies = movie_ratings.groupby("title")["rating"].count().sort_values(ascending=False).head(10)
print("\nTop 10 Most Rated Movies:")
print(top_movies)

# Plot top 10 movies
plt.figure(figsize=(10,6))
top_movies.plot(kind="bar")
plt.title("Top 10 Most Rated Movies")
plt.xlabel("Movie")
plt.ylabel("Number of Ratings")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()