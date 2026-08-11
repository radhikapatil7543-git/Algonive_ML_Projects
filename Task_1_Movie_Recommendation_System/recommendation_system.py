import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the datasets
movies = pd.read_csv("dataset/movies.csv")
ratings = pd.read_csv("dataset/ratings.csv")

# Display first 5 rows of the movies dataset
print("Movies Dataset:")
print(movies.head())

# Display first 5 rows of the ratings dataset
print("\nRatings Dataset:")
print(ratings.head())

# Create tags using genres
movies["tags"] = movies["genres"].fillna("")

# Convert text into vectors
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(movies["tags"]).toarray()

# Calculate similarity
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie_name):
    movie_name = movie_name.lower()
    if movie_name not in movies["title"].str.lower().values:
        print("\nMovie not found in dataset.")
        return 
    index = movies[movies["title"].str.lower() == movie_name].index[0]
    distances = list(enumerate(similarity[index]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)
    count = 0
    for i in distances[1:]:
     print(movies.iloc[i[0]].title)
     count += 1
     if count == 5:
      break

# User input
movie = input("\nEnter a movie name: ")
recommend(movie)