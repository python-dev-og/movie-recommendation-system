# Popularity-Based Filtering
#
# Load the Data

import pandas as pd
movies = pd.read_csv('movies.csv')
credits = pd.read_csv('credits.csv')
ratings = pd.read_csv('ratings.csv')

# Calculate a weighted rating
#
# (WR) = (v / (v + m)) * R + (m / (v + m)) * C
#
# v - number of votes for a movie
#
# m - minimum number of voted required
#
# R - average rating of the movie
#
# C - average rating across all movies

m = movies['vote_count'].quantile(0.9)
C = movies['vote_average'].mean()
movies_filtered = movies.copy().loc[movies['vote_count'] >= m]

def weighted_rating(df, m=m, C=C):
    v = df['vote_count']
    R = df['vote_average']
    wr = (v / (v + m) * R) + (m / (v + m) * C)
    return wr

movies_filtered['weighted_rating'] = movies_filtered.apply(weighted_rating, axis=1)
movies_filtered.sort_values('weighted_rating', ascending=False)[['title', 'weighted_rating']].head(10)
movies_filtered.sort_values('weighted_rating', ascending=False)[['title', 'weighted_rating']].head(10).to_dict()


# Content-Based Filtering
movies = pd.read_csv("movies.csv", sep=",")

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words="english")
movies["overview"] = movies["overview"].fillna("")

tfidf_matrix = tfidf.fit_transform(movies["overview"])
pd.DataFrame((tfidf_matrix.toarray()), columns=tfidf.get_feature_names())
tfidf_matrix.shape

# Similarity_matrix
from sklearn.metrics.pairwise import linear_kernel
similarity_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

# Find the most similar movies to a certain movie
movie_title = "John Carter"
idx = movies.loc[movies["title"] == movie_title].index[0]
idx

scores = list(enumerate(similarity_matrix[idx]))
scores

scores =sorted(scores, key=lambda x: x[1], reverse=True)
scores

movies_indices = [tpl[0] for tpl in scores[1:4]]
movies_indices

list(movies["title"].iloc[movies_indices])

def similar_movies(movie_title, nr_movies=3):
    idx = movies.loc[movies["title"] == movie_title].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores =sorted(scores, key=lambda x: x[1], reverse=True)
    movie_indices = [tpl[0] for tpl in scores[1:nr_movies+1]]
    similar_titles = list(movies["title"].iloc[movie_indices])
    return similar_titles

similar_movies("Kung Fu Panda 3", 3)


# Collaborative-Based Filtering
# Load the Data
ratings = pd.read_csv('ratings.csv')[['userId', 'movieId', 'rating']]
ratings.head()

# Create the dataset
#!pip install scikit-surprise

from surprise import Dataset, Reader

reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(ratings, reader)
dataset

# Build the trainset
trainset = dataset.build_full_trainset()
list(trainset.all_ratings())

# Train the Model
from surprise import SVD

svd = SVD()
svd.fit(trainset)
svd.predict(15, 1956)

# Validation
from surprise import model_selection

model_selection.cross_validate(svd, dataset, measures=['RMSE', 'MAE'], cv=5, verbose=True)


