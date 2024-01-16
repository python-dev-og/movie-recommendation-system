![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![Pandas](https://img.shields.io/badge/pandas-2.1.4-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit_learn-0.24.2-blue.svg)
![Scikit-Surprise](https://img.shields.io/badge/scikit_surprise-1.1.3-blue.svg)

# Machine Learning Project
## Popularity-Based, Content-Based, and Collaborative-Based Movie Recommendation System

## Overview
This repository contains the implementation of a movie recommendation system using three different approaches: 
Popularity-Based Filtering, Content-Based Filtering, and Collaborative-Based Filtering. 
The system utilizes various Python libraries and datasets to offer movie recommendations based on different criteria and algorithms.

## Prerequisites
- Python 3.x
- Pandas
- Scikit-Learn
- Scikit-Surprise

To install necessary libraries, run:
```bash
pip install pandas scikit-learn scikit-surprise
```

## Dataset
The system uses the following datasets:
- `movies.csv`: Contains movie details like titles and average votes.
- `credits.csv`: Contains movie credits information (not used in the current implementation).
- `ratings.csv`: Contains user ratings for different movies.

These files need to be present in the same directory as the script for the system to function correctly.

## Implementation Details

### Popularity-Based Filtering
This approach ranks movies based on their popularity, using a weighted rating formula. 
The `weighted_rating` function calculates this rating, and the top 10 movies are displayed.

### Content-Based Filtering
This method recommends movies similar to a given movie based on their content. 
It utilizes the TF-IDF Vectorizer to transform movie overviews into vectorized form and then computes the similarity scores between movies. 
The `similar_movies` function is used to find and return similar movies to a given title.

### Collaborative-Based Filtering
Collaborative Filtering is implemented using the Scikit-Surprise library. 
It uses user-item interactions to predict a user's interest in an item (movie) based on the ratings of other users. 
The SVD algorithm is employed for making these predictions, and the system's performance is validated using cross-validation.

## Usage

### Popularity-Based Filtering
Run the code under the section 'Popularity-Based Filtering' to see the top 10 popular movies based on the weighted rating.

### Content-Based Filtering
Use the `similar_movies` function to find movies similar to a given title, e.g.:
```python
print(similar_movies("Kung Fu Panda 3", 3))
```

### Collaborative-Based Filtering
The collaborative filtering section demonstrates the process of model training and validation. 
It includes loading data, training the SVD model, and evaluating it using cross-validation.

## Contributions
Contributions to this project are welcome. Please ensure to follow the coding standards 
and add unit tests for any new or changed functionality.

## License
This project is licensed under the [MIT License](LICENSE.md).

