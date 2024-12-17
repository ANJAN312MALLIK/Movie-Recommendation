import streamlit as st
import pandas as pd
import numpy as np

# Load your data
columns_names = ["user_id", "item_id", "rating", "timestamp"]
df = pd.read_csv(r"C:\Users\anjan\OneDrive\Desktop\dump files\u.data", sep='\t', names=columns_names)

# Create a movie titles DataFrame
file_path = r'C:\Users\anjan\OneDrive\Desktop\dump files\ml-100k\u.item'
movies = []
with open(file_path, 'r', encoding='latin-1') as file:
    for line in file:
        fields = line.strip().split('|')
        movie_id = int(fields[0])
        movie_title = fields[1]
        movies.append((movie_id, movie_title))

movie_titles = pd.DataFrame(movies, columns=['item_id', 'title'])

# Merge dataframes
df = pd.merge(df, movie_titles, on="item_id")

# Streamlit app layout
st.title("Movie Ratings Analysis")

# Display the first few rows of the DataFrame
st.subheader("Data Preview")
st.write(df.head())

# Display the number of unique users and items
st.write(f"Number of unique users: {df['user_id'].nunique()}")
st.write(f"Number of unique items: {df['item_id'].nunique()}")

# Group by title and calculate average rating
ratings = pd.DataFrame(df.groupby('title').mean()['rating'])
ratings['num of ratings'] = pd.DataFrame(df.groupby('title').count()['rating'])

# Display top rated movies
st.subheader("Top Rated Movies")
st.write(ratings.sort_values('rating', ascending=False).head(10))

# Function to predict similar movies
def predict_movies(movie_name):
    movie_user_ratings = df[df['title'] == movie_name]
    similar_to_movie = df.corrwith(movie_user_ratings['rating'])
    corr_movie = pd.DataFrame(similar_to_movie, columns=['correlation'])
    corr_movie.dropna(inplace=True)
    corr_movie = corr_movie.join(ratings['num of ratings'])
    predictions = corr_movie[corr_movie['num of ratings'] > 100].sort_values('correlation', ascending=False)
    return predictions

# User input for movie name
movie_name = st.text_input("Enter a movie name to find similar movies:", "Star Wars (1977)")

if movie_name:
    predictions = predict_movies(movie_name)
    st.subheader(f"Similar Movies to {movie_name}")
    st.write(predictions)

# Run the app
if __name__ == "__main__":
    st.run()
