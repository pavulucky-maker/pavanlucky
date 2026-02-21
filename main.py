import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🎬 Movie Recommendation System")

# ---------------- LOAD DATA ----------------
movies = pd.read_csv("movies.csv")
credits = pd.read_csv("credits.csv")

movies = movies.merge(credits, on="title")
movies = movies[['movie_id','title','overview','genres','cast','crew']]
movies.dropna(inplace=True)

# ---------------- CREATE TAGS ----------------
movies['tags'] = movies['overview'] + movies['genres']

# ---------------- VECTORIZE ----------------
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

# ---------------- SIMILARITY ----------------
similarity = cosine_similarity(vectors)

# ---------------- RECOMMEND FUNCTION ----------------
def recommend(movie):

    if movie not in movies['title'].values:
        return ["Movie not found"]

    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []

    for i in movie_list:
        recommended_movies.append(movies.iloc[i[0]].title)

    return recommended_movies

# ---------------- STREAMLIT UI ----------------
selected_movie = st.selectbox(
    "Select Movie",
    movies['title'].values
)

if st.button("Recommend"):

    recommendations = recommend(selected_movie)

    st.subheader("Recommended Movies:")

    for movie in recommendations:
        st.write(movie)
