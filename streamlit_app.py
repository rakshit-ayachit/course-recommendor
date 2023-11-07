import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the Coursera dataset
coursera_data = pd.read_csv("Coursera.csv")

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(coursera_data['Skills'].fillna(''))

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def recommend(course):
    idx = coursera_data[coursera_data['Course Name'] == course].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:7]  # Get the top 6 similar courses
    course_indices = [i[0] for i in sim_scores]
    return coursera_data['Course Name'].iloc[course_indices]

st.markdown(
    "<h1 style='text-align: center; color: black;'>Coursera Course Recommender</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; color: gray; font-size: 16px;'>Find similar courses from Coursera!</p>",
    unsafe_allow_html=True,
)

# User-friendly interface
st.sidebar.title("Recommendation Options")
selected_course = st.selectbox(
    "Select a course you like:",
    coursera_data['Course Name'].values,
)

if st.sidebar.button('Show Recommended Courses'):
    recommended_courses = recommend(selected_course)
    st.subheader("Recommended Courses:")
    for course in recommended_courses:
        st.markdown(f"[{course}]({coursera_data[coursera_data['Course Name'] == course]['Course URL'].values[0]})")

# Custom footer
st.markdown(
    "<p style='text-align: center; color: gray;'>&copy; <b>Project Procrastinate</b></p>",
    unsafe_allow_html=True,
)
