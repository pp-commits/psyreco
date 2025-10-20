# app.py
import streamlit as st
from modules.llm_analyzer import analyze_mood_mistral
from modules.recommender import recommend_books

st.set_page_config(page_title="🧠 Dora Dark – PsyReco", layout="centered")
st.title("🧠 Dora Dark Presents: PsyReco")
st.write("Books that understand your mind.")

user_input = st.text_area("How are you feeling today?")
if st.button("Discover Books") and user_input.strip():
    with st.spinner("Analyzing your mood..."):
        tags = analyze_mood_mistral(user_input)
    
    with st.spinner("Finding book recommendations..."):
        recs = recommend_books(tags, n=5)
    
    st.subheader("📚 Top Recommendations")
    for i, book in enumerate(recs["metadatas"][0]):
        st.markdown(f"**{i+1}. {book['title']}** by *{book['author']}*")
        st.write(book["summary"])
