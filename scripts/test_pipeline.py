# app.py
import streamlit as st
from modules.llm_analyzer import analyze_mood_mistral
from modules.recommender import recommend_books
import json
from modules.recommender import jaccard_similarity  # import utility for mood/tag similarity

def run_cli_test():
    text = "I feel burned out and anxious about my career and need clarity and calm. I'd like something practical."
    print("Input:", text)

    # 1. Analyze mood to get tags
    tags = analyze_mood_mistral(text)
    print("Tags:", json.dumps(tags, indent=2))

    # 2. Get recommendations (weighted)
    recs = recommend_books(tags, n=5)

    print("\nResults (Top Recommendations with Weighted Scores):")
    tags_list = tags.get("interest_tags", [])
    for idx, (md, dist) in enumerate(zip(recs.get("metadatas", [[]])[0], recs.get("distances", [[]])[0]), start=1):
        semantic_score = 1 - dist  # semantic similarity
        book_tags = md.get("emotion_tags", []) + md.get("mindset_tags", [])
        mood_score = jaccard_similarity(tags_list, book_tags)
        weighted_score = 0.7 * semantic_score + 0.3 * mood_score  # Stage 2 weights

        print(f"{idx}. {md.get('title')} by {md.get('author')} (Weighted Score ~{weighted_score:.2f})")
        print("   Summary:", md.get("summary"), "\n")


def run_streamlit_app():
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
        tags_list = tags.get("interest_tags", [])
        for i, book in enumerate(recs["metadatas"][0]):
            # Compute weighted score for display if desired
            semantic_score = 1 - recs["distances"][0][i]
            book_tags = book.get("emotion_tags", []) + book.get("mindset_tags", [])
            mood_score = jaccard_similarity(tags_list, book_tags)
            weighted_score = 0.7 * semantic_score + 0.3 * mood_score

            st.markdown(f"**{i+1}. {book['title']}** by *{book['author']}*")
            st.write(book["summary"])
            st.caption(f"Weighted Score: ~{weighted_score:.2f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_cli_test()
    else:
        run_streamlit_app()
