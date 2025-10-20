# modules/recommender.py
from modules.embeddings import model, chroma_client

def recommend_books(tags: dict, n=5):
    """
    Take interest_tags from mood analysis, query ChromaDB, return top N books
    """
    search_query = " ".join(tags.get("interest_tags", []))
    user_vec = model.encode(search_query).tolist()
    collection = chroma_client.get_collection("books")
    
    results = collection.query(query_embeddings=[user_vec], n_results=n)
    return results
