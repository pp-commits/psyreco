# modules/recommender.py
from modules.embeddings import model, chroma_client
from typing import Dict

def recommend_books(tags: Dict, n: int = 5):
    """
    Take interest_tags from mood analysis, query ChromaDB, return top N books.
    Returns the raw Chroma `results` object.
    """
    tags_list = tags.get("interest_tags", []) or []
    # fallback
    if not tags_list:
        tags_list = ["fiction", "self-help", "psychology"]

    search_query = " ".join(tags_list)
    user_vec = model.encode(search_query).tolist()

    collection = chroma_client.get_collection("books")
    try:
        results = collection.query(query_embeddings=[user_vec], n_results=n)
    except Exception as e:
        # If collection missing or query failed, return empty structure
        return {"ids": [], "metadatas": [[]], "distances": [[]]}

    return results
