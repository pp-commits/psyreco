# modules/recommender.py
from modules.embeddings import model, chroma_client
from typing import Dict
import numpy as np

def jaccard_similarity(list1, list2):
    """Compute Jaccard similarity between two lists of tags."""
    set1, set2 = set(list1), set(list2)
    if not set1 and not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

def recommend_books(tags: Dict, n: int = 5, alpha: float = 0.7, beta: float = 0.3):
    """
    Stage 2 weighted recommendation:
    - alpha: weight for semantic similarity
    - beta: weight for mood/tag similarity
    Returns top N book recommendations with weighted scores and ensures uniqueness.
    """
    tags_list = tags.get("interest_tags", []) or []
    if not tags_list:
        tags_list = ["fiction", "self-help", "psychology"]

    # 1. Encode user tags into embedding vector
    search_query = " ".join(tags_list)
    user_vec = model.encode(search_query).tolist()

    # 2. Query Chroma collection
    collection = chroma_client.get_collection("books")
    try:
        results = collection.query(query_embeddings=[user_vec], n_results=50)
    except Exception as e:
        return {"ids": [], "metadatas": [[]], "distances": [[]]}

    # 3. Compute weighted scores
    final_scores = []
    for i, metadata in enumerate(results["metadatas"][0]):
        # Semantic similarity from Chroma
        semantic_score = 1 - results["distances"][0][i]  # convert distance -> similarity

        # Mood/Tag similarity (Jaccard on tags)
        book_tags = metadata.get("emotion_tags", []) + metadata.get("mindset_tags", [])
        mood_score = jaccard_similarity(tags_list, book_tags)

        # Weighted final score
        weighted_score = alpha * semantic_score + beta * mood_score
        final_scores.append((i, weighted_score))

    # 4. Sort books by weighted score (descending)
    final_scores.sort(key=lambda x: x[1], reverse=True)

    # 5. Deduplicate by title+author and select top N
    seen = set()
    unique_indices = []
    for i, _ in final_scores:
        md = results["metadatas"][0][i]
        key = f"{md.get('title')}|{md.get('author')}"
        if key not in seen:
            seen.add(key)
            unique_indices.append(i)
        if len(unique_indices) >= n:
            break

    # 6. Prepare final top N results
    top_metadatas = [results["metadatas"][0][i] for i in unique_indices]
    top_ids = [results["ids"][0][i] for i in unique_indices]
    top_distances = [results["distances"][0][i] for i in unique_indices]

    return {"ids": top_ids, "metadatas": [top_metadatas], "distances": [top_distances]}
