# scripts/debug_chroma.py
from modules.embeddings import chroma_client

collection = chroma_client.get_collection("books")
print("Total items in 'books':", collection.count())
