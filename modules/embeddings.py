# modules/embeddings.py
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("intfloat/e5-small-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("books")

def build_embeddings(csv_path="data/books_sample.csv"):
    """
    Generate embeddings for all book summaries and store in ChromaDB
    """
    df = pd.read_csv(csv_path)
    for i, row in df.iterrows():
        embedding = model.encode(row['summary']).tolist()
        collection.add(
            embeddings=[embedding],
            metadatas=[{
                "title": row['title'],
                "author": row['author'],
                "summary": row['summary'],
                "genre": row['genre']
            }],
            ids=[str(i)]
        )
