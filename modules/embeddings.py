# modules/embeddings.py
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm
import chardet
import csv

# Initialize model and Chroma client
model = SentenceTransformer("intfloat/e5-small-v2")
chroma_client = chromadb.PersistentClient(path="data/chroma_db")
collection = chroma_client.get_or_create_collection("books")


def build_embeddings(csv_path="data/book_data.xlsx"):
    """
    Generate embeddings for all book summaries and store in ChromaDB
    """
    import chardet
    import os

    # --- Detect file type ---
    ext = os.path.splitext(csv_path)[1].lower()
    print(f"Detected file type: {ext}")

    # --- Load data safely ---
    if ext == ".xlsx":
        df = pd.read_excel(csv_path)
    else:
        with open(csv_path, "rb") as f:
            raw_data = f.read(10000)
            enc = chardet.detect(raw_data)["encoding"]

        print(f"Detected encoding: {enc}")

        df = pd.read_csv(
            csv_path,
            encoding=enc or "utf-8",
            engine="python",
            on_bad_lines="skip"
        )

    print(f"Detected columns: {df.columns.tolist()}")

    # --- Validate ---
    required_cols = {"title", "author", "summary", "genre"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"❌ Missing required columns. Found: {df.columns.tolist()}")

    # --- Clear existing ---
    try:
        collection.delete(where={"title": {"$exists": True}})
    except Exception:
        pass

    # --- Embed ---
    from tqdm import tqdm
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Embedding Books"):
        embedding = model.encode(str(row["summary"])).tolist()
        collection.add(
            embeddings=[embedding],
            metadatas=[{
                "title": row["title"],
                "author": row["author"],
                "summary": row["summary"],
                "genre": row["genre"],
                "emotion_tags": row.get("emotion_tags", ""),
                "mindset_tags": row.get("mindset_tags", "")
            }],
            ids=[str(i)]
        )

    print(f"✅ Done. Chroma collection 'books' now has {len(df)} items.")
