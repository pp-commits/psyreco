# scripts/build_embeddings.py
from modules.embeddings import build_embeddings

if __name__ == "__main__":
    print("Building embeddings from data/books_sample.csv ...")
    build_embeddings(csv_path="data/books_sample.csv")
    print("Done. Chroma collection 'books' populated.")
