# scripts/build_embeddings.py
from modules.embeddings import build_embeddings

if __name__ == "__main__":
    print("Building embeddings from data/book_data.xlsx ...")
    build_embeddings(csv_path="data/book_data.xlsx")
    print("Done. Chroma collection 'books' populated.")
