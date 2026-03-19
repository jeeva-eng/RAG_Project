import os
from src.data_loader import load_all_documents
from src.vector_db import FaissVectorStore

DATA_PATH = "data"
FAISS_PATH = "faiss_store"


def build_index():
    print("[INFO] Loading documents...")
    documents = load_all_documents(DATA_PATH)

    print(f"[INFO] Total documents: {len(documents)}")

    vectorstore = FaissVectorStore(
        persist_dir=FAISS_PATH,
        embedding_model="all-MiniLM-L6-v2"
    )

    print("[INFO] Creating FAISS index...")
    vectorstore.build_from_documents(documents)   

    print("[INFO] Saving FAISS index...")
    vectorstore.save()

    print("[SUCCESS] FAISS index built successfully!")


if __name__ == "__main__":
    build_index()