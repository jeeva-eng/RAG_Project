import os
import faiss
import numpy as np
import pickle
from typing import List, Any, Optional
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline


class FaissVectorStore:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.index: Optional[faiss.Index] = None
        self.metadata: List[Any] = []

        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        print(f"[INFO] Loaded embedding model: {embedding_model}")

        # Auto-load existing index if available
        if self._index_exists():
            self.load()

    # --------------------------------------------------
    # INTERNAL CHECK
    # --------------------------------------------------

    def _index_exists(self) -> bool:
        return (
            os.path.exists(os.path.join(self.persist_dir, "faiss.index")) and
            os.path.exists(os.path.join(self.persist_dir, "metadata.pkl"))
        )

    # --------------------------------------------------
    # BUILD VECTOR STORE
    # --------------------------------------------------

    def build_from_documents(self, documents: List[Any]):

        if not documents:
            raise ValueError("No documents provided to build vector store.")

        print(f"[INFO] Building vector store from {len(documents)} documents...")

        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        chunks = emb_pipe.chunk_documents(documents)

        if not chunks:
            raise ValueError("No chunks generated from documents.")

        print(f"[INFO] Generated {len(chunks)} chunks.")

        embeddings = emb_pipe.embed_chunks(chunks)

        # Convert to numpy FIRST
        embeddings = np.array(embeddings, dtype="float32")

        # ✅ Correct NumPy check
        if embeddings.size == 0:
            raise ValueError("Embedding generation failed.")

        # Metadata handling
        metadatas = [
            {
                "text": chunk.page_content,
                "source": os.path.basename(chunk.metadata.get("source", "Unknown")),
                "page": chunk.metadata.get("page", None),
            }
            for chunk in chunks
        ]

        self.add_embeddings(embeddings, metadatas)
        self.save()

        print("[INFO] Vector store built and saved successfully.")

    # --------------------------------------------------
    # ADD EMBEDDINGS
    # --------------------------------------------------

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        dim = embeddings.shape[1]

        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings)

        if metadatas:
            self.metadata.extend(metadatas)

        print(f"[INFO] Added {embeddings.shape[0]} vectors to index.")

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    def save(self):

        if self.index is None:
            raise ValueError("Cannot save empty index.")

        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        faiss.write_index(self.index, faiss_path)

        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

        print(f"[INFO] Index persisted at '{self.persist_dir}'")

    # --------------------------------------------------
    # LOAD
    # --------------------------------------------------

    def load(self):

        if not self._index_exists():
            raise FileNotFoundError("Persisted FAISS index not found.")

        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        self.index = faiss.read_index(faiss_path)

        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        print("[INFO] FAISS index loaded successfully.")

    # --------------------------------------------------
    # SEARCH
    # --------------------------------------------------

    def search(self, query_embedding: np.ndarray, top_k: int = 5):

        if self.index is None or self.index.ntotal == 0:
            return []

        D, I = self.index.search(query_embedding, top_k)

        results = []

        for idx, dist in zip(I[0], D[0]):
            if 0 <= idx < len(self.metadata):
                results.append({
                    "index": int(idx),
                    "distance": float(dist),
                    "metadata": self.metadata[idx]
                })

        return results

    # --------------------------------------------------
    # QUERY
    # --------------------------------------------------

    def query(self, query_text: str, top_k: int = 5):

        if not query_text.strip():
            return []

        print(f"[INFO] Querying: '{query_text}'")

        query_emb = self.model.encode(
            [query_text],
            convert_to_numpy=True
        ).astype("float32")

        return self.search(query_emb, top_k=top_k)

    # --------------------------------------------------

    def get_all_documents(self):
        return self.metadata