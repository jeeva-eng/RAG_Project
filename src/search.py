import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data")

from src.vector_db import FaissVectorStore
from src.data_loader import load_all_documents
from langchain_groq import ChatGroq


load_dotenv()


class RAGSearch:

    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama-3.1-8b-instant",
    ):

        # --------------------------------------------------
        # VECTOR STORE
        # --------------------------------------------------

        self.vectorstore = FaissVectorStore(
            persist_dir=persist_dir,
            embedding_model=embedding_model,
        )

        if not self.vectorstore._index_exists():

            print("[INFO] No existing index found. Building vector store...")

            documents = load_all_documents(DATA_PATH)

            if not documents:
                raise ValueError("No documents found in data folder.")

            self.vectorstore.build_from_documents(documents)

        else:
            print("[INFO] Loading existing vector store...")
            self.vectorstore.load()

        # --------------------------------------------------
        # SOURCES WITH ICONS
        # --------------------------------------------------

        unique_files = {
            os.path.basename(meta.get("source", "Unknown"))
            for meta in self.vectorstore.metadata
            if meta
        }

        def get_icon(filename: str):
            ext = filename.split(".")[-1].lower()

            if ext == "pdf":
                return "📄"
            elif ext in ["csv", "xlsx", "xls"]:
                return "📊"
            elif ext == "txt":
                return "📘"
            else:
                return "📁"

        self.sources = sorted(
            [
                {
                    "icon": get_icon(file),
                    "name": file
                }
                for file in unique_files
            ],
            key=lambda x: x["name"].lower()
        )

        # --------------------------------------------------
        # LLM INITIALIZATION
        # --------------------------------------------------

        groq_api_key = os.getenv("GROQ_API_KEY")

        if not groq_api_key:
            raise ValueError("❌ GROQ_API_KEY not found in environment variables")

        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=llm_model,
            temperature=0.2,
            max_tokens=300,
        )

        print(f"[INFO] Groq LLM initialized: {llm_model}")

    # --------------------------------------------------
    # PUBLIC METHODS
    # --------------------------------------------------

    def get_sources(self):
        return self.sources

    def search_and_summarize(self, query: str, top_k: int = 20) -> str:

        if not query.strip():
            return "Please enter a valid question."

        results = self.vectorstore.query(query, top_k=top_k) or []

        if not results:
            return "Sorry, this information is not in my documents."

        texts = [
            r["metadata"].get("text", "")
            for r in results
            if r.get("metadata")
        ]

        context = "\n\n".join(texts)

        if not context.strip():
            return "Sorry, this information is not in my documents."

        prompt = f"""
You are a document-based assistant.

Strict Rules:
- Answer ONLY using the Context below.
- Do NOT use outside knowledge.
- Do NOT guess.
- If answer is not in Context, say exactly:
  "Sorry, this information is not in my documents."
- Keep answer within 5–6 lines.
- Use simple English.

Context:
{context}

Question:
{query}

Answer:
"""

        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()

        except Exception as e:
            print(f"[ERROR] LLM failed: {e}")
            return "An error occurred while generating the answer."