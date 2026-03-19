import os
from dotenv import load_dotenv

from src.csv_agent import CSVAgent
from src.vector_db import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data")


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
            raise RuntimeError("FAISS index not found. Run build_index.py first.")

        self.vectorstore.load()
        print("[INFO] FAISS index loaded successfully.")

        # --------------------------------------------------
        # CSV AGENT
        # --------------------------------------------------
        self.csv_agent = CSVAgent(
            os.path.join(DATA_PATH, "CSV", "olist_customers_dataset.csv")
        )

        # --------------------------------------------------
        # SOURCES (for UI)
        # --------------------------------------------------
        unique_files = {
            os.path.basename(meta.get("source", "Unknown"))
            for meta in self.vectorstore.metadata
            if meta
        }

        def get_icon(filename: str) -> str:
            ext = filename.split(".")[-1].lower()
            icons = {"pdf": "📄", "csv": "📊", "xlsx": "📊", "xls": "📊", "txt": "📘"}
            return icons.get(ext, "📁")

        self.sources = sorted(
            [{"icon": get_icon(f), "name": f} for f in unique_files],
            key=lambda x: x["name"].lower(),
        )

        # --------------------------------------------------
        # LLM
        # --------------------------------------------------
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("❌ GROQ_API_KEY not found in environment.")

        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=llm_model,
            temperature=0.2,
            max_tokens=300,
        )
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    # --------------------------------------------------
    # ROUTE 1 DETECTOR — CSV
    # --------------------------------------------------
    def is_csv_query(self, query: str) -> bool:
        keywords = [
            "csv", "dataset", "table", "customer", "zip", "city",
            "count", "average", "sum", "filter", "rows", "list",
            "how many", "total", "show", "osasco", "state",
        ]
        return any(word in query.lower() for word in keywords)

    # --------------------------------------------------
    # ROUTE 2 DETECTOR — FAISS context retrieval
    # Returns context string or empty string if not found
    # --------------------------------------------------
    def _get_rag_context(self, query: str, top_k: int = 20) -> str:
        results = self.vectorstore.query(query, top_k=top_k) or []

        if not results:
            return ""

        # ── Similarity threshold ──────────────────────────
        # Tune MIN_SCORE based on your FAISS setup.
        # For cosine similarity: 0.0–1.0 (higher = more similar)
        # Set USE_THRESHOLD = False if your vector_db has no score key.
        SCORE_KEY     = "score"   # change to "distance"/"similarity" if needed
        MIN_SCORE     = 0.30
        USE_THRESHOLD = True

        if USE_THRESHOLD and results[0].get(SCORE_KEY) is not None:
            results = [r for r in results if r.get(SCORE_KEY, 0) >= MIN_SCORE]

        if not results:
            return ""

        texts = [
            r["metadata"].get("text", "")
            for r in results
            if r.get("metadata")
        ]
        return "\n\n".join(texts).strip()

    # --------------------------------------------------
    # ROUTE 3 — Clean "not found" message
    # Shown when neither CSV nor FAISS can answer the query
    # --------------------------------------------------
    def _not_found_response(self, query: str) -> str:
        print("[INFO] No relevant chunks found — returning not-found message.")
        return (
            f"❌  Sorry,this information not found in the documents\n"
            f"{'─' * 44}\n"
            f"  \"{query}\"\n\n"
            f"This topic is not covered in your uploaded documents.\n\n"
            f"💡 Try:\n"
            f"  • Rephrasing your question\n"
            f"  • Asking about topics in the uploaded files\n"
            f"  • Adding a document that covers this topic"
        )
    def search_and_summarize(self, query: str, top_k: int = 20) -> str:
        if not query.strip():
            return "Please enter a valid question."

        # ── ROUTE 1: CSV ─────────────────────────────────
        if self.is_csv_query(query):
            print("[INFO] Routing to CSV Agent...")
            return self.csv_agent.query(query)

        # ── ROUTE 2: RAG (FAISS) ─────────────────────────
        print("[INFO] Routing to FAISS (RAG)...")
        context = self._get_rag_context(query, top_k)

        if context:
            prompt = f"""You are a document-based assistant.

Strict Rules:
- Answer ONLY using the Context below.
- Do NOT use outside knowledge.
- Do NOT guess.
- If the answer is not in the Context, say exactly:
  "Sorry, this information is not in my documents."
- Keep the answer within 5-6 lines.
- Use simple English.

Context:
{context}

Question:
{query}

Answer:"""
            try:
                response = self.llm.invoke(prompt)
                return response.content.strip()
            except Exception as e:
                print(f"[ERROR] RAG LLM failed: {e}")
                return "An error occurred while generating the answer."

        # ── ROUTE 3: NOT FOUND ───────────────────────────
        return self._not_found_response(query)

    # --------------------------------------------------
    # GENERATE FOLLOW-UP SUGGESTIONS
    # --------------------------------------------------
    def generate_suggestions(self, question: str, answer: str) -> list[str]:
        fallback = [
            "Show top 10 results",
            "How many records are there?",
            "Filter by a specific city",
            "Group results by category",
        ]
        try:
            prompt = f"""You are a helpful data assistant.

User asked: {question}
Answer: {answer[:400]}

Generate exactly 4 short follow-up questions.

Rules:
- Each question on its own line, numbered: 1. 2. 3. 4.
- Very short (under 10 words each)
- Relevant to the topic
- No explanations, no extra text
"""
            response = self.llm.invoke(prompt)
            lines = [
                line.strip()
                for line in response.content.split("\n")
                if line.strip()
            ]

            cleaned = []
            for line in lines:
                stripped = line.lstrip("0123456789.-) ").strip()
                if len(stripped) > 3:
                    cleaned.append(stripped)
                if len(cleaned) == 4:
                    break

            return cleaned if cleaned else fallback

        except Exception as e:
            print(f"[WARN] Suggestion generation failed: {e}")
            return fallback

    # --------------------------------------------------
    # GET SOURCES
    # --------------------------------------------------
    def get_sources(self) -> list[dict]:
        return self.sources