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
            [{"icon": get_icon(file), "name": file} for file in unique_files],
            key=lambda x: x["name"].lower()
        )

        # --------------------------------------------------
        # LLM
        # --------------------------------------------------
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("❌ GROQ_API_KEY not found")

        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=llm_model,
            temperature=0.2,
            max_tokens=300,
        )

        print(f"[INFO] Groq LLM initialized: {llm_model}")

    # --------------------------------------------------
    # ROUTER HELPER
    # --------------------------------------------------
    def is_csv_query(self, query: str) -> bool:
        query = query.lower()
        csv_keywords = [
            "csv", "dataset", "table", "customer", "zip", "city",
            "count", "average", "sum", "filter", "rows"
        ]
        return any(word in query for word in csv_keywords)

    # --------------------------------------------------
    # MAIN SEARCH METHOD
    # --------------------------------------------------
    def search_and_summarize(self, query: str, top_k: int = 20) -> str:

        if not query.strip():
            return "Please enter a valid question."

        # 🔥 CSV ROUTE
        if self.is_csv_query(query):
            print("[INFO] Routing to CSV Agent...")
            return self.csv_agent.query(query)

        # 🔥 RAG ROUTE
        print("[INFO] Routing to FAISS (RAG)...")
        results = self.vectorstore.query(query, top_k=top_k) or []

        if not results:
            return "Sorry, this information is not in my documents."

        texts = [r["metadata"].get("text", "") for r in results if r.get("metadata")]
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

    # --------------------------------------------------
    # GENERATE FOLLOW-UP SUGGESTIONS
    # --------------------------------------------------
    def generate_suggestions(self, question, answer):
        try:
            prompt = f"""
You are a helpful data assistant.

User asked:
{question}

Answer:
{answer[:500]}

Generate 4 short follow-up questions.

Rules:
- Very short
- Relevant to data
- No explanation
"""

            response = self.llm.invoke(prompt)

            suggestions = [
                s.strip("- ").strip()
                for s in response.content.split("\n")
                if s.strip()
            ]

            return suggestions[:4]

        except Exception:
            # fallback suggestions
            return [
                "Show top 10 results",
                "Summarize data",
                "Filter results",
                "Group by category"
            ]

    # --------------------------------------------------
    # GET SOURCES
    # --------------------------------------------------
    def get_sources(self):
        return self.sources