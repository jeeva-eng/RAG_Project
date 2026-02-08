import os
from dotenv import load_dotenv

from src.vector_db import FaissVectorStore
from src.data_loader import load_all_documents

from langchain_groq import ChatGroq


# Load .env file
load_dotenv()


class RAGSearch:

    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama-3.1-8b-instant"
    ):

        # Initialize vector store
        self.vectorstore = FaissVectorStore(
            persist_dir,
            embedding_model
        )

        # Check if FAISS exists
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")

        # Build if not exists
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):

            print("[INFO] Building vector store...")

            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)

        else:

            print("[INFO] Loading vector store...")

            self.vectorstore.load()

        # Load API key safely
        groq_api_key = os.getenv("GROQ_API_KEY")

        if not groq_api_key:
            raise ValueError("❌ GROQ_API_KEY not found in environment variables")

        # Initialize LLM (increase max tokens)
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=llm_model,
            temperature=0.2,
            max_tokens=300   # 👈 IMPORTANT
        )

        print(f"[INFO] Groq LLM initialized: {llm_model}")


    def search_and_summarize(self, query: str, top_k: int = 5) -> str:

        # Search in vector DB
        results = self.vectorstore.query(query, top_k=top_k) or []

        # Collect sources
        sources = []

        for r in results:
            if r.get("metadata"):
                src = r["metadata"].get("source", "Unknown")
                sources.append(src)

        source_text = ", ".join(set(sources))


        # Collect text
        texts = [
            r["metadata"].get("text", "")
            for r in results
            if r.get("metadata")
        ]

        context = "\n\n".join(texts)

        if not context:
            return "No relevant documents found."


        # Strong prompt for long answers
        prompt = f"""
You are a helpful AI tutor.

Give a SHORT and CLEAR answer.

Rules:
- Limit to 5–6 lines
- Use simple English
- No headings
- No long explanations
- No conclusion
- No extra examples

Question:
{query}

Context:
{context}

Answer:
"""

        # Call LLM
        response = self.llm.invoke(prompt)

        final_answer = response.content.strip()

        return final_answer
