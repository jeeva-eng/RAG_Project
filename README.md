# RAG Project (Retrieval-Augmented Generation)

This is a simple RAG-based AI application that allows users to ask questions from their own documents.

It uses:
- FAISS for vector storage
- Sentence Transformers for embeddings
- Groq LLM for answering
- LangChain for document processing

---

## 🚀 Features

- Upload and process documents
- Store embeddings in FAISS
- Search relevant chunks
- Generate clean answers (no sources)
- Fast and lightweight

---

## 📁 Project Structure

RAG_Project/
│
├── data/ # Input documents
├── faiss_store/ # Vector database
├── src/ # Core logic
├── templates/ # HTML files
├── static/ # CSS files
├── app.py # Main app
├── requirements.txt # Dependencies
└── README.md


---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/jeeva-eng/RAG_Project.git
cd RAG_Project
2. Create Virtual Environment
python -m venv .venv
.venv\Scripts\activate
3. Install Dependencies
pip install -r requirements.txt
4. Set Environment Variable
Create .env file:

GROQ_API_KEY=YOUR_GROQ_API_KEY


▶️ Run Project
uvicorn app:app --reload

Open in browser:
http://127.0.0.1:8000

📌 Author
Jeeva Nandhan
AI & Data Science Developer


Deployment cleaned
