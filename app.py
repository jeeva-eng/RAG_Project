from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from fastapi.middleware.sessions import SessionMiddleware

from src.search import RAGSearch

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Create app
app = FastAPI()


# Session middleware (IMPORTANT)
app.add_middleware(
    SessionMiddleware,
    secret_key="rag-secret-key-123"
)


# Templates folder
templates = Jinja2Templates(directory="templates")


# Static folder
app.mount("/static", StaticFiles(directory="static"), name="static")


# Load RAG once (important for performance)
rag = None


def get_rag():
    global rag
    if rag is None:
        rag = RAGSearch()
    return rag


# Home page (Reset chat here)
@app.get("/", response_class=HTMLResponse)
def home(request: Request):

    # Clear old chat on new visit
    request.session["chat"] = []

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "chat": []
        }
    )


# Search page
@app.post("/search", response_class=HTMLResponse)
def search(request: Request, query: str = Form(...)):

    # Get chat from session
    chat = request.session.get("chat", [])

    rag_system = get_rag()
    answer = rag_system.search_and_summarize(query, top_k=5)

    chat.append({
        "user": query,
        "bot": answer
    })

    # Save back to session
    request.session["chat"] = chat

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "chat": chat
        }
    )
