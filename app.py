import os
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from starlette.middleware.sessions import SessionMiddleware

from src.search import RAGSearch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI()

app.add_middleware(
    SessionMiddleware,
    secret_key="rag-secret-key-123"
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ✅ Load RAG once (this handles build/load internally)
rag = RAGSearch()

sources = rag.get_sources()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    request.session["chat"] = []

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "chat": [],
            "sources": sources
        }
    )


@app.post("/search", response_class=HTMLResponse)
def search(request: Request, query: str = Form(...)):
    chat = request.session.get("chat", [])

    answer = rag.search_and_summarize(query, top_k=20)

    chat.append({
        "user": query,
        "bot": answer
    })

    request.session["chat"] = chat

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "chat": chat,
            "sources": sources
        }
    )