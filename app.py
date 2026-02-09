from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from src.search import RAGSearch

import os
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Create app
app = FastAPI()


# Templates folder
templates = Jinja2Templates(directory="templates")


# Static folder
app.mount("/static", StaticFiles(directory="static"), name="static")


# Load RAG once (important for performance)
rag = None
chat_history = []

def get_rag():
    global rag
    if rag is None:
        rag = RAGSearch()
    return rag


# Home page
@app.get("/", response_class=HTMLResponse)
def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


# Search page
@app.post("/search", response_class=HTMLResponse)
def search(request: Request, query: str = Form(...)):

    global chat_history

    rag_system = get_rag()
    answer = rag_system.search_and_summarize(query, top_k=5)

    chat_history.append({
        "user": query,
        "bot": answer
    })

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "chat": chat_history
        }
    )
