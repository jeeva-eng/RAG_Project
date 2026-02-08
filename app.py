from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from src.search import RAGSearch

import os


# Create app
app = FastAPI()


# Templates folder
templates = Jinja2Templates(directory="templates")


# Static folder
app.mount("/static", StaticFiles(directory="static"), name="static")


# Load RAG once (important for performance)
rag = RAGSearch()
chat_history = []


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

    answer = rag.search_and_summarize(query, top_k=5)

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
