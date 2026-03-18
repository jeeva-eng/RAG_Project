import os
import shutil

from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from starlette.middleware.sessions import SessionMiddleware

from src.search import RAGSearch
from src.csv_agent import CSVAgent


# --------------------------------------------------
# ENV SETTINGS
# --------------------------------------------------

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

UPLOAD_FOLDER = "documents"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# --------------------------------------------------
# APP INIT
# --------------------------------------------------

app = FastAPI()

app.add_middleware(
    SessionMiddleware,
    secret_key="rag-secret-key-123"
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# --------------------------------------------------
# LOAD SYSTEMS
# --------------------------------------------------

rag = RAGSearch()

# 🔥 Load CSV (change filename if needed)
csv_path = os.path.join("data", "CSV", "olist_customers_dataset.csv")
csv_agent = CSVAgent(csv_path)

sources = rag.get_sources()


# --------------------------------------------------
# ROUTER LOGIC (IMPORTANT)
# --------------------------------------------------

def is_csv_query(query: str):
    keywords = [
        "zip", "city", "count", "list",
        "average", "sum", "group", "filter",
        "customer", "price", "sales"
    ]

    query = query.lower()
    return any(k in query for k in keywords)


# --------------------------------------------------
# ROUTES
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    request.session["chat"] = []

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "chat": request.session["chat"],
            "sources": sources
        }
    )


@app.post("/search", response_class=HTMLResponse)
def search(request: Request, query: str = Form(...)):

    chat = request.session.get("chat", [])

    # --------------------------------------------------
    # 🔥 SMART ROUTING
    # --------------------------------------------------
    if is_csv_query(query):
        answer = csv_agent.query(query)
    else:
        answer = rag.search_and_summarize(query, top_k=20)

    # --------------------------------------------------
    # 💡 SUGGESTIONS
    # --------------------------------------------------
    suggestions = rag.generate_suggestions(query, answer)

    # Ensure 4 suggestions
    while len(suggestions) < 4:
        suggestions.append("Show more details")

    # --------------------------------------------------
    # 🔥 FINAL RESPONSE (NO HTML CHANGE NEEDED)
    # --------------------------------------------------
    final_answer = f"""{answer}

    
    ------------------------------

    
💡 Suggestions:
1.{suggestions[0]}
2.{suggestions[1]}
3.{suggestions[2]}
4.{suggestions[3]}
"""

    chat.append({
        "user": query,
        "bot": final_answer
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
