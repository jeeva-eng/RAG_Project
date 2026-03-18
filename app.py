import os
import re

from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from starlette.middleware.sessions import SessionMiddleware

from src.search import RAGSearch


# --------------------------------------------------
# ENV SETTINGS
# --------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
# LOAD RAG SYSTEM
# --------------------------------------------------
rag = RAGSearch()
sources = rag.get_sources()


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

# Map raw column names to friendly display names
COLUMN_LABELS = {
    "customer_zip_code_prefix": "📮 ZIP Code Prefix",
    "customer_city":            "🏙️  City",
    "customer_state":           "🗺️  State",
    "customer_id":              "🆔 Customer ID",
    "customer_unique_id":       "🆔 Unique Customer ID",
    "order_id":                 "📦 Order ID",
    "price":                    "💰 Price",
    "product_id":               "🏷️  Product ID",
}

def friendly_label(raw_label: str) -> str:
    key = raw_label.lower().replace(" ", "_")
    return COLUMN_LABELS.get(key, raw_label.replace("_", " ").title())


def extract_city(query: str) -> str:
    """Extract city/filter name from the query, e.g. 'for osasco' → 'Osasco'."""
    q = query.lower()
    match = re.search(
        r'\b(?:for|in|of|from)\s+([a-záéíóúãõâêîôûç][a-záéíóúãõâêîôûç\s]{1,30})',
        q
    )
    if match:
        city = match.group(1).strip().rstrip("?.,")
        stopwords = {"me", "the", "a", "an", "all", "my", "this", "that", "data", "results"}
        if city not in stopwords:
            return city.title()
    return ""


# --------------------------------------------------
# ANSWER FORMATTER
# --------------------------------------------------
def format_answer(raw: str, query: str = "") -> str:
    text   = raw.strip()
    tokens = text.split()

    if len(tokens) < 3:
        return text

    # Split label tokens from number tokens
    label_parts  = []
    number_parts = []
    for tok in tokens:
        if re.fullmatch(r'\d+(\.\d+)?', tok):
            number_parts.append(tok)
        else:
            if not number_parts:
                label_parts.append(tok)
            else:
                number_parts.append(tok)

    if len(number_parts) < 5:
        return text

    # Friendly column label
    raw_label = " ".join(label_parts)
    label     = friendly_label(raw_label)

    # City from query
    city = extract_city(query)

    # ── Header block ─────────────────────────────────────
    WIDTH = 44
    thin  = "·" * WIDTH
    thick = "━" * WIDTH

    if city:
        city_line  = f"  📍 {city}"
        header = (
            f"{thick}\n"
            f"  {label}\n"
            f"{city_line}\n"
            f"{thin}"
        )
    else:
        header = (
            f"{thick}\n"
            f"  {label}\n"
            f"{thin}"
        )

    # ── Number grid: 5 per row ────────────────────────────
    PER_ROW = 5
    rows = []
    for i in range(0, len(number_parts), PER_ROW):
        chunk = number_parts[i : i + PER_ROW]
        rows.append("  ".join(f"{v:>7}" for v in chunk))

    grid = "\n".join(rows)

    # ── Footer ────────────────────────────────────────────
    footer = f"{thick}\n  ✅ {len(number_parts)} records found"

    return f"{header}\n{grid}\n{footer}"


# --------------------------------------------------
# ROUTES
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    request.session["chat"] = []
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "chat": [], "sources": sources},
    )


@app.post("/search", response_class=HTMLResponse)
def search(request: Request, query: str = Form(...)):

    chat = request.session.get("chat", [])

    # ── ANSWER ─────────────────────────────────────────────
    raw_answer = rag.search_and_summarize(query)
    answer     = format_answer(raw_answer, query)

    # ── SUGGESTIONS ────────────────────────────────────────
    suggestions = rag.generate_suggestions(query, answer)

    fallbacks = [
        "Show more details",
        "List top results",
        "Filter by category",
        "Summarize this topic",
    ]
    while len(suggestions) < 4:
        suggestions.append(fallbacks[len(suggestions)])
    suggestions = suggestions[:4]

    # ── FINAL RESPONSE ─────────────────────────────────────
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(suggestions))

    final_answer = (
        f"{answer}\n"
        f"------------------------------\n"
        f"💡 Suggestions:\n"
        f"{numbered}"
    )

    chat.append({"user": query, "bot": final_answer})
    request.session["chat"] = chat

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "chat": chat, "sources": sources},
    )