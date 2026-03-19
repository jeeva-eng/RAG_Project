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
# COLUMN LABEL MAP
# --------------------------------------------------
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


# --------------------------------------------------
# ANALYTICAL QUERY DETECTOR
# Detects duplicate/unique/missing queries so they
# bypass format_answer() and display as-is
# --------------------------------------------------
ANALYTICAL_KEYWORDS = [
    "duplicate", "duplicates", "repeated",
    "unique", "distinct", "how many different",
    "missing", "null", "empty", "nan",
    "most common", "least common", "frequent", "popular",
]

def is_analytical_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in ANALYTICAL_KEYWORDS)


# --------------------------------------------------
# CITY EXTRACTOR
# Fixed: takes only first word after preposition,
# rejects stopwords, prevents "are"/"any" false matches
# --------------------------------------------------
_CITY_STOPWORDS = {
    "me", "the", "a", "an", "all", "my", "this", "that", "data",
    "results", "any", "there", "each", "every", "some", "most",
    "list", "customers", "duplicates", "records", "values", "prefix",
    "count", "rows", "city", "zip", "state", "column", "table",
    "are", "is", "was", "for", "in", "of", "from", "unique",
}

def extract_city(query: str) -> str:
    """
    'list zip codes for osasco'      → 'Osasco'
    'are there any duplicates in zip'→ ''
    'how many customers in sao paulo'→ 'Sao Paulo'
    """
    q = query.lower()
    match = re.search(
        r'\b(?:for|in|of|from)\s+([a-záéíóúãõâêîôûç]\w*(?:\s+\w+)?)',
        q
    )
    if match:
        # take only first word to avoid grabbing extra words
        city = match.group(1).strip().rstrip("?.,").split()[0]
        if city not in _CITY_STOPWORDS and len(city) > 2:
            return city.title()
    return ""


# --------------------------------------------------
# NUMBER GRID FORMATTER
# Turns a flat number dump into a clean aligned grid
# --------------------------------------------------
def format_number_grid(raw: str, query: str = "") -> str:
    """
    Input : 'customer_zip_code_prefix 6290 6286 6140 ...'
    Output: nicely aligned 5-column grid with header & footer
    """
    text   = raw.strip()
    tokens = text.split()

    if len(tokens) < 3:
        return text

    # Separate label words from number tokens
    label_parts  = []
    number_parts = []
    for tok in tokens:
        if re.fullmatch(r'\d+(\.\d+)?', tok):
            number_parts.append(tok)
        else:
            if not number_parts:
                label_parts.append(tok)
            else:
                number_parts.append(tok)  # non-numeric after numbers → keep as-is

    # Not a number grid — return unchanged
    if len(number_parts) < 5:
        return text

    label = friendly_label(" ".join(label_parts))
    city  = extract_city(query)

    WIDE  = "━" * 46
    THIN  = "·" * 46

    # Header
    if city:
        header = f"{WIDE}\n  {label}\n  📍 {city}\n{THIN}"
    else:
        header = f"{WIDE}\n  {label}\n{THIN}"

    # Grid — 5 numbers per row, each 7 chars wide
    PER_ROW = 5
    rows = []
    for i in range(0, len(number_parts), PER_ROW):
        chunk = number_parts[i : i + PER_ROW]
        # right-align each value in a 7-char field, separated by 2 spaces
        rows.append("  ".join(f"{v:>7}" for v in chunk))

    grid   = "\n".join(rows)
    footer = f"{WIDE}\n  ✅ {len(number_parts)} records found"

    return f"{header}\n{grid}\n{footer}"


# --------------------------------------------------
# MAIN ANSWER FORMATTER
# Routes to grid formatter OR returns text as-is
# --------------------------------------------------
def format_answer(raw: str, query: str = "") -> str:
    text = raw.strip()
    if not text:
        return text

    tokens = text.split()

    # Count numeric tokens in first 20 tokens
    numeric_count = sum(
        1 for t in tokens[:20]
        if re.fullmatch(r'\d+(\.\d+)?', t)
    )

    # If mostly numbers → format as grid
    if numeric_count >= 5:
        return format_number_grid(text, query)

    # Otherwise return the text unchanged (RAG answer, analytical result etc.)
    return text


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

    # ── GET ANSWER ──────────────────────────────────────────
    # Analytical queries (duplicates, unique, missing etc.)
    # → send directly to csv_agent, skip number grid formatter
    if is_analytical_query(query):
        raw_answer = rag.csv_agent.query(query)
        answer = raw_answer.strip() if raw_answer else "No results found."
    else:
        # Normal RAG / CSV route
        raw_answer = rag.search_and_summarize(query)
        # Only apply grid formatter for number dumps
        answer = format_answer(raw_answer, query)

    # ── GET SUGGESTIONS ─────────────────────────────────────
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

    # ── BUILD FINAL RESPONSE ────────────────────────────────
    # Format: <answer>
    #         ------------------------------
    #         💡 Suggestions:
    #         1. question
    #         2. question  ...
    # JS parser in index.html splits on "---" and builds chips
    # ────────────────────────────────────────────────────────
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