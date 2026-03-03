from pathlib import Path
from typing import List, Any
import pandas as pd

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_core.documents import Document


def load_all_documents(data_dir: str) -> List[Any]:

    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data Path: {data_path}")

    documents = []

    # ---------------- PDF ----------------
    pdf_files = list(data_path.glob('**/*.pdf'))
    print(f"[DEBUG] Found {len(pdf_files)} PDF Files: {[str(f) for f in pdf_files]}")

    for pdf_file in pdf_files:
        print(f"[DEBUG] Loading PDF: {pdf_file}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} PDF docs from {pdf_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load PDF {pdf_file}: {e}")

    # ---------------- TXT ----------------
    txt_files = list(data_path.glob('**/*.txt'))
    print(f"[DEBUG] Found {len(txt_files)} TXT Files: {[str(f) for f in txt_files]}")

    for txt_file in txt_files:
        print(f"[DEBUG] Loading TXT: {txt_file}")
        try:
            loader = TextLoader(str(txt_file), encoding="utf-8")
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} TXT docs from {txt_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load TXT {txt_file}: {e}")

    # ---------------- CSV (MEMORY SAFE FIX) ----------------
    csv_files = list(data_path.glob('**/*.csv'))
    print(f"[DEBUG] Found {len(csv_files)} CSV files: {[str(f) for f in csv_files]}")

    for csv_file in csv_files:
        print(f"[DEBUG] Loading CSV: {csv_file}")
        try:
            # 🔥 CRITICAL FIX: Limit rows to prevent OOM
            df = pd.read_csv(csv_file).head(1000)

            # Convert dataframe to single large text block
            text_content = df.to_string(index=False)

            doc = Document(
                page_content=text_content,
                metadata={"source": str(csv_file)}
            )

            print(f"[DEBUG] Loaded 1000 rows from {csv_file} as 1 document")
            documents.append(doc)

        except Exception as e:
            print(f"[ERROR] Failed to load CSV {csv_file}: {e}")

    # ---------------- Excel ----------------
    xlsx_files = list(data_path.glob('**/*.xlsx'))
    print(f"[DEBUG] Found {len(xlsx_files)} Excel files: {[str(f) for f in xlsx_files]}")

    for xlsx_file in xlsx_files:
        print(f"[DEBUG] Loading Excel: {xlsx_file}")
        try:
            loader = UnstructuredExcelLoader(str(xlsx_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} Excel docs from {xlsx_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load Excel {xlsx_file}: {e}")

    print(f"[INFO] Total documents loaded: {len(documents)}")
    return documents