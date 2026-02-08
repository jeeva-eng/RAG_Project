from pathlib import Path
from typing import List,Any
from langchain_community.document_loaders import PyPDFLoader,TextLoader,CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader

def load_all_documents(data_dir:str) -> List[Any]:
    """
    Load all supported files from the data directory and convert to LangChain document structure
    Supported: PDF,TXT,Excel,CSV,Word,JSON
    """
    # Use project root data folder
    data_path=Path(data_dir).resolve()
    print(f"[DEBUG] Data Path:{data_path}")
    documents=[]

    #PDF Files
    pdf_files=list(data_path.glob('**/*.pdf'))
    print(f"[DEBUG] Found {len(pdf_files)} PDF Files:{[str(f) for f in pdf_files]}")
    for pdf_file in pdf_files:
        print(f"[DEBUG] Loading PDF:{pdf_file}")
        try:
            loader=PyPDFLoader(str(pdf_file))
            loaded=loader.load()
            print(f"[DEBUG] Loaded{len(loaded)} PDF docs from {pdf_files}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load PDF {pdf_file}:{e}")

    # Text Files            
    txt_files=list(data_path.glob('**/*.txt'))
    print(f"[DEBUG] Found {len(txt_files)} TXT Files:{[str(f) for f in txt_files]}")
    for pdf_file in pdf_files:
        print(f"[DEBUG] Loading TXT:{txt_files}")
        try:
            loader=TextLoader(str(txt_files))
            loaded=loader.load()
            print(f"[DEBUG] Loaded{len(loaded)} TXT docs from {txt_files}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load TXT {txt_files}:{e}")

            # CSV files
    csv_files = list(data_path.glob('**/*.csv'))
    print(f"[DEBUG] Found {len(csv_files)} CSV files: {[str(f) for f in csv_files]}")
    for csv_file in csv_files:
        print(f"[DEBUG] Loading CSV: {csv_file}")
        try:
            loader = CSVLoader(str(csv_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} CSV docs from {csv_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load CSV {csv_file}: {e}")

    # Excel files
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

    return documents        