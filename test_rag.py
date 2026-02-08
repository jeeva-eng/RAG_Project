from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from src.vector_db import FaissVectorStore
from src.search import RAGSearch


## Example Usage

if __name__=="__main__":
    #docs=load_all_documents("data")

## Embedding file    
    # pipeline=EmbeddingPipeline()
    # chunks=pipeline.chunk_documents(docs)
    # chunkvectors=pipeline.embed_chunks(chunks)
    # print(chunkvectors)

## Vector store file
  
    # store=FaissVectorStore("faiss_store")
    # #store.build_from_documents(docs)
    # store.load()
    # print(store.query("what is database planning ?",top_k=3))

## Search file

     rag_search=RAGSearch()
     query="What is unsupervised learning? "
     summary=rag_search.search_and_summarize(query,top_k=3)
     print("Summary:",summary)