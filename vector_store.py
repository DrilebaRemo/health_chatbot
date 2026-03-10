import os
from typing import List
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Load environment variables (API Key)
load_dotenv()

# Global cache to avoid reloading the model on every request
_EMBEDDINGS_CACHE = None

def get_embeddings():
    """
    Singleton for loading embeddings to avoid reloading from disk.
    """
    global _EMBEDDINGS_CACHE
    if _EMBEDDINGS_CACHE is None:
        print("Loading local embeddings model (all-MiniLM-L6-v2) into memory...")
        _EMBEDDINGS_CACHE = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _EMBEDDINGS_CACHE

def initialize_vector_store(chunks: List[Document], persist_directory: str = "chroma_db"):
    """
    Creates a Chroma vector store from document chunks using local HuggingFace embeddings.
    """
    embeddings = get_embeddings()
    
    print(f"Creating vector store in {persist_directory}...")
    
    # Local embeddings are free and fast, so we can process all at once
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    vector_store.persist()
    print("Vector store created and persisted successfully.")
    return vector_store

def get_vector_store(persist_directory: str = "chroma_db"):
    """
    Loads an existing Chroma vector store from disk.
    """
    embeddings = get_embeddings()
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vector_store

if __name__ == "__main__":
    # Test block
    from document_processor import load_and_process_documents
    
    print("Running vector store test...")
    if not os.path.exists("chroma_db"):
        print("Loading documents (Full Indexing with Local Embeddings)...")
        # Local embeddings are quota-free, so we process the whole document set
        chunks = load_and_process_documents()
        if chunks:
            initialize_vector_store(chunks)
        else:
            print("No documents found to process.")
    else:
        print("Vector store already exists. Loading...")
        db = get_vector_store()
        print("Test search...")
        results = db.similarity_search("What is the treatment for malaria?", k=2)
        for i, res in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"Source: {res.metadata.get('source')}")
            print(f"Content: {res.page_content[:200]}...")
