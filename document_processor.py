import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import json

def load_sources_map(map_path: str = "sources.json") -> dict:
    """Loads the sources mapping file."""
    if os.path.exists(map_path):
        with open(map_path, "r") as f:
            return json.load(f)
    return {}

def load_and_process_documents(base_dir: str = "documents/mental_health", max_pages_per_doc: int = 9999) -> List[Document]:
    """
    Recursively scans the documents directory, loads a limited number of pages from PDFs, and splits them into chunks.
    Enriches metadata with descriptive names and source URLs.
    """
    all_chunks = []
    sources_map = load_sources_map()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=400,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    
    # Iterate through all subdirectories in the base_dir
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".pdf"):
                file_path = os.path.join(root, file)
                print(f"Processing (Max {max_pages_per_doc} pages): {file_path}")
                
                try:
                    # PDF loading with page limit
                    loader = PyPDFLoader(file_path)
                    full_docs = loader.load()
                    
                    # Slice pages
                    docs = full_docs[:max_pages_per_doc]
                    
                    # Enrich metadata before splitting (Case-insensitive lookup)
                    source_info = {}
                    for k, v in sources_map.items():
                        if k.lower() == file.lower():
                            source_info = v
                            break
                    
                    doc_name = source_info.get("doc_name", file)
                    source_url = source_info.get("url", "")
                    
                    for doc in docs:
                        # PyPDFLoader provides 0-indexed 'page' in metadata
                        page_num = doc.metadata.get("page", 0) + 1
                        doc.metadata.update({
                            "doc_name": doc_name,
                            "source_url": source_url,
                            "page_label": str(page_num)
                        })
                    
                    # Splitting into chunks
                    chunks = text_splitter.split_documents(docs)
                    all_chunks.extend(chunks)
                    print(f"Successfully processed {file} ({len(docs)} pages) into {len(chunks)} chunks.")
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    
    return all_chunks

if __name__ == "__main__":
    # Test block
    print("Testing document processor...")
    chunks = load_and_process_documents()
    print(f"Total chunks created: {len(chunks)}")
    if chunks:
        print("\n--- Example Chunk ---")
        print(f"Source: {chunks[0].metadata.get('source')}")
        print(f"Content Preview: {chunks[0].page_content[:200]}...")
