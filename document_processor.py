import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_and_process_documents(base_dir: str = "documents", max_pages_per_doc: int = 9999) -> List[Document]:
    """
    Recursively scans the documents directory, loads a limited number of pages from PDFs, and splits them into chunks.
    """
    all_chunks = []
    
    # Text splitter configuration
    # Larger chunks to minimize API requests and maintain more local context
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
