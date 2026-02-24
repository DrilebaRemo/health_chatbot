# Build Log: Healthcare Chatbot Implementation

This file tracks all modifications, errors encountered, and design decisions made during the project.

## [2026-02-24] - Project Initialization

### Initial Setup
- Analyzed `chatbot_implementation_guide.md` to define project scope and tech stack.
- Tech Stack defined:
  - Language: Python 3.8+ (Verified Python 3.12 in use)
  - LLM: Google Gemini 1.5 Flash (Free Tier)
  - Embeddings: Google Generative AI Embeddings
  - Vector DB: ChromaDB (Local)
  - Framework: LangChain
  - UI: Streamlit

### Phase 1: Environment Setup
- Created `requirements.txt` with core dependencies (langchain, streamlit, chromadb, etc.).
- Initialized Python virtual environment (`venv`).
- Created `.env` placeholder for Google Gemini API Key.
- Setup `.gitignore` to exclude sensitive and build-related files.
- Created project directory structure for medical documents:
  - `documents/medications`
  - `documents/clinical_guidelines`
  - `documents/drug_safety`
  - `documents/general_health`
- Successfully installed all dependencies via `pip` (including long-running `chroma-hnswlib` build).
- Verified installation of `langchain`, `streamlit`, `chromadb`, and `google-generativeai`.

### Phase 2: Knowledge Base Gathering (Completed/Partial)
- **Status:** Successfully retrieved the primary "Uganda Clinical Guidelines 2023" (13.5MB).
- **Files obtained:**
    - `documents/clinical_guidelines/Uganda_Clinical_Guidelines_2023.pdf` (13.5MB)
    - `documents/drug_safety/Uganda_NDA_ADR_Guidelines_2019.pdf` (82KB - may be excerpt)
- **Issues:** WHO and some NDA direct PDF links are behind Bitstream/Session protection. Proceeding with the Ugandan guidelines as the core knowledge base.
- **Decision:** The Uganda Clinical Guidelines is comprehensive enough (400+ pages) to build the V1 chatbot.

## Phase 3: Core RAG System Construction (In Progress)
- **Update:** Configured `rag_chain.py` to use local `all-MiniLM-L6-v2` embeddings for retrieval and `gemini-flash-latest` for generation. Tested successfully on complex medical queries (Malaria treatment).
- **Status:** Phase 3 complete.
- **Status:** Implementing `document_processor.py` and `vector_store.py`.
- **Decision (Option 1):** Proceeding with a partial index (first 100 pages) of the Uganda Clinical Guidelines to successfully populate the vector store within the 1000 RPD quota.
- **Action:** Updating `document_processor.py` to support page slicing and `vector_store.py` to handle the reduced chunk set.
1. **Separation of Concerns**: Decided to strictly follow the modular file structure (`document_processor.py`, `vector_store.py`, `rag_chain.py`) for better maintainability.
2. **Environment Management**: Use `.env` for secrets and a robust `.gitignore` from the start.
3. **PowerShell Compatibility**: Used `New-Item` for directory creation to ensure compatibility with Windows environment.

### Errors & Resolutions
- **Dependency Conflict**: `pip install` failed because `langchain-google-genai==1.0.1` required `google-generativeai>=0.4.1`, but `requirements.txt` specified `0.4.0`.
  - **Resolution**: Updated `requirements.txt` to `google-generativeai==0.4.1` and retried installation.
