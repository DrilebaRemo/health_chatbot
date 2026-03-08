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

## 🚀 Future Improvements Roadmap
1.  **Multi-Document Expansion**: 
    - Integrate the **WHO Essential Medicines List** and **NDA Drug Safety Guidelines**.
    - Add a "Source Toggle" in the UI to allow users to filter guidelines.
2.  **Advanced RAG Strategy**:
    - **Parent Document Retrieval**: Store small chunks for search but retrieve larger parent contexts to give the LLM better medical nuance.
    - **Reranking**: Use a cross-encoder to rerank the top-k results for higher medical accuracy.
3.  **Conversational Enhancements**:
    - **Session Memory**: Allow the chatbot to remember previous questions (e.g., follow-up symptoms).
    - **Streaming Responses**: Modify `app.py` to use Streamlit's `st.write_stream` for a more premium feel.
4.  **Local Embedding Optimization**:
    - Experiment with larger models like `all-mpnet-base-v2` if local hardware permits, for better semantic understanding.
5.  **Offline Capability**:
    - Ensure the entire app (Streamlit + Chroma + Local Embeddings) can run without any internet connection, purely as a local medical resource.
1. **Separation of Concerns**: Decided to strictly follow the modular file structure (`document_processor.py`, `vector_store.py`, `rag_chain.py`) for better maintainability.
2. **Environment Management**: Use `.env` for secrets and a robust `.gitignore` from the start.
3. **PowerShell Compatibility**: Used `New-Item` for directory creation to ensure compatibility with Windows environment.

### Errors & Resolutions
- **Dependency Conflict**: `pip install` failed because `langchain-google-genai==1.0.1` required `google-generativeai>=0.4.1`, but `requirements.txt` specified `0.4.0`.
  - **Resolution**: Updated `requirements.txt` to `google-generativeai==0.4.1` and retried installation.

### Running app
venv\Scripts\activate
streamlit run app.py
Whenever you drop new PDFs in there in the future, just remember those two steps:

Delete the chroma_db folder.
Run python vector_store.py (making sure your virtual environment is active).

## Phase 4: Implementation of 4-Layer Guardrails
- Implemented Layer 1: Enhanced System Prompt in rag_chain.py to decline diagnosing and strictly enforce medical disclaimers.
- Implemented Layer 2: Pre-search Input Classification using gemini-flash-latest to catch CRISIS, UNSAFE, and OUT_OF_SCOPE prompts.
- Implemented Layer 3: Post-generation Output Validation using gemini-flash-latest to detect hallucinatory medical diagnoses.
- Implemented Layer 4: Human Escalation Paths in app.py to prominently display emergency helpline numbers on CRISIS flag and gracefully decline out-of-scope requests.
- All test boundaries passed successfully.

## Phase 5: Knowledge Base Expansion
- User successfully added new documents to the documents/ subdirectories.
- App verified to be successfully retrieving from the newly compiled vector store.

## Phase 6: Multi-Provider Rate Limit Workaround
- Updated requirements.txt with langchain-groq.
- Replaced gemini-flash-latest with ChatGroq (llama3-8b-8192) in rag_chain.py for check_input_guardrails and check_output_guardrails.
- Added startup check in app.py to verify GROQ_API_KEY presence.
- Waiting on user to supply the GROQ API Key to proceed with testing.

## Phase 6 (Update): Groq Integration
- Successfully updated app to use llama-3.1-8b-instant via Groq for input/output checks.
- Encountered Google API limit during final RAG generation step due to exhausted free tier (20 requests/day per model on Gemini 3-flash). System functions normally once quota resets.
