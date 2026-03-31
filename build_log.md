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

## Phase 7: Conversational Memory Implementation
- Swapped `RetrievalQA` with `create_history_aware_retriever` and `create_retrieval_chain`.
- Integrated `ChatPromptTemplate` and `MessagesPlaceholder` for context-aware querying.
- Updated `app.py` to maintain message history as a list of `HumanMessage` and `AIMessage` objects for the LangChain pipeline.
- Verified that follow-up questions successfully reference previous context without triggering API exhaustion.

## Phase 8: Context-Aware Guardrails
- Fixed false-positive `OUT_OF_SCOPE` blocks on follow-up questions.
- Updated `check_input_guardrails` to accept `chat_history`.
- Injected recent context into the Llama-3 guardrail prompt.
- Result: Bot correctly identifies ambiguous follow-ups as safe medical queries.

## Phase 9: Enhanced Metadata & Citations
- Created `sources.json` to map PDF filenames to descriptive names and official source URLs.
- Updated `document_processor.py` to inject `doc_name`, `source_url`, and `page_label` (1-indexed) into chunk metadata.
- Refactored `rag_chain.py` to extract structured metadata objects instead of raw source paths.
- Updated `app.py` UI to display clickable citations in the format: **[Document Name](URL) (Page X)**.
- Rebuilt the vector store index to apply metadata changes across the entire knowledge base.
## Phase 10: Mental Health Bot Specialization (Uganda Focus)
- Created a separate `mental-health-bot` branch for domain-specific development.
- Specialized `rag_chain.py` system prompt:
    - Shifted persona to "Compassionate Mental Health Assistant".
    - Emphasized empathy, active listening, and strictly defined non-therapist boundaries.
    - Tailored branding for the Ugandan context and Ministry of Health standards.
- Rebranded `app.py` UI:
    - Updated titles, icons (🧠), and introductory text for mental wellness.
    - Replaced general medical emergency contacts with specialized mental health resources in Uganda (e.g., Uganda Counselling Association, Butabika Hospital).
    - Added a dedicated mental health support section to the sidebar.
- Created `documents/mental_health/` directory to house specialized knowledge base assets.
- Isolated Knowledge Base Architecture:
    - Updated `vector_store.py` to use `chroma_db_mental` as the default persistence directory.
    - Updated `document_processor.py` to use `documents/mental_health` as the default source directory.
    - Ensures total data separation between the mental health branch and the main medical branch.
- Specialized 4-Layer Guardrails:
    - Refined Input Guardrails: Added sensitivity for "severe hopelessness" and "active psychosis" detection.
    - Specialized Crisis Escalation: Integrated mental health specific hotlines (UCA, Butabika) into the emergency response layer.
    - Tone Governance: Enforced empathic validation while maintaining strict non-therapeutic clinical boundaries.
- Knowledge Base Integration:
    - Successfully indexed 5 mental health specialized documents including the Mental Health Act 2018 and Child & Adolescent Policy Guidelines.

## Phase 11: Automated RAG Evaluation (Ragas + Colab)
- Integrated **Ragas** framework for automated quality assessment of the RAG system.
- Implemented `evals/run_ragas_eval.py`:
    - Automated inference pipeline using the existing `ask_health_question` function.
    - Configured Ragas to use Gemini 1.5 Flash as a judge LLM.
    - Local `HuggingFaceEmbeddings` used for metric calculations (Faithfulness, Answer Relevancy, Context Precision/Recall).
- Created `evals/manual_eval_cases.json` with ground truth for testing.
- Developed `evals/COLAB_EVAL_GUIDE.md`:
    - Detailed Google Colab execution steps (Upload -> Setup -> Run).
    - Explicit dependency handling for `langchain-groq` and `langchain-core` to resolve `ModuleNotFoundError`.
    - Instructions for `GOOGLE_API_KEY` and `GROQ_API_KEY` configuration.
- Verified evaluation results tracking in `evals/results/`.
