# RAG System Evaluation Report

## Project Context

This project evaluates the Retrieval-Augmented Generation (RAG) system used by the Uganda-focused mental well-being chatbot. The goal of the evaluation work was not only to check whether answers "look good", but also to verify that the system is safe, grounded in retrieved evidence, and stable for conversational follow-up questions.

As of April 7, 2026, the evaluation setup in this repository combines deterministic application checks with semantic RAG quality assessment.

## System That Was Evaluated

The evaluated chatbot pipeline is the real application pipeline in `rag_chain.py`, not a mocked demo.

The main flow is:

1. User question enters the chatbot.
2. Input guardrails classify the question as `SAFE`, `CRISIS`, `UNSAFE`, or `OUT_OF_SCOPE`.
3. If safe, chat history is used to reformulate follow-up questions into standalone queries.
4. The retriever searches the local Chroma vector store.
5. Gemini generates the final answer using retrieved context.
6. Output guardrails validate that the answer remains safe.
7. Sources are returned with document name, URL, page, and retrieved chunk text.

## Tools, Libraries, and Components Used

### Core RAG stack

- `LangChain`
- `LangChain Classic` chains for retrieval and history-aware QA
- `ChromaDB` as the local vector database
- `HuggingFaceEmbeddings` with `all-MiniLM-L6-v2`
- `ChatGoogleGenerativeAI` with `gemini-flash-latest` for answer generation
- `ChatGroq` with `llama-3.1-8b-instant` for guardrail checks
- `Streamlit` for the chatbot UI

### Evaluation stack

- `Ragas` for semantic RAG evaluation
- `datasets` from Hugging Face to build evaluation datasets
- JSON test datasets in `evals/`
- Local JSON result artifacts in `evals/results/`

### Knowledge-base and retrieval assets

- `document_processor.py` for PDF loading, chunking, and metadata enrichment
- `vector_store.py` for embedding and persistent Chroma storage
- `documents/mental_health/` as the mental-health source corpus
- `sources.json` for mapping PDF files to human-readable names and URLs
- `chroma_db_mental_v2/` as the persisted vector store used by the current mental-health bot

## Files Used for Evaluation

### Main evaluation scripts

- `evals/run_app_eval.py`
- `evals/run_ragas_eval.py`
- `evals/run_all_evals.py`

### Evaluation datasets

- `evals/manual_eval_cases.json`
- `evals/guardrail_eval_cases.json`

### Supporting documentation

- `evals/README.md`
- `evals/EVALUATION_PROCESS.md`
- `evals/COLAB_EVAL_GUIDE.md`
- `build_log.md`

### Application files that directly support evaluation

- `rag_chain.py`
- `document_processor.py`
- `vector_store.py`
- `requirements.txt`

## What Exactly Was Evaluated

The evaluation work covered two main layers.

### 1. Deterministic application evaluation

This layer checks engineering behavior with strict pass/fail rules.

It validates:

- status classification
- source presence
- source matching
- required disclaimer text
- forbidden phrases such as second-person diagnostic wording
- source chunk text availability
- follow-up question handling with serialized chat history

This is handled by `evals/run_app_eval.py`.

### 2. Semantic RAG quality evaluation

This layer checks whether the answer is semantically grounded in retrieved evidence.

It uses Ragas metrics:

- `Faithfulness`
- `ResponseRelevancy`
- `LLMContextPrecisionWithReference`
- `LLMContextRecall`

This is handled by `evals/run_ragas_eval.py`.

## Evaluation Cases That Were Prepared

### Manual RAG cases

The manual dataset in `evals/manual_eval_cases.json` contains three main cases:

- patient rights under the Mental Health Act 2018
- support for adolescents facing emotional distress
- a follow-up conversational question: "What about for children?"

Each case includes:

- question
- optional chat history
- expected status
- expected source hint
- required phrases
- forbidden phrases
- whether sources are required
- a `ground_truth` answer for Ragas

### Guardrail cases

The guardrail dataset in `evals/guardrail_eval_cases.json` contains cases for:

- crisis intent
- safety/jailbreak attempts
- out-of-scope requests

## Important Implementation Details Used in Evaluation

Several code-level choices were necessary to make evaluation meaningful.

### Real source text is returned in `sources`

`rag_chain.py` was updated so each returned source now includes:

- `doc_name`
- `url`
- `page`
- `content`

The `content` field is critical because Ragas needs the retrieved chunk text, not just citation metadata.

### Chat history is normalized for evaluation

The evaluation scripts convert JSON chat history into LangChain `HumanMessage` and `AIMessage` objects before calling `ask_health_question()`.

This was required to properly test follow-up questions.

### Prompting was tightened to reduce diagnostic wording

The system prompt in `rag_chain.py` explicitly tells the model not to use second-person diagnostic phrases such as:

- "you have"
- "you are suffering from"
- "this means you have"

This change mattered because early evaluation runs failed on this exact issue.

### Judge model and compatibility handling were added

`evals/run_ragas_eval.py` was updated to:

- use `parse_known_args()` for notebook compatibility
- normalize chat history
- support different Ragas result object formats
- default to `gemini-flash-latest`
- allow override with `RAGAS_JUDGE_MODEL`

## What We Did During the Evaluation Work

The following evaluation work was implemented in the repository:

1. Built a local deterministic evaluation harness around the real `ask_health_question()` function.
2. Added manual RAG test cases with expected outputs and ground-truth references.
3. Added dedicated guardrail regression cases.
4. Added a semantic evaluation path using Ragas.
5. Updated the app pipeline so retrieved chunk text is exposed to evaluators.
6. Fixed chat-history handling for follow-up evaluation.
7. Tightened prompting to reduce unsafe or diagnostic phrasing.
8. Improved compatibility for Colab and notebook execution.
9. Added documentation so the evaluation process can be repeated.

## Problems Found and Fixes Applied

### 1. Follow-up evaluation could break because chat history was JSON

Problem:
The saved evaluation data used plain JSON objects, but the app expected LangChain message objects.

Fix:
History normalization was added in the evaluation code.

### 2. Semantic evaluation originally had weak evidence input

Problem:
The evaluator could only see source metadata, not the retrieved chunk text.

Fix:
`rag_chain.py` was updated so returned sources include the chunk `content`.

### 3. Early manual runs failed because of diagnostic wording

Problem:
Some answers used second-person language like "you have", which violated the expected response policy.

Fix:
The system prompt was tightened, and later runs passed.

### 4. Colab and Ragas compatibility issues appeared

Problem:
Package/API behavior differed across environments and Ragas versions.

Fix:
The evaluation script was made more defensive, including result normalization and notebook-safe argument parsing.

### 5. Judge model naming and quota limits caused semantic-eval instability

Problem:
Older Gemini model naming and free-tier quota exhaustion could stop semantic runs.

Fix:
The default judge model was changed to `gemini-flash-latest`, and the workflow now treats deterministic evaluation as the stable daily baseline.

## Recorded Evaluation Results in This Repository

### Deterministic manual evaluation history

There are three saved result files for the manual deterministic dataset:

- `evals/results/manual_eval_cases_20260331T071155Z.json`
- `evals/results/manual_eval_cases_20260331T071941Z.json`
- `evals/results/manual_eval_cases_20260331T073103Z.json`

Observed progression:

- On March 31, 2026 at `07:11:55 UTC`, manual eval passed `2/3` cases.
- On March 31, 2026 at `07:19:41 UTC`, manual eval again passed `2/3` cases.
- On March 31, 2026 at `07:31:03 UTC`, manual eval passed `3/3` cases.

The failing issue in the earlier runs was the `must_not_include_match` check, caused by disallowed second-person diagnostic phrasing. After prompt tightening, all three manual cases passed.

### Guardrail regression results

The saved guardrail result file is:

- `evals/results/guardrail_eval_cases_20260331T071432Z.json`

Recorded outcome:

- On March 31, 2026 at `07:14:32 UTC`, guardrail regression passed `3/3` cases.

That means the current saved evaluation artifacts show correct handling for:

- crisis detection
- unsafe/jailbreak refusal
- out-of-scope refusal

### Semantic RAG evaluation status

The repository contains the full semantic evaluation pipeline in `evals/run_ragas_eval.py`, but there is no committed `ragas_*.json` result artifact in `evals/results/` at the moment.

Based on the code, documentation, and build log, the Ragas path was implemented and debugged, but practical runs can still be interrupted by:

- Gemini free-tier quota exhaustion
- dependency/version drift
- notebook environment issues

So the semantic evaluation path is structurally in place, while deterministic evaluation is the most reliable recorded evidence currently saved in the repo.

## Current Evaluation Strengths

- Evaluation is run against the real application pipeline.
- Safety behavior is tested separately from answer quality.
- Retrieved chunk text is now included, making semantic evaluation meaningful.
- Conversational follow-up behavior is covered.
- Result artifacts are saved as JSON for traceability.
- The evaluation process is documented in multiple repo files.

## Current Limitations

- The manual dataset is still small.
- Ragas depends on external model quota and package compatibility.
- There is no committed semantic score artifact in `evals/results/` yet.
- Some answer quality checks still rely on phrase heuristics rather than a larger benchmark set.

## Recommended Way to Run the Evaluation

### Daily regression

```bash
python evals/run_app_eval.py --dataset evals/manual_eval_cases.json
python evals/run_app_eval.py --dataset evals/guardrail_eval_cases.json
```

### Semantic RAG review

```bash
python evals/run_ragas_eval.py --dataset evals/manual_eval_cases.json
```

### Combined view

```bash
python evals/run_all_evals.py --dataset evals/manual_eval_cases.json
```

## Final Summary

The RAG evaluation work in this chatbot project produced a practical two-layer evaluation system:

- a deterministic layer for behavior, safety, citations, and follow-up stability
- a semantic layer for grounding and retrieval quality using Ragas

The repository shows that the deterministic evaluation harness is working and has recorded successful results. It also shows that the semantic evaluation pipeline has been implemented and prepared, even though its repeatability is more sensitive to model quota and dependency conditions.

In short, the chatbot now has a real evaluation process for its RAG system instead of relying only on informal testing.
