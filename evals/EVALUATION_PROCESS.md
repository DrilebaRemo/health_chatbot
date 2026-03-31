# RAG Evaluation Process

This document explains the full evaluation workflow used for the Uganda mental well-being chatbot RAG system. It covers what was evaluated, why it was evaluated, which files implement the evaluation, how the scripts work, what problems were found and fixed, and how to run the process going forward.

---

## 1. Evaluation Goal

The chatbot is a Retrieval-Augmented Generation system. That means a user question is:

1. checked by guardrails
2. reformulated with chat history when needed
3. used to retrieve relevant chunks from the vector store
4. answered by the language model using the retrieved context
5. checked again by output guardrails

Because of that architecture, evaluation had to cover more than just "did the model answer correctly?"

We needed to evaluate:

- application behavior
- guardrail behavior
- retrieval behavior
- answer quality
- citation/source behavior
- conversational follow-up behavior
- semantic grounding quality

That led to two evaluation layers:

1. deterministic application checks
2. semantic RAG quality checks with Ragas

---

## 2. What We Are Evaluating

### A. Deterministic Technical Checks

These are strict pass/fail checks against expected system behavior.

They validate:

- expected response status such as `SAFE`, `CRISIS`, `UNSAFE`, or `OUT_OF_SCOPE`
- whether required source documents are returned
- whether source chunk text is actually present
- whether the answer includes required phrases such as the disclaimer
- whether the answer avoids forbidden phrases such as diagnostic wording
- whether follow-up conversation history from JSON works without crashing

These checks are useful because they are reproducible and do not depend on an external LLM judge.

### B. Semantic RAG Checks

These use Ragas to evaluate how good the RAG system is semantically.

The target metrics are:

- `Faithfulness`: whether the answer is grounded in the retrieved context
- `ResponseRelevancy`: whether the answer addresses the user question
- `Context Precision`: whether the retrieved chunks are relevant
- `Context Recall`: whether the retrieved chunks contain enough information for the ground truth

These checks are useful because they go beyond exact phrase matching and estimate whether the system is actually behaving like a good RAG pipeline.

---

## 3. Files Created or Updated

### Core Evaluation Files

- [run_app_eval.py](/C:/Users/user/Desktop/health_chatbot/evals/run_app_eval.py)
- [run_ragas_eval.py](/C:/Users/user/Desktop/health_chatbot/evals/run_ragas_eval.py)
- [run_all_evals.py](/C:/Users/user/Desktop/health_chatbot/evals/run_all_evals.py)
- [manual_eval_cases.json](/C:/Users/user/Desktop/health_chatbot/evals/manual_eval_cases.json)
- [guardrail_eval_cases.json](/C:/Users/user/Desktop/health_chatbot/evals/guardrail_eval_cases.json)
- [README.md](/C:/Users/user/Desktop/health_chatbot/evals/README.md)
- [COLAB_EVAL_GUIDE.md](/C:/Users/user/Desktop/health_chatbot/evals/COLAB_EVAL_GUIDE.md)
- [EVALUATION_PROCESS.md](/C:/Users/user/Desktop/health_chatbot/evals/EVALUATION_PROCESS.md)

### Application Files Updated to Support Evaluation

- [rag_chain.py](/C:/Users/user/Desktop/health_chatbot/rag_chain.py)
- [requirements.txt](/C:/Users/user/Desktop/health_chatbot/requirements.txt)

---

## 4. What Each File Does

### `evals/run_app_eval.py`

This is the deterministic evaluation runner.

It:

- loads test cases from a JSON file
- converts serialized chat history into LangChain `HumanMessage` and `AIMessage` objects
- calls `ask_health_question()`
- captures the answer, sources, status, and latency
- performs pass/fail checks
- writes a JSON report to `evals/results/`

The checks currently include:

- `status_match`
- `source_match`
- `must_include_match`
- `must_not_include_match`
- `sources_present_match`
- `source_content_present_match`

This script is the stable baseline evaluation tool.

### `evals/run_ragas_eval.py`

This is the semantic evaluation runner using Ragas.

It:

- loads test cases from JSON
- converts serialized chat history into LangChain messages
- calls `ask_health_question()`
- extracts the answer and retrieved chunk text
- builds a HuggingFace `Dataset` for Ragas
- runs semantic evaluation metrics
- normalizes Ragas results across versions
- writes a JSON report to `evals/results/`

Important implementation details:

- it uses `parse_known_args()` so it can run in notebook environments without crashing on Jupyter kernel arguments
- it normalizes chat history before passing it into the app
- it uses retrieved chunk text, not just source metadata
- it now defaults the judge model to `gemini-flash-latest`
- it can read newer `EvaluationResult` objects instead of assuming a plain dict

### `evals/run_all_evals.py`

This is the convenience runner that executes both evaluation paths.

It:

- runs deterministic checks first
- saves the deterministic report
- then attempts the Ragas evaluation
- captures and reports Ragas failures cleanly instead of crashing the entire process

This is useful when you want one command to summarize the entire evaluation state.

### `evals/manual_eval_cases.json`

This dataset contains the main retrieval and answer-quality test cases.

It includes:

- questions
- optional chat history
- expected status
- expected source fragment
- required phrases
- forbidden phrases
- whether sources are required
- ground truth text for Ragas

It also includes a conversational follow-up case to verify that evaluation handles serialized chat history correctly.

### `evals/guardrail_eval_cases.json`

This dataset contains guardrail-specific cases for:

- crisis detection
- unsafe prompt handling
- out-of-scope refusal

It is used to validate that the application-level safety contract still holds after changes.

### `rag_chain.py`

This is the main application logic file.

For evaluation support, it was updated so that:

- `sources` now include retrieved chunk `content`
- the prompt more explicitly avoids diagnostic second-person phrases like `you have`

Without the `content` field, semantic evaluation would have been meaningless because Ragas would have received empty evidence strings.

### `requirements.txt`

This was updated to include local evaluation dependencies such as:

- `datasets`
- `ragas`
- `langchain-core`
- `langchain-classic`
- `langchain-groq`

This helps make local evaluation reproducible.

### `evals/README.md`

This is the short operational guide for the evaluation harness.

It explains:

- what each evaluation script does
- which datasets exist
- how to run each evaluator
- how to interpret the outputs

### `evals/COLAB_EVAL_GUIDE.md`

This was rewritten to document the Colab flow more accurately.

It now points to:

- `documents/`
- `chroma_db_mental_v2/`

and includes the missing setup packages.

In practice, local execution turned out to be more reliable than Colab for this project.

---

## 5. Problems Found During Evaluation

Several important issues were discovered while building and running the evaluation harness.

### 1. Chat History Serialization Bug

Problem:

- `manual_eval_cases.json` included chat history as plain JSON objects
- `ask_health_question()` expected LangChain message objects
- the evaluation script passed raw dicts directly

Impact:

- follow-up test cases could crash in guardrail processing

Fix:

- added chat history normalization in `run_ragas_eval.py`
- existing deterministic evaluation code already had explicit message conversion logic

### 2. Ragas Was Evaluating Empty Evidence

Problem:

- the evaluator built context strings from `result["sources"]`
- `sources` originally contained only metadata such as doc name, URL, and page
- retrieved chunk text was missing

Impact:

- semantic metrics like faithfulness and context precision/recall were not evaluating the real evidence

Fix:

- updated `rag_chain.py` so each source includes `content: doc.page_content`
- updated `run_ragas_eval.py` to pass retrieved chunk text into the Ragas dataset

### 3. Colab Packaging and Dependency Issues

Problem:

- the Colab guide referenced wrong directories
- required dependencies were missing
- notebook execution also collided with Jupyter kernel arguments

Impact:

- Colab runs failed before evaluation could complete

Fix:

- corrected the Colab guide
- switched CLI parsing to `parse_known_args()`
- documented the real project data directories

### 4. Ragas API Compatibility Drift

Problem:

- the installed `ragas` version did not behave like the earlier code expected
- metrics had to be initialized differently
- results came back as `EvaluationResult` objects rather than dicts

Impact:

- Ragas crashed or report saving failed

Fix:

- updated metric construction to use explicit metric classes
- added `extract_ragas_scores()` to normalize result objects before saving and printing

### 5. Gemini Judge Model Name Mismatch

Problem:

- the Ragas evaluator used `gemini-1.5-flash`
- the active Gemini endpoint rejected that model name

Impact:

- semantic evaluation jobs failed with `404 NOT_FOUND`

Fix:

- changed the default judge model to `gemini-flash-latest`
- allowed override through `RAGAS_JUDGE_MODEL`

### 6. Quota Exhaustion

Problem:

- both the app and Ragas use Gemini calls
- running everything together on the free tier quickly exhausted the available request quota

Impact:

- later runs failed with `429 RESOURCE_EXHAUSTED`

Fix:

- no code bug here; this is an external quota constraint
- recommended workflow is to run deterministic evaluation and semantic evaluation separately when on the free tier

---

## 6. Current Evaluation Architecture

The current evaluation architecture is:

### Step 1: Deterministic Evaluation

`run_app_eval.py` loads test cases and checks whether the application behaves correctly from an engineering and safety perspective.

This is the preferred everyday regression tool.

### Step 2: Guardrail Evaluation

The same deterministic runner is used with `guardrail_eval_cases.json` to verify safety and refusal behaviors.

### Step 3: Semantic RAG Evaluation

`run_ragas_eval.py` reuses the actual app output and measures semantic quality with Ragas.

This is the preferred tool for judging retrieval grounding and answer relevance, but it depends on external LLM quota and package compatibility.

### Step 4: Combined View

`run_all_evals.py` provides one entrypoint that summarizes both layers.

---

## 7. Current Known Results

At the end of the debugging and evaluation work:

- guardrail regression cases passed
- manual deterministic evaluation passed after prompt tightening
- the combined runner correctly reports deterministic and Ragas phases separately
- the remaining operational limitation is Gemini free-tier quota during semantic evaluation

This means the application-level validation is functioning, and the semantic evaluation path is structurally in place, but external quota limits can still interrupt full Ragas runs.

---

## 8. How the Deterministic Evaluation Works Internally

For each case in the dataset:

1. load the question and chat history
2. convert JSON history into LangChain message objects
3. call `ask_health_question(question, chat_history)`
4. capture:
   - answer
   - sources
   - status
   - latency
5. compare actual output against expectations in the dataset
6. store the case result and aggregate summary counts

This lets the project verify not just correctness, but conformance to expected application behavior.

---

## 9. How the Ragas Evaluation Works Internally

For each case in the dataset:

1. load the question and ground truth
2. normalize chat history
3. call `ask_health_question()`
4. extract:
   - question
   - answer
   - retrieved chunk text
   - ground truth
5. build a `Dataset`
6. pass that dataset into `ragas.evaluate()`
7. use Gemini as the judge LLM
8. use local embeddings for vector similarity work
9. aggregate semantic scores and save them

This makes Ragas evaluate the real app pipeline, not a mocked or synthetic version.

---

## 10. Recommended Usage Going Forward

### For Daily Development

Use:

```bash
python evals/run_app_eval.py --dataset evals/manual_eval_cases.json
python evals/run_app_eval.py --dataset evals/guardrail_eval_cases.json
```

Reason:

- fast
- deterministic
- stable
- low dependency risk

### For Semantic Quality Review

Use:

```bash
python evals/run_ragas_eval.py --dataset evals/manual_eval_cases.json
```

Reason:

- evaluates faithfulness and retrieval quality
- better for deeper RAG quality analysis
- useful for demos, reporting, and model comparison

### For One Combined Report

Use:

```bash
python evals/run_all_evals.py --dataset evals/manual_eval_cases.json
```

Reason:

- gives a single summary
- captures deterministic status even if Ragas fails

---

## 11. Recommended Environment

The preferred execution path is local, not Colab.

Recommended environment:

```bash
py -3.11 -m venv .venv-clean
.venv-clean\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Why:

- better control of package versions
- easier debugging
- avoids notebook argument issues
- avoids stale uploaded files
- better suited for repeated engineering evaluation

---

## 12. Limitations

Current limitations include:

- Ragas depends on external API quota
- Gemini free-tier rate limits can interrupt full semantic runs
- deprecation warnings remain in the current LangChain embedding and Chroma setup
- Chroma telemetry warnings are noisy, though not blocking

These do not invalidate the deterministic evaluation path, but they affect convenience and reliability of the semantic path.

---

## 13. Summary

The final evaluation system now has two complementary layers:

1. a deterministic, stable technical validation layer
2. a semantic, LLM-judged RAG quality layer

Together they cover:

- correctness of application behavior
- safety and guardrails
- source retrieval presence
- retrieved chunk availability
- conversational follow-up handling
- semantic grounding quality
- question relevance
- retrieval completeness

This gives the project a practical and defensible RAG evaluation process rather than relying on a single fragile notebook or a single metric source.
