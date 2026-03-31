# Evaluation Harness

This folder contains repo-local evaluation assets for the Uganda mental well-being chatbot.

## Do You Need Google Colab?

No. The preferred path is local execution against the real application code in [rag_chain.py](/C:/Users/user/Desktop/health_chatbot/rag_chain.py).

Use Colab only when:
- your local Python environment is unavailable or broken
- you want to experiment interactively with notebook-based RAGAS workflows
- you need temporary hosted compute for larger evaluation batches

For repeatable system evaluation, local scripts are the better default.

## What This Covers

### 1. Logic & Guardrails (`run_app_eval.py`)
Runs fast, local checks against the real `ask_health_question()` pipeline.
- Guardrail regression (SAFE, CRISIS, UNSAFE, OUT_OF_SCOPE)
- Basic answer validation (expected status, source presence, phrase matching)
- Retrieved chunk validation (`sources[].content` must be present when sources are required)
- Conversational follow-up stability using serialized chat history from JSON fixtures

### 2. RAG Quality (`run_ragas_eval.py`)
Uses the [Ragas](https://docs.ragas.io/) framework with Gemini as a judge.
- **Faithfulness**: detectable hallucinations (is the answer grounded in context?)
- **Answer Relevancy**: how well the answer addresses the specific question.
- **Context Precision**: whether the retrieved documents are relevant.
- **Context Recall**: whether the necessary info was present in the context.

## Files

- `run_app_eval.py`: primary guardrail/logic check
- `run_ragas_eval.py`: LLM-based RAG quality evaluation
- `run_all_evals.py`: convenience runner for both deterministic checks and Ragas
- `manual_eval_cases.json`: core cases (retrieval + answers + ground truth)
- `guardrail_eval_cases.json`: refusal and crisis behavior cases
- `results/`: output folder for run artifacts

## How to Run

### Create a Clean Local Environment
`ragas` is more reliable locally than in Colab. Prefer Python `3.11`.

```bash
py -3.11 -m venv .venv-clean
.venv-clean\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Run Local Checks
```bash
python evals/run_app_eval.py --dataset evals/manual_eval_cases.json
```

### Run Ragas Quality Evaluation
```bash
python evals/run_ragas_eval.py --dataset evals/manual_eval_cases.json
```

### Run Both in One Command
```bash
python evals/run_all_evals.py --dataset evals/manual_eval_cases.json
```

### Run Guardrail Regression Cases
```bash
python evals/run_app_eval.py --dataset evals/guardrail_eval_cases.json
```

## How To Read Results

Each run writes a JSON report into `evals/results/`. 
- Logic checks return a `passed: true/false`.
- Logic checks also report whether retrieved chunk text was actually returned to the evaluator.
- Ragas returns scores between `0.0` and `1.0`. Aim for `>0.8` on Faithfulness and Relevancy.
- `run_all_evals.py` always reports deterministic results first and includes any Ragas failure details instead of hiding them.
