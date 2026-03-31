# RAG Evaluation Guide (Ragas + Colab)

This guide explains how to evaluate the health chatbot's Retrieval-Augmented Generation (RAG) system using `ragas` in Google Colab.

---

## 1. Understanding the Metrics

Ragas uses an LLM judge, Gemini 1.5 Flash in this project, to score the RAG system across four dimensions:

### Faithfulness
- **What it measures**: Whether the answer is grounded in the retrieved context.
- **Goal**: Reduce hallucinations.
- **Score**: `0` to `1` where higher is better.

### Answer Relevancy
- **What it measures**: Whether the generated answer actually addresses the user's question.
- **Goal**: Reduce off-topic or vague responses.
- **Score**: `0` to `1` where higher is better.

### Context Precision
- **What it measures**: Whether the most useful retrieved chunks are ranked highly.
- **Goal**: Put the best evidence in front of the model.
- **Score**: `0` to `1` where higher is better.

### Context Recall
- **What it measures**: Whether the retrieved context contains enough information to support the ground-truth answer.
- **Goal**: Avoid missing key evidence during retrieval.
- **Required**: Needs a `ground_truth` field in the dataset.
- **Score**: `0` to `1` where higher is better.

---

## 2. Preparing Your Data

Ragas expects each test case to include:

| Field | Description |
| :--- | :--- |
| `question` | The user query. |
| `contexts` | A list of retrieved text chunks. |
| `answer` | The chatbot's generated answer. |
| `ground_truth` | A human-verified reference answer. |

### Example JSON Format (`evals/manual_eval_cases.json`)
```json
[
  {
    "question": "What are the common symptoms of the flu?",
    "ground_truth": "Common flu symptoms include fever, cough, sore throat, muscle aches, and fatigue."
  }
]
```

---

## 3. Google Colab Execution Steps

### Step A: Zip and Upload
1. In Windows Explorer, select:
   - `evals/`
   - `documents/`
   - `chroma_db_mental_v2/`
   - `rag_chain.py`
   - `vector_store.py`
   - `requirements.txt`
2. Create a zip archive named `project.zip`.
3. Open [colab.research.google.com](https://colab.research.google.com) and create a new notebook.
4. Open the Files panel in the left sidebar.
5. Upload `project.zip`.

`documents/` contains the source corpus and `chroma_db_mental_v2/` contains the persisted Chroma index used by `get_vector_store()`. If either one is missing, the notebook will not reproduce the local RAG pipeline correctly.

### Step B: Setup Environment (Cell 1)
Paste this into the first Colab cell:

```python
# 1. Unzip the uploaded project
!unzip -q project.zip

# 2. Install the dependencies needed by the evaluation script and app imports
!pip install -q ragas datasets langchain-google-genai langchain-community langchain-core langchain-classic langchain-groq sentence-transformers pypdf faiss-cpu chromadb python-dotenv

# 3. Set API keys
import os
from getpass import getpass

os.environ["GOOGLE_API_KEY"] = getpass("Enter your Gemini API Key: ")
os.environ["GROQ_API_KEY"] = getpass("Enter your Groq API Key (needed for guardrails): ")
```

If Colab says the runtime needs to restart after installation, restart it before running the next cell.

### Step C: Execute Evaluation (Cell 2)
Paste this into the second Colab cell:

```python
import os
import sys
from pathlib import Path

# Ensure the project root is importable
sys.path.append(os.getcwd())

from evals.run_ragas_eval import main

# Option 1: Run with the default dataset
main()

# Option 2: Manual control
# from evals.run_ragas_eval import load_cases, run_predictions, run_ragas_evaluation
# cases = load_cases(Path("evals/manual_eval_cases.json"))
# eval_dataset = run_predictions(cases)
# results = run_ragas_evaluation(eval_dataset)
# print(results)
```

---

## 4. What `run_ragas_eval.py` Does

The script performs three steps:

1. **Inference**: Calls `ask_health_question()` for each evaluation case and captures the generated answer plus the retrieved chunk text.
2. **Judgment**: Sends the question, retrieved contexts, answer, and ground truth into Ragas for scoring.
3. **Reporting**: Saves a JSON report in `evals/results/`.

After running `main()`, a rough interpretation of scores is:
- `0.0 - 0.4`: Poor performance.
- `0.5 - 0.7`: Usable, but retrieval or prompting likely needs work.
- `0.8 - 1.0`: Strong grounding and relevance.

Use the report in `evals/results/` to inspect which cases scored poorly and why.
