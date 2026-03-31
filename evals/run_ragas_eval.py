import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage

try:
    from ragas.metrics import (
        Faithfulness,
        ResponseRelevancy,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )
except ImportError:
    from ragas.metrics import Faithfulness, ResponseRelevancy
    from ragas.metrics.collections import ContextPrecision as LLMContextPrecisionWithReference
    from ragas.metrics.collections import ContextRecall as LLMContextRecall

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "evals" / "results"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Local imports
from rag_chain import ask_health_question
from vector_store import get_embeddings

def load_cases(dataset_path: Path) -> List[Dict[str, Any]]:
    with dataset_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_ragas_scores(result: Any) -> Dict[str, float]:
    """
    Normalizes Ragas outputs across versions into a plain metric->score dict.
    """
    if hasattr(result, "items"):
        return {k: float(v) for k, v in result.items()}

    if hasattr(result, "to_pandas"):
        df = result.to_pandas()
        score_columns = [
            column for column in df.columns
            if column not in {"question", "answer", "contexts", "ground_truth"}
        ]
        numeric_scores: Dict[str, float] = {}
        for column in score_columns:
            series = df[column]
            if getattr(series, "dtype", None) is not None and str(series.dtype) != "object":
                numeric_scores[column] = float(series.mean())
        if numeric_scores:
            return numeric_scores

    if hasattr(result, "__dict__"):
        return {
            key: float(value)
            for key, value in vars(result).items()
            if isinstance(value, (int, float))
        }

    raise TypeError(f"Unsupported Ragas result type: {type(result)!r}")

def normalize_chat_history(chat_history: List[Dict[str, Any]]) -> List[Any]:
    """
    Converts JSON chat history into LangChain message objects expected by the app.
    """
    normalized_history = []
    for message in chat_history:
        if hasattr(message, "content"):
            normalized_history.append(message)
            continue

        role = str(message.get("role", "")).lower()
        content = message.get("content", "")

        if role == "user":
            normalized_history.append(HumanMessage(content=content))
        elif role == "assistant":
            normalized_history.append(AIMessage(content=content))

    return normalized_history

def run_predictions(cases: List[Dict[str, Any]]) -> Dataset:
    """
    Runs the RAG system for each case and prepares a HuggingFace Dataset for Ragas.
    """
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print(f"Running predictions for {len(cases)} cases...")

    for i, case in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] Question: {case['question'][:50]}...")

        # Execute the RAG chain
        chat_history = normalize_chat_history(case.get("chat_history", []))
        result = ask_health_question(case["question"], chat_history)

        questions.append(case["question"])
        answers.append(result.get("answer", ""))

        # Ragas expects context as a list of strings (each string is a chunk)
        # We pass the retrieved chunk text rather than only source metadata.
        case_contexts = [
            s.get("content", "").strip()
            for s in result.get("sources", [])
            if s.get("content", "").strip()
        ]
        contexts.append(case_contexts)
        ground_truths.append(case.get("ground_truth", ""))

    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    
    return Dataset.from_dict(data)

def run_ragas_evaluation(dataset: Dataset):
    """
    Performs the Ragas evaluation using Gemini as the judge and local embeddings.
    """
    # Configure the judge LLM
    judge_llm = ChatGoogleGenerativeAI(
        model=os.getenv("RAGAS_JUDGE_MODEL", "gemini-flash-latest"),
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # Configure local embeddings for Ragas
    embeddings_model = get_embeddings()
    
    # Wrap for Ragas
    ragas_llm = LangchainLLMWrapper(judge_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings_model)
    
    # Metrics to evaluate
    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]

    print("Starting Ragas evaluation (this may take a few minutes)...")
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )
    
    return result

def save_report(dataset_path: Path, result: Any, raw_data: Dataset) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = RESULTS_DIR / f"ragas_{dataset_path.stem}_{timestamp}.json"
    scores = extract_ragas_scores(result)
    
    report = {
        "dataset_source": str(dataset_path),
        "generated_at_utc": timestamp,
        "judge_model": os.getenv("RAGAS_JUDGE_MODEL", "gemini-flash-latest"),
        "scores": scores,
        "cases": raw_data.to_dict()
    }
    
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    
    return output_path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ragas evaluations against ask_health_question().")
    parser.add_argument(
        "--dataset",
        default="evals/manual_eval_cases.json",
        help="Path to a JSON evaluation dataset with ground_truth.",
    )
    args, _unknown = parser.parse_known_args()
    return args

def main():
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = (ROOT / dataset_path).resolve()
        
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return

    # 1. Load cases
    cases = load_cases(dataset_path)
    
    # 2. Run inference to get model answers
    eval_dataset = run_predictions(cases)
    
    # 3. Run Ragas evaluation
    try:
        results = run_ragas_evaluation(eval_dataset)
        scores = extract_ragas_scores(results)
        
        # 4. Print summary
        print("\n=== Ragas Evaluation Results ===")
        for metric, score in scores.items():
            print(f"{metric}: {score:.4f}")
        
        # 5. Save report
        report_path = save_report(dataset_path, results, eval_dataset)
        print(f"\nSaved Ragas report to: {report_path}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
