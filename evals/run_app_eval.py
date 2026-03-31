import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "evals" / "results"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_chain import ask_health_question


def load_cases(dataset_path: Path) -> List[Dict[str, Any]]:
    with dataset_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_chat_history(raw_history: List[Dict[str, str]]) -> List[Any]:
    history = []
    for item in raw_history:
        role = item.get("role")
        content = item.get("content", "")
        if role == "user":
            history.append(HumanMessage(content=content))
        elif role == "assistant":
            history.append(AIMessage(content=content))
    return history


def normalize_text(value: str) -> str:
    return (value or "").strip().lower()


def check_source_match(expected_fragment: str, sources: List[Dict[str, Any]]) -> bool:
    if not expected_fragment:
        return True
    expected = normalize_text(expected_fragment)
    for source in sources:
        doc_name = normalize_text(source.get("doc_name", ""))
        url = normalize_text(source.get("url", ""))
        if expected in doc_name or expected in url:
            return True
    return False


def check_non_empty_source_content(sources: List[Dict[str, Any]]) -> bool:
    if not sources:
        return False
    return all((source.get("content", "") or "").strip() for source in sources)


def check_required_phrases(answer: str, phrases: List[str]) -> bool:
    answer_normalized = normalize_text(answer)
    return all(normalize_text(phrase) in answer_normalized for phrase in phrases)


def check_forbidden_phrases(answer: str, phrases: List[str]) -> bool:
    answer_normalized = normalize_text(answer)
    return all(normalize_text(phrase) not in answer_normalized for phrase in phrases)


def evaluate_case(case: Dict[str, Any]) -> Dict[str, Any]:
    question = case["question"]
    chat_history = build_chat_history(case.get("chat_history", []))

    start = time.perf_counter()
    result = ask_health_question(question, chat_history)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    answer = result.get("answer", "")
    sources = result.get("sources", [])
    status = result.get("status", "UNKNOWN")

    checks = {
        "status_match": status == case.get("expected_status"),
        "source_match": check_source_match(case.get("expected_source_contains", ""), sources),
        "must_include_match": check_required_phrases(answer, case.get("must_include", [])),
        "must_not_include_match": check_forbidden_phrases(answer, case.get("must_not_include", [])),
        "sources_present_match": (len(sources) > 0) == bool(case.get("require_sources", False)),
        "source_content_present_match": check_non_empty_source_content(sources) if case.get("require_sources", False) else True,
    }

    return {
        "id": case.get("id"),
        "question": question,
        "expected_status": case.get("expected_status"),
        "actual_status": status,
        "latency_ms": elapsed_ms,
        "answer": answer,
        "sources": sources,
        "checks": checks,
        "passed": all(checks.values()),
    }


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    passed = sum(1 for item in results if item["passed"])
    failed = total - passed

    check_totals: Dict[str, int] = {}
    for item in results:
        for name, ok in item["checks"].items():
            check_totals.setdefault(name, 0)
            if ok:
                check_totals[name] += 1

    average_latency = round(
        sum(item["latency_ms"] for item in results) / total, 2
    ) if total else 0.0

    return {
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": failed,
        "pass_rate": round((passed / total), 4) if total else 0.0,
        "average_latency_ms": average_latency,
        "check_pass_counts": check_totals,
    }


def print_summary(summary: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
    print("\n=== Evaluation Summary ===")
    print(json.dumps(summary, indent=2))

    failures = [item for item in results if not item["passed"]]
    if failures:
        print("\n=== Failed Cases ===")
        for item in failures:
            print(f"- {item['id']}: expected {item['expected_status']}, got {item['actual_status']}")
            for check_name, ok in item["checks"].items():
                if not ok:
                    print(f"  - failed check: {check_name}")


def save_report(dataset_path: Path, summary: Dict[str, Any], results: List[Dict[str, Any]]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = RESULTS_DIR / f"{dataset_path.stem}_{timestamp}.json"
    payload = {
        "dataset": str(dataset_path),
        "generated_at_utc": timestamp,
        "summary": summary,
        "results": results,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local evaluations against ask_health_question().")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to a JSON evaluation dataset.",
    )
    args, _unknown = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    dataset_path = (ROOT / args.dataset).resolve() if not Path(args.dataset).is_absolute() else Path(args.dataset)
    cases = load_cases(dataset_path)
    results = [evaluate_case(case) for case in cases]
    summary = summarize(results)
    report_path = save_report(dataset_path, summary, results)
    print_summary(summary, results)
    print(f"\nSaved report to: {report_path}")


if __name__ == "__main__":
    main()
