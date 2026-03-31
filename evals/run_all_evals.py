import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
EVALS_DIR = ROOT / "evals"
for path in (ROOT, EVALS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_app_eval import (
    evaluate_case as evaluate_app_case,
    load_cases as load_app_cases,
    save_report as save_app_report,
    summarize as summarize_app_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic app checks and Ragas quality evaluation.")
    parser.add_argument(
        "--dataset",
        default="evals/manual_eval_cases.json",
        help="Path to a JSON evaluation dataset with ground_truth.",
    )
    args, _unknown = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = (ROOT / dataset_path).resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print("Running deterministic application checks...")
    app_cases = load_app_cases(dataset_path)
    app_results = [evaluate_app_case(case) for case in app_cases]
    app_summary = summarize_app_results(app_results)
    app_report_path = save_app_report(dataset_path, app_summary, app_results)

    combined_report: Dict[str, Any] = {
        "dataset": str(dataset_path),
        "app_eval": {
            "summary": app_summary,
            "report_path": str(app_report_path),
        }
    }

    print("\nRunning Ragas quality evaluation...")
    try:
        from run_ragas_eval import (
            extract_ragas_scores,
            load_cases as load_ragas_cases,
            run_predictions,
            run_ragas_evaluation,
            save_report as save_ragas_report,
        )

        ragas_cases = load_ragas_cases(dataset_path)
        ragas_dataset = run_predictions(ragas_cases)
        ragas_results = run_ragas_evaluation(ragas_dataset)
        ragas_report_path = save_ragas_report(dataset_path, ragas_results, ragas_dataset)
        combined_report["ragas_eval"] = {
            "scores": extract_ragas_scores(ragas_results),
            "report_path": str(ragas_report_path),
        }
    except Exception as exc:
        combined_report["ragas_eval"] = {
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

    print("\n=== Combined Evaluation Summary ===")
    print(json.dumps(combined_report, indent=2))


if __name__ == "__main__":
    main()
