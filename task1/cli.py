"""Command-line entry point.

    python -m task1.cli                        # run every question in the bank
    python -m task1.cli --question-id bio_cell_001
    python -m task1.cli --out ./my_output
"""
from __future__ import annotations

import argparse
import json
import sys

from task1.pipeline import run_all, run_question


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Task 1: generate + validate image-based questions.")
    parser.add_argument("--question-id", help="run a single question bank entry (default: run all)")
    parser.add_argument("--out", help="output directory (default: task1/../output)")
    args = parser.parse_args(argv)

    results = [run_question(args.question_id, args.out)] if args.question_id else run_all(args.out)

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {r['question_id']:16} image={r['image_path']}  report={r['report_path']}")
    print(json.dumps({"total": len(results), "passed": sum(r["passed"] for r in results)}, indent=2))
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
