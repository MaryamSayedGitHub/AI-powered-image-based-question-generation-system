"""Top-level orchestrator: generate -> validate -> report -> integration metadata, for one question
or for the whole bank. This is the one function other code (a CLI, a notebook, the real app) should
call; everything else in this package is a building block it composes.
"""
from __future__ import annotations

from pathlib import Path

from task1.config import SETTINGS
from task1.generator.bank import get_question, load_question_bank
from task1.generator.renderer import render_question_image
from task1.metadata import build_click_answer_metadata, save_metadata
from task1.reporter.report import build_report, save_report
from task1.schemas import dataclass_to_jsonable
from task1.validator.pipeline import validate, validate_many


def _finish(item, result, out_dir: Path) -> dict:
    report = build_report(item, result)
    report_path = save_report(report, out_dir / f"{item.question.id}.report.json")
    metadata = build_click_answer_metadata(item)
    metadata_path = save_metadata(metadata, out_dir / f"{item.question.id}.metadata.json")
    return {"question_id": item.question.id, "image_path": item.image_path, "report_path": str(report_path),
            "metadata_path": str(metadata_path), "report": report, "metadata": metadata,
            "passed": result.passed, "targets": dataclass_to_jsonable(item.question.targets)}


def run_question(question_id: str, output_dir: Path | str | None = None) -> dict:
    """Generates, validates and reports on one question bank entry. Returns the combined artifact:
    {question_id, image_path, report_path, metadata_path, report, metadata, passed}."""
    out_dir = Path(output_dir) if output_dir else SETTINGS.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    question = get_question(question_id)
    item = render_question_image(question, out_dir / f"{question.id}.png")
    result = validate(item)
    return _finish(item, result, out_dir)


def run_all(output_dir: Path | str | None = None) -> list[dict]:
    """Generates every question-bank item, then validates them in ONE batch (embeddings.validate_batch):
    a single GPU forward-pass for all images/texts instead of one per question — the way to actually
    use a GPU here, since Task 1's own bank is small but a real deployment's question bank isn't."""
    out_dir = Path(output_dir) if output_dir else SETTINGS.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    questions = load_question_bank()
    items = [render_question_image(q, out_dir / f"{q.id}.png") for q in questions]
    results = validate_many(items)
    return [_finish(item, result, out_dir) for item, result in zip(items, results)]
