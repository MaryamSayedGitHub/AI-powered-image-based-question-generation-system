"""Smoke tests — no LLM, no network needed. easyocr / sentence-transformers are optional: the
embedding test is skipped (not failed) when sentence-transformers isn't installed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from task1.generator.bank import get_question, load_question_bank
from task1.generator.renderer import render_question_image
from task1.metadata import build_click_answer_metadata
from task1.pipeline import run_question
from task1.validator.pipeline import validate


def test_question_bank_loads():
    questions = load_question_bank()
    assert len(questions) >= 3
    assert {q.type for q in questions} == {"diagram_labeled", "shape_question", "chart_question"}


@pytest.mark.parametrize("question_id", ["bio_cell_001", "geo_shapes_001", "chart_bar_001"])
def test_generate_and_validate_each_type(tmp_path, question_id):
    question = get_question(question_id)
    item = render_question_image(question, tmp_path / f"{question_id}.png")

    assert Path(item.image_path).exists()
    assert all(t.bbox.w > 0 and t.bbox.h > 0 for t in question.targets), "every target must have a real bbox"

    result = validate(item)
    assert result.quality.passed, result.quality.as_dict()
    assert result.ocr.passed, result.ocr.as_dict()
    assert all(r.passed for r in result.regions), [r.as_dict() for r in result.regions]


def test_metadata_matches_the_required_contract(tmp_path):
    question = get_question("bio_cell_001")
    item = render_question_image(question, tmp_path / "_task1_meta_test.png")
    metadata = build_click_answer_metadata(item)
    for entry in metadata:
        assert set(entry) == {"target_name", "target_description", "target_category", "expected_visual_region"}
        assert set(entry["expected_visual_region"]) == {"x", "y", "w", "h"}


def test_backend_registry_has_all_three_and_programmatic_is_default():
    from task1.config import SETTINGS
    from task1.generator.backends import BACKENDS, get_backend

    assert set(BACKENDS) == {"programmatic", "diffusion", "gan"}
    assert SETTINGS.generation_backend == "programmatic"
    assert get_backend() is BACKENDS["programmatic"]


def test_run_question_end_to_end(tmp_path):
    result = run_question("chart_bar_001", tmp_path)
    assert result["passed"] is True
    assert Path(result["report_path"]).exists()
    assert Path(result["metadata_path"]).exists()
