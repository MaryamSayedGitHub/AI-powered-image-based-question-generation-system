"""Loads the local JSON question bank into the shared dataclasses (schemas.py)."""
from __future__ import annotations

import json
from pathlib import Path

from task1.config import SETTINGS
from task1.schemas import BBox, Question, Target


def load_question_bank(path: Path | str | None = None) -> list[Question]:
    path = Path(path) if path else SETTINGS.question_bank_path
    raw = json.loads(path.read_text(encoding="utf-8"))
    questions = []
    for row in raw:
        targets = [Target(label=t["label"], name=t["name"], description=t["description"],
                          category=t["category"], bbox=BBox(0, 0, 0, 0))   # filled in by the renderer
                   for t in row["targets"]]
        questions.append(Question(id=row["id"], type=row["type"], topic=row["topic"],
                                  question_text=row["question_text"], targets=targets,
                                  correct_label=row["correct_label"], rubric=row.get("rubric", []),
                                  background_prompt=row.get("background_prompt")))
    return questions


def get_question(question_id: str, path: Path | str | None = None) -> Question:
    for q in load_question_bank(path):
        if q.id == question_id:
            return q
    raise KeyError(f"no question '{question_id}' in the question bank")
