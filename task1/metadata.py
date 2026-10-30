"""4. Integration Metadata (Click-and-Answer Ready) — the exact per-target contract requested, built
straight from the real pixel bboxes the generator measured (task1.generator.diagrams.bbox_of), not
estimated after the fact. This is what a click-and-answer / segmentation task downstream reads to
know which target a click landed on.
"""
from __future__ import annotations

import json
from pathlib import Path

from task1.schemas import GeneratedItem


def build_click_answer_metadata(item: GeneratedItem) -> list[dict]:
    """[{"target_name", "target_description", "target_category", "expected_visual_region"}, ...]"""
    return [target.to_metadata() for target in item.question.targets]


def save_metadata(metadata: list[dict], out_path: Path | str) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path
