"""Plain dataclasses shared by every module — one definition of "what a target/question/report
looks like", so generator, validator and reporter can't drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class BBox:
    """Pixel box in the FINAL rendered image (after the question banner is added)."""
    x: int
    y: int
    w: int
    h: int

    def as_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    def crop(self, image):
        return image.crop((self.x, self.y, self.x + self.w, self.y + self.h))


@dataclass
class Target:
    """One clickable/answerable part of the diagram — doubles as the Task-6 integration record."""
    label: str                     # "A", "B", "C" — what's drawn on the diagram itself
    name: str                      # "Nucleus" — the real identity of that part
    description: str               # "The organelle that controls the cell and stores its DNA"
    category: str                  # "organelle", "organ", "shape", ...
    bbox: BBox

    def to_metadata(self) -> dict[str, Any]:
        """Matches the "Integration Metadata (Click-and-Answer Ready)" contract exactly."""
        return {
            "target_name": self.name,
            "target_description": self.description,
            "target_category": self.category,
            "expected_visual_region": self.bbox.as_dict(),
        }


@dataclass
class Question:
    """One entry of the local question bank."""
    id: str
    type: str                      # "diagram_labeled" | "shape_question" | "chart_question"
    topic: str                     # e.g. "biology.cell" — used by the embedding check
    question_text: str
    targets: list[Target]
    correct_label: str
    rubric: list[str] = field(default_factory=list)
    background_prompt: str | None = None   # only used by the diffusion/GAN backends (generator/backends.py)

    def target_by_label(self, label: str) -> Target | None:
        return next((t for t in self.targets if t.label == label), None)

    def correct_target(self) -> Target | None:
        return self.target_by_label(self.correct_label)


@dataclass
class GeneratedItem:
    """What `generator` hands to `validator`: the finished image plus everything needed to check it."""
    question: Question
    image_path: str
    image_size: tuple[int, int]
    text_bbox: BBox                # where the question banner text was drawn, for OCR cropping


def dataclass_to_jsonable(obj: Any) -> Any:
    """asdict() but tolerant of nested dataclasses/tuples the json module can't serialize directly."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: dataclass_to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [dataclass_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: dataclass_to_jsonable(v) for k, v in obj.items()}
    return obj
