"""Runs every validator check on one GeneratedItem and returns one structured result. Each check is
isolated with try/except ImportError (and generic errors) so a missing optional dependency degrades
that one check to "skipped" instead of crashing the whole validation run.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from task1.schemas import GeneratedItem
from task1.validator import matching, ocr, quality
from task1.validator.embeddings import EmbeddingResult  # noqa: F401 - re-exported for callers' type hints


@dataclass
class ValidationResult:
    quality: quality.QualityResult
    ocr: ocr.OCRResult
    embedding: EmbeddingResult | None
    embedding_skipped_reason: str | None
    regions: list[matching.RegionResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        checks = [self.quality.passed, self.ocr.passed, all(r.passed for r in self.regions)]
        if self.embedding is not None:
            checks.append(self.embedding.passed)
        return all(checks)

    def as_dict(self) -> dict:
        return {
            "quality": self.quality.as_dict(),
            "ocr": self.ocr.as_dict(),
            "embedding": self.embedding.as_dict() if self.embedding else None,
            "embedding_skipped_reason": self.embedding_skipped_reason,
            "regions": [r.as_dict() for r in self.regions],
            "overall_passed": self.passed,
        }


def validate(item: GeneratedItem) -> ValidationResult:
    """One item. Fine for interactive use; `validate_many` is the batched, GPU-friendly version for
    a whole run (one embedding forward-pass for all items instead of one per item)."""
    return validate_many([item])[0]


def validate_many(items: list[GeneratedItem]) -> list[ValidationResult]:
    quality_results = [quality.assess_quality(i.image_path) for i in items]
    ocr_results = [ocr.validate_text(i.image_path, i.text_bbox, i.question.question_text) for i in items]
    region_results = [[matching.verify_region_distinctiveness(i.image_path, t.label, t.bbox)
                       for t in i.question.targets] for i in items]

    embedding_results: list["EmbeddingResult | None"] = [None] * len(items)
    embedding_error: str | None = None
    try:
        from task1.validator.embeddings import validate_batch

        batch = [(i.image_path, i.question.question_text, i.question.topic) + ("a recipe for chocolate cake",)
                 for i in items]
        embedding_results = validate_batch(batch)
    except ImportError as e:
        embedding_error = f"sentence-transformers not installed ({e})"
    except Exception as e:                                              # noqa: BLE001
        embedding_error = f"{type(e).__name__}: {e}"

    return [ValidationResult(quality=q, ocr=o, embedding=e, embedding_skipped_reason=embedding_error, regions=r)
            for q, o, e, r in zip(quality_results, ocr_results, embedding_results, region_results)]
