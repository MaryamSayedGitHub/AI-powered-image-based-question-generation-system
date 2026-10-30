"""3. Automated Quality Report — one structured dict per generated item, saved as JSON."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from task1.schemas import GeneratedItem
from task1.validator.pipeline import ValidationResult


def build_report(item: GeneratedItem, result: ValidationResult) -> dict:
    return {
        "question_id": item.question.id,
        "question_type": item.question.type,
        "topic": item.question.topic,
        "image_path": item.image_path,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "image": {"resolution": {"width": item.image_size[0], "height": item.image_size[1]},
                  "blur_score": result.quality.blur_variance, "contrast_score": result.quality.contrast_std},
        "ocr": {"engine": result.ocr.engine, "extracted_text": result.ocr.read_text,
                "confidence": round(result.ocr.confidence, 3), "similarity_to_expected": round(result.ocr.similarity, 3)},
        "embedding": result.embedding.as_dict() if result.embedding else {"skipped": result.embedding_skipped_reason},
        "regions": [r.as_dict() for r in result.regions],
        "status": "pass" if result.passed else "fail",
        "checks": {"quality_passed": result.quality.passed, "ocr_passed": result.ocr.passed,
                   "embedding_passed": result.embedding.passed if result.embedding else None,
                   "regions_passed": all(r.passed for r in result.regions)},
    }


def save_report(report: dict, out_path: Path | str) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path
