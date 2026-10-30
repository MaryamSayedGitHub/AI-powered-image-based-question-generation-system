"""A. OCR Validation — EasyOCR primary (better on natural/complex layouts, GPU-optional, CPU is
fine for a single banner crop), pytesseract as a zero-extra-dependency fallback so the pipeline
still runs on a machine that only has the system `tesseract` binary.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from task1.config import SETTINGS
from task1.schemas import BBox


@dataclass
class OCRResult:
    engine: str
    read_text: str
    expected_text: str
    confidence: float         # 0-1
    similarity: float         # 0-1, difflib ratio(read, expected)
    passed: bool

    def as_dict(self) -> dict:
        return {"engine": self.engine, "read_text": self.read_text, "expected_text": self.expected_text,
                "confidence": round(self.confidence, 3), "similarity": round(self.similarity, 3),
                "passed": self.passed}


_easyocr_reader = None


def _get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr   # noqa: PLC0415 - optional, imported lazily so the package is importable without it

        from task1.device import resolve_device

        _easyocr_reader = easyocr.Reader(list(SETTINGS.easyocr_langs), gpu=resolve_device() == "cuda")
    return _easyocr_reader


def _read_with_easyocr(crop: Image.Image) -> tuple[str, float]:
    import numpy as np

    reader = _get_easyocr_reader()
    results = reader.readtext(np.array(crop))
    if not results:
        return "", 0.0
    text = " ".join(r[1] for r in results)
    confidence = sum(r[2] for r in results) / len(results)
    return text.strip(), float(confidence)


def _read_with_tesseract(crop: Image.Image) -> tuple[str, float]:
    import pytesseract

    data = pytesseract.image_to_data(crop, output_type=pytesseract.Output.DICT)
    words = [(w, int(c)) for w, c in zip(data["text"], data["conf"]) if w.strip() and int(c) >= 0]
    text = " ".join(w for w, _ in words).strip()
    confidence = (sum(c for _, c in words) / len(words) / 100.0) if words else 0.0
    return text, confidence


def validate_text(image_path: Path | str, text_bbox: BBox, expected_text: str) -> OCRResult:
    th = SETTINGS.thresholds
    crop = Image.open(image_path).crop((text_bbox.x, text_bbox.y, text_bbox.x + text_bbox.w,
                                        text_bbox.y + text_bbox.h))
    try:
        engine = "easyocr"
        text, confidence = _read_with_easyocr(crop)
    except ImportError:
        engine = "pytesseract"
        text, confidence = _read_with_tesseract(crop)

    similarity = difflib.SequenceMatcher(None, text.lower(), expected_text.lower()).ratio()
    passed = bool(similarity >= th.ocr_similarity_min and confidence >= th.ocr_confidence_min)
    return OCRResult(engine=engine, read_text=text, expected_text=expected_text, confidence=confidence,
                     similarity=similarity, passed=passed)
