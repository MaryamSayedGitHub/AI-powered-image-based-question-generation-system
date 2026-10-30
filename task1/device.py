"""Picks CPU vs. your GPU once, for whichever optional validators are installed (EasyOCR,
sentence-transformers). An RTX 5060 8GB comfortably fits both: EasyOCR's detector+recognizer are a
few hundred MB, and `clip-ViT-B-32` is ~350MB — nowhere near the 8GB ceiling, so there's no need to
pick a smaller/bigger model for VRAM reasons; GPU just makes each check a few ms instead of ~1s.
"""
from __future__ import annotations

from functools import lru_cache

from task1.config import SETTINGS


@lru_cache(maxsize=1)
def resolve_device() -> str:
    """SETTINGS.device: "auto" (default) picks cuda if torch sees one, else cpu; "cuda"/"cpu" force it."""
    if SETTINGS.device != "auto":
        return SETTINGS.device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"     # torch isn't installed at all -> easyocr/sentence-transformers aren't either
