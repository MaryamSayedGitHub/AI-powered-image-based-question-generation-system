"""Central configuration and thresholds for Task 1.

Nothing here calls an LLM and nothing here needs a network connection at import time — model
weights (EasyOCR detector/recognizer, the CLIP checkpoint) are downloaded once, lazily, the first
time they're actually used, and then cached locally by their own libraries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_QUESTION_BANK = PACKAGE_ROOT / "question_bank.json"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT.parent / "output"


@dataclass(frozen=True)
class Thresholds:
    """Every pass/fail cutoff in one place, so tuning the pipeline means editing numbers here,
    not hunting through validator modules."""

    # B. image quality (validator/quality.py)
    min_width: int = 400
    min_height: int = 400
    blur_min_variance: float = 100.0      # Laplacian variance below this = too blurry
    contrast_min_std: float = 20.0        # grayscale std-dev below this = too flat / low contrast

    # A. OCR (validator/ocr.py)
    ocr_similarity_min: float = 0.85      # difflib ratio between read text and expected text
    ocr_confidence_min: float = 0.40      # EasyOCR / pytesseract confidence, 0-1

    # B. image-text embeddings (validator/embeddings.py)
    embedding_similarity_min: float = 0.20   # cosine sim, image vs. its own question/topic
    embedding_decoy_margin_min: float = 0.05  # how much higher than an unrelated sentence

    # C. region / template verification (validator/matching.py)
    region_distinctiveness_min: float = 0.08  # how different the target bbox is from the background


@dataclass(frozen=True)
class Settings:
    thresholds: Thresholds = field(default_factory=Thresholds)
    easyocr_langs: tuple[str, ...] = ("en",)      # add "ar" once an Arabic-capable font/text is used
    clip_model_name: str = "clip-ViT-B-32"        # small, CPU-friendly sentence-transformers CLIP
    device: str = "auto"                          # "auto" | "cuda" | "cpu" — see device.py
    question_bank_path: Path = DEFAULT_QUESTION_BANK
    output_dir: Path = DEFAULT_OUTPUT_DIR
    image_width: int = 900
    image_height: int = 700
    font_path: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

    # generator/backends.py — "programmatic" (default, required by the spec) | "diffusion" | "gan"
    generation_backend: str = "programmatic"
    diffusion_model: str = "stabilityai/sd-turbo"   # 1-step distilled SD; fits+runs fast on 8GB at fp16
    diffusion_steps: int = 1
    diffusion_guidance_scale: float = 0.0           # SD-Turbo is trained for guidance-free sampling
    gan_hub_model: str = "DCGAN"                    # facebookresearch/pytorch_GAN_zoo hub entry point


SETTINGS = Settings()
