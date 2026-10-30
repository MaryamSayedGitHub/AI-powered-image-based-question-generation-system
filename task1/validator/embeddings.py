"""B. Image-Text Embeddings — a small, local, CPU-friendly CLIP checkpoint via sentence-transformers
(`clip-ViT-B-32`, ~350MB, downloaded once and cached by the library). No LLM: this is a single
forward pass through an encoder, not text generation or reasoning.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from task1.config import SETTINGS

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer   # noqa: PLC0415 - optional, lazy import

        from task1.device import resolve_device

        _model = SentenceTransformer(SETTINGS.clip_model_name, device=resolve_device())
    return _model


@dataclass
class EmbeddingResult:
    question_similarity: float
    topic_similarity: float
    decoy_similarity: float | None
    margin_over_decoy: float | None
    passed: bool

    def as_dict(self) -> dict:
        return {"question_similarity": round(self.question_similarity, 4),
                "topic_similarity": round(self.topic_similarity, 4),
                "decoy_similarity": round(self.decoy_similarity, 4) if self.decoy_similarity is not None else None,
                "margin_over_decoy": round(self.margin_over_decoy, 4) if self.margin_over_decoy is not None else None,
                "passed": self.passed}


def _cosine(a, b) -> float:
    import numpy as np

    a, b = np.asarray(a), np.asarray(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def validate_image_text_match(image_path: Path | str, question_text: str, topic: str,
                              decoy_text: str = "a recipe for chocolate cake") -> EmbeddingResult:
    """One item at a time — simplest call, fine for Task 1's 3-item bank or interactive use."""
    results = validate_batch([(image_path, question_text, topic, decoy_text)])
    return results[0]


def validate_batch(items: list[tuple[Path | str, str, str, str]]) -> list[EmbeddingResult]:
    """Same check as `validate_image_text_match`, but ONE forward pass for every image and ONE for
    every text instead of one pair of calls per item — this is where a GPU (your RTX 5060) actually
    pays off: a batch of N images encodes in roughly the time one does on CPU, instead of N times
    that. Use this for the whole question bank (`pipeline.run_all`) rather than looping the
    single-item function."""
    th = SETTINGS.thresholds
    from PIL import Image

    model = _get_model()
    images = [Image.open(p).convert("RGB") for p, _, _, _ in items]
    texts = [t for _, q, topic, decoy in items for t in (q, topic, decoy)]   # 3 texts per item, flattened

    image_embs = model.encode(images, batch_size=min(32, len(images)) or 1)
    text_embs = model.encode(texts, batch_size=min(96, len(texts)) or 1)

    results = []
    for i, (image_emb) in enumerate(image_embs):
        q_emb, topic_emb, decoy_emb = text_embs[3 * i], text_embs[3 * i + 1], text_embs[3 * i + 2]
        q_sim, topic_sim, decoy_sim = _cosine(image_emb, q_emb), _cosine(image_emb, topic_emb), _cosine(image_emb, decoy_emb)
        margin = q_sim - decoy_sim
        passed = bool(q_sim >= th.embedding_similarity_min and topic_sim >= th.embedding_similarity_min
                      and margin >= th.embedding_decoy_margin_min)
        results.append(EmbeddingResult(question_similarity=q_sim, topic_similarity=topic_sim,
                                       decoy_similarity=decoy_sim, margin_over_decoy=margin, passed=passed))
    return results
