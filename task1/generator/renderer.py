"""Composes a finished item: a background image from whichever backend is configured
(generator.backends — programmatic/diffusion/gan) + the question text drawn on top with PIL
(template-based rendering — exact glyphs, real text-wrapping, same regardless of backend).
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from task1.config import SETTINGS
from task1.fonts import resolve_font_path
from task1.generator.backends import get_backend
from task1.schemas import BBox, GeneratedItem, Question


def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """A real TTF at the requested size if one can be found at all (see fonts.py); PIL's built-in
    bitmap font — fixed size, no bold, but always available — only if truly nothing else is found,
    so the pipeline still runs end to end on a bare-bones machine instead of crashing on import."""
    path = resolve_font_path(SETTINGS.font_path)
    if path:
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textbbox((0, 0), trial, font=font)[2] <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def render_question_image(question: Question, out_path: Path | str,
                          image_size: tuple[int, int] | None = None,
                          backend: str | None = None) -> GeneratedItem:
    """Builds the background image with whichever backend is configured (default: programmatic —
    the only one with real per-target bboxes), draws the question banner on top, saves the PNG, and
    returns a GeneratedItem ready for the validator. `backend` overrides SETTINGS.generation_backend
    for this one call without touching global config."""
    out_path = Path(out_path)
    width, height = image_size or (SETTINGS.image_width, SETTINGS.image_height)
    banner_h = int(height * 0.18)
    diagram_h = height - banner_h

    diagram_img, bboxes = get_backend(backend).build(question, width, diagram_h)

    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(diagram_img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, diagram_h, width, height], fill="#1b2a4a")

    size = 30
    font = _load_font(size)
    max_w = width - 48
    lines = _wrap_text(draw, question.question_text, font, max_w)
    while len(lines) > 2 and size > 14:
        size -= 2
        font = _load_font(size)
        lines = _wrap_text(draw, question.question_text, font, max_w)
    line_h = draw.textbbox((0, 0), "A", font=font)[3] + 6
    total_h = line_h * len(lines)
    y = diagram_h + (banner_h - total_h) // 2
    for line in lines:
        w = draw.textbbox((0, 0), line, font=font)[2]
        draw.text(((width - w) / 2, y), line, font=font, fill="white")
        y += line_h
    text_bbox = BBox(x=24, y=diagram_h, w=width - 48, h=banner_h)

    canvas.save(out_path)

    for target in question.targets:
        if target.label in bboxes:
            target.bbox = bboxes[target.label]

    return GeneratedItem(question=question, image_path=str(out_path), image_size=(width, height),
                         text_bbox=text_bbox)
