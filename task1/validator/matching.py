"""C. Object Detection / Template Matching — pure OpenCV, no detector model to download.

Two checks, both useful beyond Task 1:
  - `verify_region_distinctiveness`: is there actually something drawn at the target's claimed
    `expected_visual_region`, or is the bbox pointing at blank background? (catches a generator/
    bbox-math bug before it ever reaches a student).
  - `match_template`: does a saved crop (e.g. a Task-6 SAM crop, or a known icon) re-appear inside
    a given image, and where? Reusable by Task 6's click-and-answer flow unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from task1.config import SETTINGS
from task1.schemas import BBox


@dataclass
class RegionResult:
    label: str
    distinctiveness: float     # 0-1-ish: how different the region is from the image background
    passed: bool

    def as_dict(self) -> dict:
        return {"label": self.label, "distinctiveness": round(self.distinctiveness, 4), "passed": self.passed}


def _edge_density(gray_region: np.ndarray) -> float:
    edges = cv2.Canny(gray_region, 50, 150)
    return float(np.count_nonzero(edges)) / edges.size


def verify_region_distinctiveness(image_path: Path | str, label: str, bbox: BBox) -> RegionResult:
    """Compares the target region's edge density + color variance against a background sample
    (the image's four corners) — a real drawn shape has far more edges/variance than blank fill."""
    th = SETTINGS.thresholds
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    h, w = img.shape[:2]
    x0, y0 = max(0, bbox.x), max(0, bbox.y)
    x1, y1 = min(w, bbox.x + bbox.w), min(h, bbox.y + bbox.h)
    region = img[y0:y1, x0:x1]
    if region.size == 0:
        return RegionResult(label=label, distinctiveness=0.0, passed=False)

    region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    sample = max(6, min(40, w // 20, h // 20))
    corners = [img[0:sample, 0:sample], img[0:sample, w - sample:w],
               img[h - sample:h, 0:sample], img[h - sample:h, w - sample:w]]
    # each corner's OWN edge-density/std, then averaged — never concatenate pixels from different
    # uniform-colour corners directly, which would fabricate variance between the corners
    # themselves (e.g. a white corner next to a solid-colour banner corner) rather than measuring
    # local texture.
    corner_grays = [cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) for c in corners]
    bg_score = float(np.mean([_edge_density(g) + np.std(g) / 255.0 for g in corner_grays]))

    region_score = _edge_density(region_gray) + np.std(region_gray) / 255.0
    distinctiveness = float(max(0.0, region_score - bg_score))
    return RegionResult(label=label, distinctiveness=distinctiveness,
                        passed=bool(distinctiveness >= th.region_distinctiveness_min))


@dataclass
class TemplateMatchResult:
    score: float               # 0-1, normalized cross-correlation at the best match
    location: tuple[int, int]  # top-left pixel of the best match in the searched image
    found: bool

    def as_dict(self) -> dict:
        return {"score": round(self.score, 4), "location": self.location, "found": self.found}


def match_template(search_image_path: Path | str, template_path: Path | str,
                   min_score: float = 0.6) -> TemplateMatchResult:
    """Does `template_path` (e.g. a saved crop) appear inside `search_image_path`, and where?
    Plain `cv2.matchTemplate`, resizing the template down if it's larger than the search image."""
    search = cv2.imread(str(search_image_path), cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    if search is None or template is None:
        raise FileNotFoundError("could not read search image or template")
    if template.shape[0] > search.shape[0] or template.shape[1] > search.shape[1]:
        scale = min(search.shape[0] / template.shape[0], search.shape[1] / template.shape[1]) * 0.9
        template = cv2.resize(template, None, fx=scale, fy=scale)
    result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return TemplateMatchResult(score=float(max_val), location=max_loc, found=bool(max_val >= min_score))
