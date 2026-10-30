"""D. Image Quality Assessment — OpenCV only, no model weights, instant and free."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from task1.config import SETTINGS


@dataclass
class QualityResult:
    width: int
    height: int
    blur_variance: float          # Laplacian variance — higher = sharper
    contrast_std: float           # grayscale std-dev — higher = more contrast
    resolution_ok: bool
    blur_ok: bool
    contrast_ok: bool

    @property
    def passed(self) -> bool:
        return self.resolution_ok and self.blur_ok and self.contrast_ok

    def as_dict(self) -> dict:
        return {"width": self.width, "height": self.height, "blur_variance": round(self.blur_variance, 2),
                "contrast_std": round(self.contrast_std, 2), "resolution_ok": self.resolution_ok,
                "blur_ok": self.blur_ok, "contrast_ok": self.contrast_ok, "passed": self.passed}


def assess_quality(image_path: Path | str) -> QualityResult:
    th = SETTINGS.thresholds
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    h, w = (int(d) for d in img.shape[:2])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(np.std(gray))
    return QualityResult(
        width=w, height=h, blur_variance=blur_var, contrast_std=contrast,
        resolution_ok=bool(w >= th.min_width and h >= th.min_height),
        blur_ok=bool(blur_var >= th.blur_min_variance),
        contrast_ok=bool(contrast >= th.contrast_min_std),
    )
