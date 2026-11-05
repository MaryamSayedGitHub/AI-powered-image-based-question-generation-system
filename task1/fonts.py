"""Resolves a real TrueType font file to draw the question-banner text with, portably.

`renderer.py` needs an actual .ttf (PIL's built-in default font has no usable size control and
looks wrong at banner scale). The old approach hardcoded one Debian/Ubuntu path
(`/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf`) — correct here, but it breaks for any
teammate on macOS, Windows, or another Linux distro that doesn't have that exact file at that
exact path.

matplotlib is already a hard dependency (`generator/diagrams.py`) and it VENDORS its own copy of
DejaVu Sans Bold inside the installed package — same font file, same license, same look, on every
OS matplotlib runs on — so `matplotlib.font_manager.findfont()` is a free, always-available way to
find a real bold TTF without asking the user to have specific system fonts installed.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def resolve_font_path(configured_path: str | None = None) -> str | None:
    """`configured_path` (SETTINGS.font_path) wins if it's set and the file actually exists — that
    lets anyone pin a specific font on purpose. Otherwise falls back to matplotlib's bundled DejaVu
    Sans Bold (present wherever this project runs, since matplotlib is a required dependency), then
    to a short list of common system paths as a last resort. Returns None only if nothing usable was
    found at all, in which case the caller should fall back to PIL's built-in bitmap font."""
    if configured_path and Path(configured_path).is_file():
        return configured_path

    try:
        import matplotlib.font_manager as fm

        found = fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold"), fallback_to_default=False)
        if found and Path(found).is_file():
            return found
    except Exception:                                                    # noqa: BLE001 - fonts are best-effort
        pass

    common_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",            # Debian/Ubuntu
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",                     # Fedora/RHEL
        "/Library/Fonts/Arial Bold.ttf",                                   # macOS
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",               # macOS
        "C:\\Windows\\Fonts\\arialbd.ttf",                                 # Windows
    ]
    for candidate in common_paths:
        if Path(candidate).is_file():
            return candidate

    return None
