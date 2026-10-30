"""Programmatic generation of the educational diagram itself (no generative model, no network):
matplotlib draws the shapes/chart, every labeled part is a real `matplotlib.patches.Patch` so its
on-screen pixel bounding box can be read back exactly (see `bbox_of`) instead of guessed.

One builder per question `type`. Add a new diagram family by writing one function with this
signature and registering it in `BUILDERS`.
"""
from __future__ import annotations

from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from task1.schemas import BBox, Question

Builder = Callable[["plt.Axes", Question], dict[str, "mpatches.Patch"]]


def _label(ax, patch_center: tuple[float, float], label: str, color: str = "#102a43") -> None:
    ax.text(*patch_center, label, ha="center", va="center", fontsize=16, fontweight="bold", color=color)


def build_diagram_labeled(ax: "plt.Axes", question: Question) -> dict[str, mpatches.Patch]:
    """A generic "labeled blob diagram" (cell, body, machine, ...) — an outer boundary plus one
    ellipse per target, arranged automatically around the centre."""
    import math

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.add_patch(mpatches.Ellipse((5, 5), 8.6, 8.6, fc="#fdf6ec", ec="#444", lw=2.5))

    palette = [("#c6dcf0", "#2b5f8a"), ("#f3c78e", "#8a5a1f"), ("#cdeccb", "#2f7a2a"),
               ("#f0c6df", "#8a2b6a"), ("#e3d9c6", "#6a5a2b")]
    n = len(question.targets)
    patches: dict[str, mpatches.Patch] = {}
    for i, target in enumerate(question.targets):
        angle = math.pi / 2 - 2 * math.pi * i / max(n, 1)
        cx, cy = 5 + 2.6 * math.cos(angle), 5 + 2.2 * math.sin(angle)
        face, edge = palette[i % len(palette)]
        w, h = (2.6, 2.3) if target.label == question.correct_label else (1.9, 1.5)
        patch = mpatches.Ellipse((cx, cy), w, h, fc=face, ec=edge, lw=2)
        ax.add_patch(patch)
        _label(ax, (cx, cy), target.label, edge)
        patches[target.label] = patch
    return patches


def build_shape_question(ax: "plt.Axes", question: Question) -> dict[str, mpatches.Patch]:
    """A row of distinct geometric shapes, one per target, so the shapes themselves (not just the
    label) answer the question — a square/triangle/pentagon drawn as real `Polygon` patches."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    shape_by_name = {
        "square": lambda c: mpatches.RegularPolygon(c, numVertices=4, radius=1.3, orientation=0.785),
        "triangle": lambda c: mpatches.RegularPolygon(c, numVertices=3, radius=1.4, orientation=0),
        "pentagon": lambda c: mpatches.RegularPolygon(c, numVertices=5, radius=1.3, orientation=0),
    }
    n = len(question.targets)
    xs = [10 * (i + 1) / (n + 1) for i in range(n)]
    patches: dict[str, mpatches.Patch] = {}
    for x, target in zip(xs, question.targets):
        key = next((k for k in shape_by_name if k in target.name.lower()), None)
        builder = shape_by_name.get(key, shape_by_name["square"])
        patch = builder((x, 5.2))
        patch.set_facecolor("#c6dcf0")
        patch.set_edgecolor("#2b5f8a")
        patch.set_linewidth(2)
        ax.add_patch(patch)
        _label(ax, (x, 2.6), target.label)
        patches[target.label] = patch
    return patches


def build_chart_question(ax: "plt.Axes", question: Question) -> dict[str, mpatches.Patch]:
    """A bar chart where each bar IS a target — values parsed from each target's name,
    e.g. "Bar C (27)" -> 27, so the chart and the question bank never disagree."""
    import re

    labels, values = [], []
    for target in question.targets:
        match = re.search(r"\((\d+(?:\.\d+)?)\)", target.name)
        values.append(float(match.group(1)) if match else 0.0)
        labels.append(target.label)

    bars = ax.bar(labels, values, color="#8fb8de", edgecolor="#2b5f8a", linewidth=2)
    ax.set_ylim(0, max(values) * 1.25 if values else 1)
    ax.set_xlabel("")
    ax.set_ylabel("value")
    ax.tick_params(axis="both", labelsize=11)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return dict(zip(labels, bars.patches))


BUILDERS: dict[str, Builder] = {
    "diagram_labeled": build_diagram_labeled,
    "shape_question": build_shape_question,
    "chart_question": build_chart_question,
}


def bbox_of(fig: "plt.Figure", patch: "mpatches.Patch") -> BBox:
    """The patch's REAL on-screen pixel box (matplotlib display coords, origin bottom-left) ->
    image pixel coords (origin top-left) — read back, never estimated from data coordinates."""
    renderer = fig.canvas.get_renderer()
    extent = patch.get_window_extent(renderer=renderer)
    fig_h_px = fig.get_size_inches()[1] * fig.dpi
    x0, y1 = extent.x0, fig_h_px - extent.y0   # top-left y = flip the lower display y
    w, h = extent.width, extent.height
    return BBox(x=int(round(x0)), y=int(round(y1 - h)), w=int(round(w)), h=int(round(h)))
