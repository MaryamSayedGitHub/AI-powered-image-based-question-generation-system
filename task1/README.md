# Task 1 — Generate Image-Based Questions (zero-LLM, local, CPU-friendly)

A local pipeline that **draws** an educational diagram + its question text by code (no generative
model), then runs a full Computer-Vision validation gate on the result, and emits the exact
metadata a click-and-answer / segmentation task downstream needs.

```
task1/
  config.py          thresholds + settings, one place to tune
  schemas.py          BBox / Target / Question / GeneratedItem dataclasses shared by every module
  question_bank.json   3 worked examples: a labeled diagram, a shapes question, a bar chart
  generator/
    bank.py            loads question_bank.json into the dataclasses
    diagrams.py         matplotlib builders, one per question type (diagram_labeled / shape_question /
                        chart_question) — returns real `Patch` objects so bboxes are READ, not guessed
    renderer.py          composes diagram + question-text banner (PIL, text-wrapped), saves the PNG
  validator/
    quality.py          D. OpenCV: resolution, blur (Laplacian variance), contrast (grayscale std)
    ocr.py               A. EasyOCR primary / pytesseract fallback, text-similarity gate
    embeddings.py         B. local CLIP (`clip-ViT-B-32` via sentence-transformers), image-text cosine sim
    matching.py           C. OpenCV: region-distinctiveness check (bbox vs background) +
                        `match_template` (cv2.matchTemplate, reusable by Task 6)
    pipeline.py           runs all four checks, one ValidationResult
  reporter/report.py    3. the structured JSON quality report
  metadata.py          4. click-and-answer integration metadata, exactly the requested shape
  pipeline.py           top-level orchestrator: generate -> validate -> report -> metadata
  cli.py                `python -m task1.cli`
tests/test_pipeline.py  pytest smoke tests (no network, no LLM)
```

## Run it

```bash
pip install -r requirements.txt
python -m task1.cli                       # all 3 question-bank entries -> ./output/
python -m task1.cli --question-id bio_cell_001
python -m pytest tests/ -q
```

## Try it interactively — `app.py` (simple local deployment)

```bash
uv add streamlit
uv run streamlit run app.py
```

Opens a local page (`http://localhost:8501`) in your browser. Pick a question from the bank, or
switch to "Write a custom question" to type your own question text, topic, and 2-5 labeled parts
(name/description/category per part + which one is correct) — press **Generate & validate** and it
draws the image, runs the full CV validation gate, and shows the image, a pass/fail breakdown per
check, the full JSON report, and the click-and-answer metadata, all on one page. Nothing leaves your
machine except what the chosen backend/validator already needs (none, by default).

`easyocr` and `sentence-transformers` are optional: OCR falls back to `pytesseract` (needs the
system `tesseract-ocr` binary) if `easyocr` isn't installed, and the embedding check is skipped
(reported as `"skipped": "<reason>"`, never silently passed) if `sentence-transformers` isn't
installed, rather than crashing the whole run.

## Verified, not just written

This was generated and validated for real in a sandbox that had OpenCV/PIL/matplotlib/pytesseract
but **not** `easyocr` or `sentence-transformers` (no network to install them). All 3 question-bank
items passed quality + OCR + region checks for real; the embedding check correctly reports itself
as skipped with the exact missing-dependency reason rather than faking a score. Install the two
optional packages in your own environment and every check runs for real, no code changes needed.

## Generation backends — `generator/backends.py`

The spec's default (and the one the question bank + tests use) is **`programmatic`**: matplotlib
draws the diagram, so every target has a real, exact pixel bbox (`generator/diagrams.py: bbox_of`).
Two opt-in alternatives are wired in for when a diagram alone isn't the right picture:

| `SETTINGS.generation_backend` | What it does | Target bboxes | Needs |
|---|---|---|---|
| `"programmatic"` (default) | matplotlib shapes/chart | real, exact | nothing extra |
| `"diffusion"` | local SD-Turbo (1-step, GPU-friendly at fp16, fits 8GB) text-to-image for a photographic/illustrative background | **none** — see below | `torch`, `diffusers`, `accelerate`, `safetensors` |
| `"gan"` | pretrained DCGAN via `torch.hub` (`facebookresearch/pytorch_GAN_zoo`) | **none** | `torch` |

Switch with `SETTINGS.generation_backend = "diffusion"` in `config.py`, or per-call:
`render_question_image(question, path, backend="diffusion")`.

In every backend, the question **text** is still drawn by the same deterministic PIL step
afterward — never by the generative model — because in-image text generation is the one thing
diffusion/GAN models are still unreliable at (see the Task 7 study). Generative backends only
supply the background picture.

**Why the generative backends return no bboxes:** a diffusion/GAN image has no code-level notion of
"where the nucleus is" the way a matplotlib ellipse does — claiming one would be a guess reported as
a fact. This is exactly the Task 6 handoff point: run SAM on a generated image to get real
`expected_visual_region` boxes, the same way a teacher-supplied photo would be handled, instead of
Task 1 fabricating them.

**Why GAN is included but not recommended for this task:** an unconditional GAN (DCGAN's hub default
is faces) can't be asked for "a cell" or "a heart" — there's no text-conditioning path, so it can't
target a specific topic the way the diffusion backend can. It's wired up and genuinely runs (useful
for a generic textured/decorative backdrop), but `programmatic` or `diffusion` are the backends to
use for anything that has to depict a specific topic. See `generator/backends.py: GANBackend`'s
docstring for the full reasoning.

## Why code-drawn diagrams, not a text-to-image model

A generative model's in-image text is unreliable, especially outside English (see the companion
Task 7 study) — the question text, labels and numbers must be **exact**. Drawing with
matplotlib/PIL guarantees every character is a real font glyph, every label's pixel position is
read back from the actual rendered patch (`generator/diagrams.py: bbox_of`, via
`get_window_extent()`), not estimated — and it's free and instant.

## Output per question (`output/<id>.*`)

- `<id>.png` — the generated image
- `<id>.report.json` — the full quality report (resolution, blur, contrast, OCR text + confidence,
  embedding similarity, region distinctiveness, overall pass/fail)
- `<id>.metadata.json` — one entry per target:
  ```json
  {
    "target_name": "Nucleus",
    "target_description": "The organelle that controls the cell and stores its DNA",
    "target_category": "organelle",
    "expected_visual_region": {"x": 371, "y": 142, "w": 181, "h": 102}
  }
  ```

## Using a GPU (e.g. RTX 5060 8GB)

Nothing to configure by hand — `task1/device.py: resolve_device()` auto-detects CUDA and both
optional validators use it: `easyocr.Reader(gpu=...)` and `SentenceTransformer(..., device=...)`.
Force it with `SETTINGS.device = "cuda"` / `"cpu"` in `config.py` if you ever need to override the
auto-detection.

8GB is far more than either model needs (EasyOCR's models and `clip-ViT-B-32` are both well under
1GB), so the GPU isn't about fitting a bigger model — it's about **throughput on a batch**.
`validator/embeddings.py: validate_batch()` encodes every image and every text for a whole run in
one forward pass each, and `pipeline.run_all()` uses it (generate every question first, then one
batched validation pass) instead of looping the single-item path — that's where a GPU actually pays
off on a bank of more than a handful of questions. `run_question()` (single item) still uses the
one-item path; there's nothing to batch with just one image.

## Connecting to Task 6 (click-and-answer / segmentation)

- `expected_visual_region` in the metadata is exactly what a click-and-answer flow needs to decide
  which target a click landed on — generated here, consumed there, no format translation.
- `validator/matching.py: match_template` is plain `cv2.matchTemplate` and takes any two images —
  it works unchanged on a Task-6 SAM crop against a full image, not just on Task-1 output.
- `validator/embeddings.py`'s `clip-ViT-B-32` is a reasonable **local, zero-LLM** stand-in for the
  multilingual SigLIP 2 backbone recommended in the Task 7 study; swap the model name in
  `config.py` for SigLIP 2 once Task 6/7 share one backbone and one similarity index.
