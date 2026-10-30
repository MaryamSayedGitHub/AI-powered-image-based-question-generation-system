"""Simple local deployment — a one-page Streamlit app to try Task 1 interactively: pick a question
from the bank (or type your own + its parts), hit Generate, and see the image + the full validation
report + the click-and-answer metadata, live.

Run:
    uv add streamlit
    uv run streamlit run app.py

Nothing here talks to the network except whatever the chosen backend/validator already needs
(diffusion/gan backends, easyocr/sentence-transformers model downloads on first use) — the app
itself is 100% local.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from task1.generator.bank import load_question_bank
from task1.generator.renderer import render_question_image
from task1.metadata import build_click_answer_metadata
from task1.reporter.report import build_report
from task1.schemas import BBox, Question, Target
from task1.validator.pipeline import validate

st.set_page_config(page_title="Task 1 — Image-Based Questions", page_icon="🖼️", layout="centered")
st.title("🖼️ Generate Image-Based Questions")
st.caption("Local, zero-LLM pipeline: draw a question → run it through the CV validation gate.")

OUT_DIR = Path(__file__).resolve().parent / "output" / "app_runs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIAGRAM_TYPES = {
    "diagram_labeled": "Labeled diagram (e.g. a cell, a body) — targets are circles around the blob",
    "shape_question": "Geometric shapes (square / triangle / pentagon) in a row",
    "chart_question": "Bar chart — name each target like \"Bar C (27)\" so the number becomes the bar's height",
}

mode = st.radio("Source", ["Pick from the question bank", "Write a custom question"], horizontal=True)

question: Question | None = None

if mode == "Pick from the question bank":
    bank = {q.id: q for q in load_question_bank()}
    qid = st.selectbox("Question", list(bank), format_func=lambda i: f"{i} — {bank[i].question_text}")
    question = bank[qid]
    st.info(f"**{question.question_text}**  (type: `{question.type}`, topic: `{question.topic}`)")

else:
    st.subheader("Your question")
    q_type = st.selectbox("Diagram type", list(DIAGRAM_TYPES), format_func=lambda t: f"{t} — {DIAGRAM_TYPES[t]}")
    q_text = st.text_input("Question text (drawn on the image)", "Which labeled part is X?")
    q_topic = st.text_input("Topic (used by the embedding check)", "general")
    n_targets = st.slider("How many labeled parts (A, B, C, ...)", 2, 5, 3)

    st.caption("For a chart question, put the bar's value in the name, e.g. `Bar C (27)`.")
    targets = []
    cols = st.columns(n_targets)
    labels = [chr(ord("A") + i) for i in range(n_targets)]
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"**{labels[i]}**")
            name = st.text_input("name", key=f"name_{i}", label_visibility="collapsed",
                                 placeholder="e.g. Nucleus" if q_type != "chart_question" else "e.g. Bar A (12)")
            desc = st.text_input("description", key=f"desc_{i}", label_visibility="collapsed",
                                 placeholder="short description")
            category = st.text_input("category", key=f"cat_{i}", label_visibility="collapsed",
                                     placeholder="category", value="part")
            targets.append(Target(label=labels[i], name=name or f"Part {labels[i]}",
                                  description=desc or "", category=category or "part", bbox=BBox(0, 0, 0, 0)))

    correct_label = st.selectbox("Correct answer", labels)
    rubric_text = st.text_area("Rubric (one point per line)", "Names the correct part\nGives a reason")

    question = Question(id=f"custom_{int(time.time())}", type=q_type, topic=q_topic, question_text=q_text,
                        targets=targets, correct_label=correct_label,
                        rubric=[r for r in rubric_text.splitlines() if r.strip()])

st.divider()

if st.button("Generate & validate", type="primary", use_container_width=True):
    with st.spinner("Drawing the image..."):
        img_path = OUT_DIR / f"{question.id}.png"
        item = render_question_image(question, img_path)

    st.image(str(img_path), caption=question.question_text, use_container_width=True)

    with st.spinner("Running the validation gate (quality / OCR / embedding / region)..."):
        result = validate(item)
        report = build_report(item, result)
        metadata = build_click_answer_metadata(item)

    st.subheader("✅ Passed" if result.passed else "❌ Failed")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Quality", "OK" if result.quality.passed else "FAIL")
    c2.metric("OCR", "OK" if result.ocr.passed else "FAIL", help=f"read: {result.ocr.read_text!r}")
    c3.metric("Embedding", "skipped" if result.embedding is None else ("OK" if result.embedding.passed else "FAIL"),
             help=result.embedding_skipped_reason or "")
    c4.metric("Regions", "OK" if all(r.passed for r in result.regions) else "FAIL")

    with st.expander("Full JSON report"):
        st.json(report)
    with st.expander("Click-and-answer metadata (for Task 6)"):
        st.json(metadata)

    st.caption(f"Saved to `{img_path}`")
