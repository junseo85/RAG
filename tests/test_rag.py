import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rag.chunking import chunk_document
from rag.cli import answer, build_index

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"


def test_chunks_respect_sections_and_pages():
    raw = (DATA / "osha_fall_protection.txt").read_text()
    chunks = chunk_document("x", raw)
    assert any("1926.502" in c.heading and c.page == 2 for c in chunks)
    assert all("Page 2 of 3" not in c.text for c in chunks)


def test_section_continuing_across_text_pages_has_correct_citations():
    raw = "1. Scope\nFirst page text.\nPage 2 of 2\nSecond page text."
    chunks = chunk_document("x", raw)
    assert [(c.heading, c.page, c.text.split("\n")[-1]) for c in chunks] == [
        ("1. Scope", 1, "First page text."),
        ("1. Scope", 2, "Second page text."),
    ]


def test_pdf_pages_use_physical_pages_and_preserve_section():
    pages = [
        "1. Scope\nFirst page text.",
        "Page 99\nSecond page text.",
    ]
    chunks = chunk_document("x.pdf", "", pages=pages)
    assert [(c.heading, c.page, c.text.split("\n")[-1]) for c in chunks] == [
        ("1. Scope", 1, "First page text."),
        ("1. Scope", 2, "Second page text."),
    ]


def test_build_index_reads_pdf(tmp_path):
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "safety.pdf"
    with fitz.open() as pdf:
        pdf.new_page().insert_text((72, 72), "1. Scope\nFirst page text.")
        pdf.new_page().insert_text(
            (72, 72), "Guardrails must withstand 200 pounds."
        )
        pdf.save(path)

    out = answer(build_index(tmp_path), "guardrails withstand 200 pounds", k=1)
    assert "200 pounds" in out
    assert "safety.pdf, p.2" in out


def test_build_index_reports_pdf_without_extractable_text(tmp_path):
    fitz = pytest.importorskip("fitz")
    with fitz.open() as pdf:
        pdf.new_page()
        pdf.save(tmp_path / "scanned.pdf")

    with pytest.raises(ValueError, match="OCR may be required"):
        build_index(tmp_path)


def test_long_section_is_windowed_with_overlap():
    raw = "1. Scope\n" + " ".join(f"Sentence number {i} is here." for i in range(60))
    chunks = chunk_document("x", raw, max_chars=200, overlap=60)
    assert len(chunks) > 3 and all(len(c.text) < 300 for c in chunks)


def test_hybrid_search_and_citation():
    idx = build_index(DATA)
    out = answer(idx, "what is the guardrail top rail height?")
    assert "42 inches" in out and "osha_fall_protection.txt" in out
    out = answer(idx, "concrete cylinders testing frequency")
    assert "spec_03_30_00.txt" in out
    out = answer(idx, "trench depth requiring protective system")
    assert "excavation_procedure.md" in out
