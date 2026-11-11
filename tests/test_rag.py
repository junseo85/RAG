import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rag.chunking import chunk_document
from rag.cli import answer, build_index

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"


def test_chunks_respect_sections_and_pages():
    raw = (DATA / "osha_fall_protection.txt").read_text()
    chunks = chunk_document("x", raw)
    assert any("1926.502" in c.heading and c.page == 2 for c in chunks)
    assert all("Page 2 of 3" not in c.text for c in chunks)


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
