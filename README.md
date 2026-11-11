# RAG
small production-style RAG system built over messy construction and safety documents

## Quick start
Dependency-free (stdlib only) RAG over text and Markdown construction/safety docs in `data/`.
For PDF files, install PyMuPDF with `python -m pip install pymupdf`.
PDFs must contain extractable text; scanned images require OCR first.

    python -m rag.cli "what is the guardrail top rail height?"
    python -m pytest tests

- **Chunking** (`rag/chunking.py`): cleans OCR/page noise, splits on headings and page boundaries, then sentence-bounded windows (600 chars, 100 overlap); heading + page kept as metadata. PDF citations use physical page numbers.
- **Hybrid search** (`rag/search.py`): BM25 + char-4-gram TF-IDF cosine (robust to typos/OCR), fused with Reciprocal Rank Fusion.
- **Citations** (`rag/cli.py`): extractive answer; each sentence tagged `[n]` mapping to document, page, section heading and chunk id.
