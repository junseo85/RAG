# RAG
small production-style RAG system built over messy construction and safety documents

## Quick start
Dependency-free (stdlib only) RAG over messy construction/safety docs in `data/`.

    python -m rag.cli "how high must a guardrail top rail be?"
    python -m pytest tests

- **Chunking** (`rag/chunking.py`): cleans OCR/page noise, splits on headings/clauses, then sentence-bounded windows (600 chars, 100 overlap); heading + page kept as metadata.
- **Hybrid search** (`rag/search.py`): BM25 + char-4-gram TF-IDF cosine (robust to typos/OCR), fused with Reciprocal Rank Fusion.
- **Citations** (`rag/cli.py`): extractive answer; each sentence tagged `[n]` mapping to document, page, section heading and chunk id.
