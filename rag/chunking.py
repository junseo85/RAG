"""Section-aware chunking with sentence-bounded sliding windows.

Decisions:
* Text is cleaned (page markers dropped, hyphenated line breaks joined, whitespace
  normalised) but page numbers are tracked so citations can point at a page.
* Documents are first split on structural headings (markdown '#', numbered clauses
  like '1926.501' / '2.01' / '3.', 'PART n', 'SECTION ...') so a chunk never straddles
  two sections; the heading is stored as metadata and prepended to the chunk text.
* Oversized sections are split on sentence boundaries into windows of ~max_chars with
  an overlap of ~overlap chars so facts at a boundary stay retrievable.
"""
import re
from dataclasses import dataclass

PAGE_RE = re.compile(r"^\s*(?:page|pg\.?)\s*(\d+)(?:\s*of\s*\d+)?\s*$", re.I)
HEADING_RE = re.compile(
    r"^\s*(?:#{1,6}\s+.+|SECTION\s+[\d ]+.*|PART\s+\d+.*|\d+(?:\.\d+)+\s+[A-Z].*|"
    r"\d+\.\s+[A-Z][^.]*|[A-Z][A-Z0-9 ,/&\-]{6,})\s*$"
)
SENT_RE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z(0-9])")


@dataclass
class Chunk:
    id: str
    doc: str
    heading: str
    page: int
    text: str


def clean_line(line):
    return re.sub(r"\s+", " ", line).strip()


def split_sections(raw, *, pages=None):
    """Yield (heading, page, body) tuples."""
    heading, page, start_page, body, out = "", 1, 1, [], []

    def flush():
        text = " ".join(body).strip()
        if text:
            out.append((heading, start_page, text))

    for physical_page, text in enumerate(pages if pages is not None else [raw], 1):
        if pages is not None and physical_page != page:
            flush()
            body = []
            page = physical_page
        text = re.sub(r"-\n(?=[a-z])", "", text)
        for line in text.splitlines():
            m = PAGE_RE.match(line)
            if m:
                marker_page = int(m.group(1))
                if pages is None and marker_page != page:
                    flush()
                    body = []
                    page = marker_page
                continue
            line = clean_line(line)
            if not line:
                body.append("\n") if body and body[-1] != "\n" else None
                continue
            if HEADING_RE.match(line) and len(line) < 100:
                flush()
                heading, start_page, body = line.lstrip("# ").strip(), page, []
                continue
            if not body:
                start_page = page
            body.append(line)
    flush()
    return [(h, p, re.sub(r"\s*\n\s*", " ", t)) for h, p, t in out]


def window(text, max_chars, overlap):
    if len(text) <= max_chars:
        return [text]
    sents, chunks, cur = SENT_RE.split(text), [], []
    for s in sents:
        if cur and sum(len(x) + 1 for x in cur) + len(s) > max_chars:
            chunks.append(" ".join(cur))
            keep, n = [], 0
            for x in reversed(cur):
                if n + len(x) > overlap:
                    break
                keep.insert(0, x)
                n += len(x) + 1
            cur = keep
        cur.append(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def chunk_document(doc, raw, max_chars=600, overlap=100, *, pages=None):
    chunks = []
    for heading, page, body in split_sections(raw, pages=pages):
        for piece in window(body, max_chars, overlap):
            text = f"{heading}\n{piece}" if heading else piece
            chunks.append(Chunk(f"{doc}#{len(chunks)}", doc, heading, page, text))
    return chunks
