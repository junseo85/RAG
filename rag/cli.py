"""Usage: python -m rag.cli "question" [--data data] [-k 3]"""
import argparse
import pathlib

from .chunking import chunk_document
from .search import HybridIndex, tokens


def build_index(data_dir):
    chunks = []
    for p in sorted(pathlib.Path(data_dir).glob("*")):
        if p.suffix.lower() in (".txt", ".md"):
            chunks += chunk_document(p.name, p.read_text(errors="replace"))
    return HybridIndex(chunks)


def best_sentences(chunk, query, n=2):
    q = set(tokens(query))
    sents = [s for s in chunk.text.split("\n", 1)[-1].replace("\n", " ").split(". ") if s]
    sents.sort(key=lambda s: -len(q & set(tokens(s))))
    return ". ".join(s.rstrip(".") for s in sents[:n]) + "."


def answer(index, query, k=3):
    hits = index.search(query, k)
    if not hits:
        return "No supporting passage found."
    lines = []
    for n, (c, score) in enumerate(hits, 1):
        lines.append(f"{best_sentences(c, query)} [{n}]")
    lines.append("")
    for n, (c, score) in enumerate(hits, 1):
        lines.append(f"[{n}] {c.doc}, p.{c.page}, \"{c.heading}\" (chunk {c.id}, score {score:.4f})")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--data", default="data")
    ap.add_argument("-k", type=int, default=3)
    a = ap.parse_args()
    print(answer(build_index(a.data), a.query, a.k))


if __name__ == "__main__":
    main()
