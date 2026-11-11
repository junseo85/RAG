"""Hybrid retrieval: BM25 (lexical) + hashed char-n-gram TF-IDF cosine (fuzzy, handles
OCR noise and morphology), fused with Reciprocal Rank Fusion. Pure standard library."""
import math
import re
from collections import Counter

STOP = set("a an and are as at be by for from in is it of on or shall the to with must".split())


def stem(t):
    return t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t


def tokens(text):
    return [stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


def ngrams(text, n=4):
    out = []
    for t in tokens(text):
        t = f"#{t}#"
        out += [t[i:i + n] for i in range(max(1, len(t) - n + 1))]
    return out


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = sum(self.len) / max(1, len(docs))
        df = Counter(t for d in self.tf for t in d)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + .5) / (c + .5)) for t, c in df.items()}

    def scores(self, q):
        res = []
        for tf, ln in zip(self.tf, self.len):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            res.append(s)
        return res


class NgramCosine:
    def __init__(self, docs):
        n = len(docs)
        df = Counter(t for d in docs for t in set(d))
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
        self.vecs = [self._vec(d) for d in docs]

    def _vec(self, grams):
        v = {t: c * self.idf.get(t, 1.0) for t, c in Counter(grams).items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def scores(self, q):
        qv = self._vec(q)
        return [sum(w * v.get(t, 0.0) for t, w in qv.items()) for v in self.vecs]


def ranks(scores):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return {i: r for r, i in enumerate(order) if scores[i] > 0}


class HybridIndex:
    def __init__(self, chunks, rrf_k=60):
        self.chunks, self.rrf_k = chunks, rrf_k
        self.bm25 = BM25([tokens(c.text) for c in chunks])
        self.ngram = NgramCosine([ngrams(c.text) for c in chunks])

    def search(self, query, k=3):
        lex = ranks(self.bm25.scores(tokens(query)))
        fuzzy = ranks(self.ngram.scores(ngrams(query)))
        fused = {}
        for r in (lex, fuzzy):
            for i, rank in r.items():
                fused[i] = fused.get(i, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        top = sorted(fused, key=lambda i: -fused[i])[:k]
        return [(self.chunks[i], fused[i]) for i in top]
