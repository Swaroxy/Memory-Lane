"""Hybrid retrieval: FAISS semantic search + BM25 keyword search -> RRF fusion -> cross-encoder rerank."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .index import Chunk, KnowledgeBase

STOP = set("""a an and are as at be by can do does for from how i if in is it its me my of on or
our that the their there this to was we what when where which who why will with you your
many much long often must should tell please about""".split())


def _stem(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def tokens(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


def relevance(reranker, query: str, texts: list[str]) -> np.ndarray:
    """Cross-encoder relevance as probabilities in 0..1 (sigmoid applied if raw logits come back)."""
    if not texts:
        return np.zeros(0)
    p = np.asarray(reranker.predict([(query, t) for t in texts]), dtype="float64")
    if p.size and (p.min() < 0 or p.max() > 1):
        p = 1 / (1 + np.exp(-np.clip(p, -30, 30)))
    return p


@dataclass
class Hit:
    chunk: Chunk
    semantic: float = 0.0          # cosine similarity
    sem_rank: int | None = None
    bm25_rank: int | None = None
    fused: float = 0.0             # weighted reciprocal-rank fusion, 0..1
    rerank: float | None = None    # cross-encoder probability, 0..1
    rank: int = 0

    @property
    def score(self) -> float:
        return self.rerank if self.rerank is not None else self.fused


class Retriever:
    def __init__(self, kb: KnowledgeBase, reranker=None):
        self.kb, self.reranker = kb, reranker
        self._bm25, self._bm25_ids, self._bm25_version = None, [], -1

    def _bm25_index(self) -> BM25Okapi | None:
        if self._bm25_version != self.kb.version:
            self._bm25_ids = sorted(self.kb.chunks)
            corpus = [tokens(f"{self.kb.chunks[i].title} {self.kb.chunks[i].heading or ''} {self.kb.chunks[i].text}")
                      for i in self._bm25_ids]
            self._bm25 = BM25Okapi(corpus) if corpus else None
            self._bm25_version = self.kb.version
        return self._bm25

    def search(self, query: str, pool: int = 20, semantic_weight: float = 0.6, rrf_k: int = 60,
               rerank: bool = True) -> list[Hit]:
        """Return the candidate pool, best first. Callers take ``[:top_k]`` for the final evidence."""
        kb = self.kb
        if not kb.chunks or not query.strip():
            return []
        breadth = max(pool * 2, 30)

        qvec = kb.embed_query(query)                                   # 1. semantic (FAISS)
        scores, ids = kb.index.search(qvec[None, :], min(breadth, len(kb.chunks)))
        sem = {int(i): (r + 1, float(s)) for r, (i, s) in enumerate(zip(ids[0], scores[0])) if i != -1}

        qtok, bm = tokens(query), {}                                   # 2. keyword (BM25)
        if qtok and self._bm25_index() is not None:
            raw = self._bm25.get_scores(qtok)
            top = sorted(((i, s) for i, s in zip(self._bm25_ids, raw) if s > 0), key=lambda x: -x[1])[:breadth]
            bm = {i: r + 1 for r, (i, _) in enumerate(top)}

        w, best = float(np.clip(semantic_weight, 0, 1)), 1 / (rrf_k + 1)   # 3. weighted RRF
        fused = {}
        for i in set(sem) | set(bm):
            s = w / (rrf_k + sem[i][0]) if i in sem else 0.0
            b = (1 - w) / (rrf_k + bm[i]) if i in bm else 0.0
            fused[i] = (s + b) / best
        top_ids = sorted(fused, key=lambda i: -fused[i])[:pool]

        hits = []
        for i in top_ids:
            c = kb.chunks[i]
            cos = sem[i][1] if i in sem else float(qvec @ kb.index.reconstruct(i))
            hits.append(Hit(c, cos, sem.get(i, (None,))[0], bm.get(i), fused[i]))

        if rerank and self.reranker is not None and hits:               # 4. cross-encoder rerank
            probs = relevance(self.reranker, query, [h.chunk.text for h in hits])
            for h, p in zip(hits, probs):
                h.rerank = float(p)
        hits.sort(key=lambda h: -h.score)
        for r, h in enumerate(hits, 1):
            h.rank = r
        return hits
