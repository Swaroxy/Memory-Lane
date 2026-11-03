"""Tiny deterministic stand-ins for the Hugging Face models, so tests need no downloads."""
import zlib

import numpy as np

from memory_lane.retrieve import tokens


class FakeEmbeddings:
    """Hashed bag-of-words (+ character trigrams) vectors with the LangChain Embeddings interface."""

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(256)
        for t in tokens(text):
            for f in [t] + [f"^{t}$"[i:i + 3] for i in range(len(t))]:
                h = zlib.crc32(f.encode())
                v[h % 256] += 1 if (h >> 16) & 1 else -1
        return (v / (np.linalg.norm(v) or 1)).tolist()

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeCrossEncoder:
    """Query-term coverage as a 0..1 'probability' (stands in for sentence_transformers.CrossEncoder)."""

    def __init__(self, *args, **kwargs):
        pass

    def predict(self, pairs, **kwargs):
        out = []
        for q, t in pairs:
            qt = set(tokens(q))
            out.append(len(qt & set(tokens(t))) / max(1, len(qt)) ** 1.0 * 0.95)
        return np.asarray(out)
