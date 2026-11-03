"""Knowledge graph: chunks as nodes around their document hub, linked by embedding similarity."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .index import KnowledgeBase
from .retrieve import Hit

PALETTE = ["#22d3ee", "#a78bfa", "#f472b6", "#34d399", "#fbbf24", "#60a5fa", "#fb7185", "#2dd4bf"]
TEMPLATE = Path(__file__).with_name("graph.html")


def build_graph(kb: KnowledgeBase, hits: list[Hit] | None = None, cited: dict[int, int] | None = None,
                top_k: int = 5, neighbors: int = 3, threshold: float = 0.45) -> dict:
    """Nodes + edges + the retrieval highlight, ready for the canvas renderer."""
    ids, V = kb.matrix()
    if not ids:
        return {"nodes": [], "edges": [], "docs": [], "highlight": None}
    pos = {cid: j for j, cid in enumerate(ids)}
    fresh = kb.freshness()

    # seed the layout with a 2-D PCA of the embeddings; the browser force simulation refines it
    X = V - V.mean(axis=0)
    xy = X @ np.linalg.svd(X, full_matrices=False)[2][:2].T if len(V) > 2 else np.random.rand(len(V), 2)
    xy = xy / (np.abs(xy).max() or 1)

    doc_ids = sorted({kb.chunks[i].doc for i in ids})
    docs = [{"title": kb.docs[d]["title"], "color": PALETTE[k % len(PALETTE)], "status": fresh[d][0],
             "reason": fresh[d][1]} for k, d in enumerate(doc_ids)]
    doc_no = {d: k for k, d in enumerate(doc_ids)}

    nodes = []
    for j, cid in enumerate(ids):
        c = kb.chunks[cid]
        nodes.append({"kind": "chunk", "doc": doc_no[c.doc], "title": c.title, "n": c.n + 1,
                      "snip": " ".join(c.text.split())[:200], "status": fresh[c.doc][0],
                      "x": round(float(xy[j, 0]), 4), "y": round(float(xy[j, 1]), 4)})
    hub = {}
    for d in doc_ids:
        members = [pos[i] for i in ids if kb.chunks[i].doc == d]
        hub[d] = len(nodes)
        nodes.append({"kind": "doc", "doc": doc_no[d], "title": kb.docs[d]["title"], "status": fresh[d][0],
                      "snip": f"{len(members)} chunks · {fresh[d][0]}",
                      "x": round(float(xy[members, 0].mean()), 4), "y": round(float(xy[members, 1].mean()), 4)})

    edges, seen = [], set()

    def link(a: int, b: int, kind: str, w: float = 1.0) -> None:
        key = (min(a, b), max(a, b), kind)
        if a != b and key not in seen:
            seen.add(key)
            edges.append({"s": a, "t": b, "kind": kind, "w": round(w, 3)})

    for j, cid in enumerate(ids):
        link(j, hub[kb.chunks[cid].doc], "doc")
    if len(ids) > 1:
        S = V @ V.T
        np.fill_diagonal(S, -1)
        k = min(neighbors, len(ids) - 1)
        for a in range(len(ids)):
            for b in np.argpartition(-S[a], k - 1)[:k]:
                if S[a, b] >= threshold:
                    link(a, int(b), "sim", float(S[a, b]))

    highlight = None
    if hits:
        shown = hits[:top_k]
        highlight = {
            "evidence": [{"i": pos[h.chunk.id], "rank": h.rank, "score": round(h.score, 3),
                          "cite": (cited or {}).get(h.chunk.id)} for h in shown],
            "considered": [pos[h.chunk.id] for h in hits[top_k:]],
        }
    return {"nodes": nodes, "edges": edges, "docs": docs, "highlight": highlight}


def render_graph(graph: dict, height: int = 620) -> str:
    data = json.dumps(graph, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return TEMPLATE.read_text(encoding="utf-8").replace("__GRAPH__", data).replace("__HEIGHT__", str(height))
