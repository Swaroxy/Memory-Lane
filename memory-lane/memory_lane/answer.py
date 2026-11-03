"""Source-grounded answers with citations, evidence indicators and hallucination prevention.

The answer is **extractive**: it is assembled from verbatim sentences of the retrieved passages,
so it cannot contain a claim that is not in a source. On top of that:

* abstain when the best passage is not relevant enough or the question's key terms are absent,
* skip passages from documents that were superseded by a newer indexed document,
* warn when a cited document is stale / expired / modified / missing,
* report an evidence level (strong / moderate / weak / none) with its components.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .index import KnowledgeBase
from .retrieve import Hit, relevance, tokens


@dataclass
class Answer:
    text: str = ""
    citations: list[Hit] = field(default_factory=list)       # citations[n-1] is cited as [n]
    abstained: bool = False
    level: str = "none"                                       # strong | moderate | weak | none
    confidence: float = 0.0
    parts: dict = field(default_factory=dict)                 # relevance / coverage / freshness
    warnings: list[str] = field(default_factory=list)
    missing_terms: list[str] = field(default_factory=list)


def sentences(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "---")):
            continue
        line = re.sub(r"^([-*+]|\d+[.)])\s+", "", line)
        line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)
        line = re.sub(r"[*_`]", "", line)
        out += [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", line) if len(s.strip()) > 25]
    return out


def _level(confidence: float) -> str:
    return "strong" if confidence >= 0.6 else "moderate" if confidence >= 0.38 else "weak" if confidence >= 0.2 else "none"


def answer(query: str, hits: list[Hit], kb: KnowledgeBase, reranker=None, top_k: int = 5,
           abstain_below: float = 0.10, max_sentences: int = 3) -> Answer:
    fresh = kb.freshness()
    top, warnings = hits[:top_k], []

    # skip passages whose document was superseded by another document that is in the index
    usable = []
    for h in top:
        newer = kb.docs.get(h.chunk.doc, {}).get("meta", {}).get("superseded_by")
        if newer and any(newer == d or newer == d.rsplit("/", 1)[-1] for d in kb.docs):
            warnings.append(f"Skipped “{h.chunk.title}” — superseded by {newer}.")
        else:
            usable.append(h)
    usable = usable or top
    if not usable:
        return Answer(abstained=True, warnings=["Nothing in the index matched the question."])

    # evidence assessment
    best = usable[0].score
    q_terms = set(tokens(query))
    evidence_terms = set(tokens(" ".join(f"{h.chunk.title} {h.chunk.heading or ''} {h.chunk.text}" for h in usable)))
    coverage = len(q_terms & evidence_terms) / len(q_terms) if q_terms else 1.0
    missing = sorted(q_terms - evidence_terms)
    if best < abstain_below or (coverage < 0.34 and usable[0].semantic < 0.55):
        why = (f"The best passage scored {best:.0%} relevance (threshold {abstain_below:.0%})."
               if best < abstain_below else "None of the question's key terms appear in the retrieved passages.")
        return Answer(abstained=True, parts={"relevance": best, "coverage": coverage}, missing_terms=missing,
                      warnings=warnings + [why])

    # choose the best verbatim sentences
    cands = [(h, i, s) for h in usable for i, s in enumerate(sentences(h.chunk.text))]
    if not cands:
        return Answer(abstained=True, warnings=warnings + ["No sentence in the retrieved passages answers the question."])
    if reranker is not None:
        sc = relevance(reranker, query, [s for _, _, s in cands])
    else:
        sc = np.array([len(q_terms & set(tokens(s))) / max(1, len(q_terms)) for _, _, s in cands])
    order = [int(k) for k in np.argsort(-sc)]
    floor = max(0.15, 0.6 * sc[order[0]])
    chosen = [k for k in order if sc[k] >= floor][:max_sentences]
    chosen.sort(key=lambda k: (cands[k][0].rank, cands[k][1]))

    citations: list[Hit] = []
    pieces: list[tuple[int, str]] = []
    for k in chosen:
        h = cands[k][0]
        if h not in citations:
            citations.append(h)
        n = citations.index(h) + 1
        if pieces and pieces[-1][0] == n:
            pieces[-1] = (n, pieces[-1][1] + " " + cands[k][2])
        else:
            pieces.append((n, cands[k][2]))
    text = " ".join(f"{t} [{n}]" for n, t in pieces)

    # freshness of what was actually cited
    ok = [fresh.get(h.chunk.doc, ("fresh", ""))[0] == "fresh" for h in citations]
    freshness = float(np.mean([1.0 if x else 0.3 for x in ok]))
    for n, h in enumerate(citations, 1):
        status, reason = fresh.get(h.chunk.doc, ("fresh", ""))
        if status != "fresh":
            warnings.append(f"[{n}] “{h.chunk.title}” is {status}: {reason}")
    if missing:
        warnings.append("Not found in the evidence: " + ", ".join(f"“{t}”" for t in missing[:5]) + ".")

    confidence = (0.6 * best + 0.4 * coverage) * (0.7 + 0.3 * freshness)
    return Answer(text, citations, False, _level(confidence), float(np.clip(confidence, 0, 1)),
                  {"relevance": best, "coverage": coverage, "freshness": freshness}, warnings, missing)
