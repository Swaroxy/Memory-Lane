import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from fakes import FakeCrossEncoder, FakeEmbeddings  # noqa: E402
from memory_lane import KnowledgeBase, Retriever, answer, build_graph, render_graph  # noqa: E402

SAMPLES = Path(__file__).parent.parent / "knowledge_base"


@pytest.fixture()
def env(tmp_path):
    kb_dir = tmp_path / "kb"
    shutil.copytree(SAMPLES, kb_dir)
    kb = KnowledgeBase(kb_dir, tmp_path / "idx", FakeEmbeddings(), "fake")
    kb.sync()
    return kb, Retriever(kb, FakeCrossEncoder()), kb_dir, tmp_path


def ask(kb, retriever, q, **kw):
    hits = retriever.search(q)
    return hits, answer(q, hits, kb, retriever.reranker, **kw)


def test_ingest_chunk_and_index(env):
    kb, *_ = env
    assert len(kb.docs) == 6 and len(kb.chunks) > 6
    assert kb.index.ntotal == len(kb.chunks)
    assert all(c.heading for c in kb.chunks.values() if c.n > 0)        # section headings are tracked


def test_hybrid_retrieval_exposes_both_signals(env):
    kb, retriever, *_ = env
    hits = retriever.search("How quickly must a SEV1 page be acknowledged?")
    assert hits[0].chunk.title == "Incident Response Runbook"
    assert hits[0].rerank is not None and hits[0].fused > 0
    assert any(h.sem_rank for h in hits) and any(h.bm25_rank for h in hits)
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


def test_grounded_answer_is_verbatim_and_cited(env):
    kb, retriever, *_ = env
    _, res = ask(kb, retriever, "How many days per week can I work remotely?")
    assert not res.abstained and "3 days per week" in res.text and "[1]" in res.text
    source = res.citations[0].chunk.text.replace("*", "")
    sentence = res.text.split(" [")[0]
    assert sentence in source                                            # nothing invented
    assert res.level in {"strong", "moderate"}


def test_abstains_on_unsupported_question(env):
    kb, retriever, *_ = env
    _, res = ask(kb, retriever, "What is the airspeed velocity of an unladen swallow?")
    assert res.abstained and res.level == "none" and res.text == ""


def test_superseded_document_is_skipped(env):
    kb, retriever, *_ = env
    _, res = ask(kb, retriever, "How many days per week can I work remotely?")
    assert all("2022" not in c.chunk.title for c in res.citations)
    assert any("superseded" in w for w in res.warnings)


def test_expired_document_is_flagged(env):
    kb, retriever, *_ = env
    assert kb.freshness()["security_handbook.md"][0] == "expired"
    _, res = ask(kb, retriever, "How long must passwords be?")
    assert "14 characters" in res.text
    assert any("expired" in w for w in res.warnings) and res.parts["freshness"] < 1


def test_modified_file_is_detected_then_reindexed_incrementally(env):
    kb, retriever, kb_dir, tmp = env
    path = kb_dir / "engineering_onboarding.md"
    path.write_text(path.read_text().replace("two approvals", "three approvals"), encoding="utf-8")
    assert kb.freshness()["engineering_onboarding.md"][0] == "modified"

    before = len(kb.chunks)
    report = kb.sync()
    assert report == {"added": [], "updated": ["engineering_onboarding.md"], "removed": []}
    assert kb.freshness()["engineering_onboarding.md"][0] == "fresh"
    assert kb.index.ntotal == len(kb.chunks) and abs(len(kb.chunks) - before) <= 1
    _, res = ask(kb, retriever, "How many approvals does a code review need?")
    assert "three approvals" in res.text

    path.unlink()
    assert kb.freshness()["engineering_onboarding.md"][0] == "missing"
    assert kb.sync()["removed"] == ["engineering_onboarding.md"]
    assert kb.index.ntotal == len(kb.chunks)


def test_index_persists_and_signature_change_rebuilds(env):
    kb, _, kb_dir, tmp = env
    again = KnowledgeBase(kb_dir, tmp / "idx", FakeEmbeddings(), "fake")
    assert len(again.chunks) == len(kb.chunks) and again.sync() == {"added": [], "updated": [], "removed": []}
    other = KnowledgeBase(kb_dir, tmp / "idx", FakeEmbeddings(), "fake", chunk_size=300)
    assert not other.chunks and len(other.sync()["added"]) == 6


def test_graph_highlights_retrieved_evidence(env):
    kb, retriever, *_ = env
    hits, res = ask(kb, retriever, "How many days per week can I work remotely?")
    graph = build_graph(kb, hits, {h.chunk.id: n for n, h in enumerate(res.citations, 1)})
    assert len(graph["nodes"]) == len(kb.chunks) + len(kb.docs)
    assert {e["kind"] for e in graph["edges"]} >= {"doc", "sim"}
    evidence = graph["highlight"]["evidence"]
    assert len(evidence) == 5
    cited = {e["i"]: e["cite"] for e in evidence if e["cite"]}
    assert 1 in cited.values()                                           # the answer's [1] is highlighted
    old = next(i for i, n in enumerate(graph["nodes"]) if "2022" in n["title"] and n["kind"] == "chunk")
    assert old in {e["i"] for e in evidence} and old not in cited        # retrieved, but not used as evidence
    html = render_graph(graph)
    assert "__GRAPH__" not in html and "__HEIGHT__" not in html and "<script>" in html
