"""Memory Lane — run with:  streamlit run app.py"""
from pathlib import Path

import pandas as pd
import streamlit as st

from memory_lane import KnowledgeBase, Retriever, answer, build_graph, render_graph

ROOT = Path(__file__).parent
KB_DIR, INDEX_DIR = ROOT / "knowledge_base", ROOT / ".memory_lane"
EMBEDDING_MODELS = ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LEVEL_ICON = {"strong": "🟢", "moderate": "🟡", "weak": "🟠", "none": "🔴"}
STATUS_ICON = {"fresh": "🟢", "stale": "🟠", "expired": "🔴", "superseded": "🟣", "modified": "🔵", "missing": "⚫"}

st.set_page_config(page_title="Memory Lane", page_icon="🧠", layout="wide")
st.markdown("""<style>
.block-container{padding-top:2rem;max-width:1200px}
h1{background:linear-gradient(90deg,#22d3ee,#a78bfa,#f472b6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:700}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stMarkdownContainer"]){border-color:rgba(34,211,238,.35);box-shadow:0 0 28px rgba(34,211,238,.08)}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading Hugging Face models (first run downloads them)…")
def load_models(embedding_model: str):
    from langchain_huggingface import HuggingFaceEmbeddings
    from sentence_transformers import CrossEncoder

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model, encode_kwargs={"normalize_embeddings": True})
    return embeddings, CrossEncoder(RERANKER)


@st.cache_resource
def load_kb(embedding_model: str) -> tuple[KnowledgeBase, Retriever]:
    embeddings, reranker = load_models(embedding_model)
    kb = KnowledgeBase(KB_DIR, INDEX_DIR, embeddings, embedding_model)
    return kb, Retriever(kb, reranker)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Knowledge base")
    model = st.selectbox("Embedding model", EMBEDDING_MODELS)
    kb, retriever = load_kb(model)

    uploads = st.file_uploader("Add documents", type=["md", "txt", "pdf"], accept_multiple_files=True)
    if uploads:
        KB_DIR.mkdir(exist_ok=True)
        for f in uploads:
            (KB_DIR / Path(f.name).name).write_bytes(f.getvalue())

    c1, c2 = st.columns(2)
    do_sync = c1.button("Sync", help="Index new files, re-index changed ones, drop deleted ones", width="stretch")
    do_rebuild = c2.button("Rebuild", help="Re-embed everything from scratch", width="stretch")
    if do_sync or do_rebuild or (not kb.docs and any(KB_DIR.glob("*"))):
        with st.spinner("Indexing…"):
            report = kb.sync(rebuild=do_rebuild)
        if any(report.values()):
            st.success(" · ".join(f"{len(v)} {k}" for k, v in report.items() if v))
        else:
            st.info("Index is up to date.")

    st.caption(f"{len(kb.docs)} documents · {len(kb.chunks)} chunks")
    with st.expander("Retrieval settings"):
        top_k = st.slider("Evidence chunks", 1, 10, 5)
        semantic_weight = st.slider("Semantic ⟷ keyword", 0.0, 1.0, 0.6, help="1.0 = vectors only, 0.0 = BM25 only")
        use_rerank = st.toggle("Cross-encoder reranking", value=True)

# ------------------------------------------------------------------ main
st.title("Memory Lane")
st.caption("Ask your documents. Every answer is quoted from a source, cited, and checked for freshness.")

with st.form("ask", border=False):
    query = st.text_input("Question", placeholder="e.g. How many days a week can I work remotely?", label_visibility="collapsed")
    asked = st.form_submit_button("Ask", type="primary")

if asked and query.strip():
    if not kb.chunks:
        st.warning("The index is empty — add documents and press Sync.")
    else:
        hits = retriever.search(query, semantic_weight=semantic_weight, rerank=use_rerank)
        result = answer(query, hits, kb, retriever.reranker if use_rerank else None, top_k=top_k)
        st.session_state["last"] = {"query": query, "hits": hits, "answer": result, "top_k": top_k}

last = st.session_state.get("last")
if last:
    res, hits = last["answer"], last["hits"]
    if res.abstained:
        st.warning("I couldn't find enough evidence in your documents to answer that reliably.")
    else:
        st.container(border=True).markdown(res.text)
        cols = st.columns(5)
        for col, (label, value) in zip(cols, [
            ("Evidence", f"{LEVEL_ICON[res.level]} {res.level}"), ("Confidence", f"{res.confidence:.0%}"),
            ("Relevance", f"{res.parts['relevance']:.0%}"), ("Term coverage", f"{res.parts['coverage']:.0%}"),
            ("Freshness", f"{res.parts['freshness']:.0%}")]):
            col.metric(label, value)
    for w in res.warnings:
        st.warning(w, icon="⚠️")
    for n, h in enumerate(res.citations, 1):
        where = f"{h.chunk.title}" + (f" › {h.chunk.heading}" if h.chunk.heading else "") + (f" · p.{h.chunk.page}" if h.chunk.page else "")
        with st.expander(f"[{n}] {where}"):
            st.write(h.chunk.text)
            st.caption(f"{h.chunk.doc} · relevance {h.score:.0%} · semantic {h.semantic:.2f}")

graph_tab, chunks_tab, docs_tab = st.tabs(["Knowledge graph", "Retrieved chunks", "Documents"])
with graph_tab:
    cited = {h.chunk.id: n for n, h in enumerate(last["answer"].citations, 1)} if last else None
    graph = build_graph(kb, last["hits"] if last else None, cited, last["top_k"] if last else 5)
    st.iframe(render_graph(graph), height=640)
    st.caption("Drag nodes · scroll to zoom · hover for text. Dashed rings mark documents that need attention.")
with chunks_tab:
    if last and last["hits"]:
        st.dataframe(pd.DataFrame([{
            "rank": h.rank, "document": h.chunk.title, "section": h.chunk.heading, "rerank": h.rerank,
            "fused": h.fused, "semantic": h.semantic, "semantic rank": h.sem_rank, "keyword rank": h.bm25_rank,
            "used": h.chunk.id in {c.chunk.id for c in last["answer"].citations},
        } for h in last["hits"]]), hide_index=True, width="stretch")
    else:
        st.info("Ask a question to see how each chunk was scored.")
with docs_tab:
    fresh = kb.freshness()
    st.dataframe(pd.DataFrame([{
        "document": rec["title"], "file": doc, "chunks": len(rec["chunk_ids"]),
        "status": f"{STATUS_ICON[fresh[doc][0]]} {fresh[doc][0]}", "detail": fresh[doc][1],
    } for doc, rec in kb.docs.items()]), hide_index=True, width="stretch")
    if any(s != "fresh" for s, _ in fresh.values()):
        st.caption("Press **Sync** after editing a file to re-index just that document.")
