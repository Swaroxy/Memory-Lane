# Memory Lane

A small, local RAG explorer: ask questions about your documents, get answers quoted from the sources
with citations, and watch the retrieved evidence light up in a knowledge graph.

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads two small Hugging Face models (~130 MB). Six sample documents in
`knowledge_base/` are indexed automatically; drop in your own `.md`, `.txt` or `.pdf` files
(or use the sidebar uploader) and press **Sync**.

**Prefer Jupyter?** Open `Memory_Lane.ipynb` and press Run All — the same app as a single self-contained notebook.

## How it works

| Stage | Where | What happens |
|---|---|---|
| Ingestion | `index.py` | Reads md / txt / pdf, parses optional front matter (`last_updated`, `valid_until`, `superseded_by`) |
| Chunking | `index.py` | LangChain `RecursiveCharacterTextSplitter` — sections → paragraphs → sentences, with overlap |
| Embeddings | `app.py` | `HuggingFaceEmbeddings` (Sentence Transformers), L2-normalised |
| Vector DB | `index.py` | FAISS `IndexIDMap2(IndexFlatIP)` — cosine search, per-document add/remove |
| Hybrid search | `retrieve.py` | FAISS semantic + BM25 keyword, merged with weighted reciprocal-rank fusion |
| Reranking | `retrieve.py` | Sentence Transformers `CrossEncoder` re-scores the top 20 candidates |
| Grounded answer | `answer.py` | Verbatim sentences from the evidence, each tagged `[n]` with its source |
| Graph | `graph.py`, `graph.html` | Chunks as nodes, document hubs, similarity links; retrieved evidence glows |

### Hallucination prevention

- **Extractive answers.** The answer is built only from sentences that exist in the retrieved passages, so it cannot state anything the sources don't.
- **Abstention.** If the best passage's cross-encoder relevance is below 10%, or the question's key terms don't appear in the evidence, Memory Lane says it can't answer instead of guessing.
- **Evidence indicator.** *strong / moderate / weak* from relevance, query-term coverage and freshness of the cited sources. Terms missing from the evidence are listed.

### Stale-context detection and re-indexing

Each document gets a status: `fresh`, `stale` (not updated within a year), `expired` (`valid_until` passed),
`superseded` (`superseded_by` set), `modified` (file changed since indexing) or `missing`. Cited documents
that aren't fresh trigger a warning, superseded passages are skipped in favour of their replacement, and
unhealthy documents get a dashed ring in the graph. **Sync** re-embeds only new or changed files and drops
deleted ones; **Rebuild** starts over. Changing the embedding model rebuilds automatically.

Front matter example:

```yaml
---
title: Security Handbook
last_updated: 2025-05-30
valid_until: 2026-06-30
superseded_by: security_handbook_v2.md
---
```

## Tests

```bash
pip install pytest && python -m pytest
```

Tests use tiny fake embedding / cross-encoder classes (`tests/fakes.py`), so they run offline in under a second.
