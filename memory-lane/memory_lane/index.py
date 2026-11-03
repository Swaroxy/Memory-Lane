"""Ingestion, recursive chunking, embeddings, FAISS index, incremental re-indexing, freshness."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

EXTENSIONS = {".md", ".txt", ".pdf"}
SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]   # sections > paragraphs > lines > sentences


@dataclass
class Chunk:
    id: int            # FAISS id
    doc: str           # path relative to the knowledge-base folder
    title: str
    n: int             # position within the document
    text: str
    heading: str | None = None
    page: int | None = None


# ----------------------------------------------------------------------------- ingestion
def parse_front_matter(text: str) -> tuple[dict, str]:
    """Flat ``key: value`` block between two ``---`` lines at the top of a file."""
    m = re.match(r"^---[ \t]*\n(.*?)\n---[ \t]*\n?", text, re.S)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip().lower()] = v.strip().strip("\"'")
    return meta, text[m.end():]


def load_file(path: Path) -> tuple[str, dict, list[tuple[int | None, str]]]:
    """Return (title, metadata, [(page, text), ...])."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        pages = [(i, p.extract_text() or "") for i, p in enumerate(PdfReader(str(path)).pages, 1)]
        return path.stem.replace("_", " ").title(), {}, pages
    meta, body = parse_front_matter(path.read_text(encoding="utf-8", errors="replace"))
    h1 = re.search(r"^#\s+(.+)$", body, re.M)
    title = meta.get("title") or (h1.group(1).strip() if h1 else path.stem.replace("_", " ").title())
    return title, meta, [(None, body)]


def chunk_pages(doc: str, title: str, pages, size: int, overlap: int) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, separators=SEPARATORS, add_start_index=True
    )
    chunks: list[Chunk] = []
    for page, text in pages:
        heads = [(m.start(), m.group(2)) for m in re.finditer(r"^(#{1,4})\s+(.+)$", text, re.M)]
        for d in splitter.create_documents([text]):
            body = d.page_content.strip()
            if len(body) < 20:
                continue
            start, heading = d.metadata["start_index"], None
            for pos, h in heads:
                if pos > start + 2:
                    break
                heading = h
            chunks.append(Chunk(0, doc, title, len(chunks), body, heading, page))
    return chunks


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _date(value: str | None) -> datetime | None:
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d %B %Y", "%B %d, %Y"):
        try:
            return datetime.strptime((value or "").strip(), fmt)
        except ValueError:
            pass
    return None


# ----------------------------------------------------------------------------- knowledge base
class KnowledgeBase:
    """FAISS-backed index of a folder of documents.

    ``embeddings`` is any LangChain ``Embeddings`` object (e.g. ``HuggingFaceEmbeddings``).
    Each document's vectors are stored under their own ids, so re-indexing a changed file
    only touches that file (``IndexIDMap2`` supports removal).
    """

    def __init__(self, kb_dir, index_dir, embeddings, model_name: str,
                 chunk_size: int = 700, chunk_overlap: int = 120, stale_days: int = 365):
        self.kb_dir, self.index_dir = Path(kb_dir), Path(index_dir)
        self.emb, self.stale_days = embeddings, stale_days
        self.signature = {"model": model_name, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
        self.chunks: dict[int, Chunk] = {}
        self.docs: dict[str, dict] = {}
        self.index: faiss.Index | None = None
        self.next_id, self.version = 1, 0
        self._hash_cache: dict[str, tuple] = {}
        self._matrix: tuple | None = None
        self._load()

    # ---- embeddings
    def _norm(self, v) -> np.ndarray:
        v = np.asarray(v, dtype="float32")
        return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9, None)

    def embed_query(self, text: str) -> np.ndarray:
        return self._norm(self.emb.embed_query(text))

    # ---- persistence
    def _load(self) -> None:
        f_idx, f_json = self.index_dir / "index.faiss", self.index_dir / "store.json"
        if not (f_idx.exists() and f_json.exists()):
            return
        data = json.loads(f_json.read_text(encoding="utf-8"))
        if data["signature"] != self.signature:      # different model / chunking -> rebuild from scratch
            return
        self.index = faiss.read_index(str(f_idx))
        self.chunks = {c["id"]: Chunk(**c) for c in data["chunks"]}
        self.docs, self.next_id = data["docs"], data["next_id"]

    def save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_dir / "index.faiss"))
        (self.index_dir / "store.json").write_text(json.dumps({
            "signature": self.signature, "next_id": self.next_id, "docs": self.docs,
            "chunks": [asdict(c) for c in self.chunks.values()],
        }), encoding="utf-8")

    # ---- indexing
    def _remove(self, doc: str) -> None:
        ids = self.docs.pop(doc)["chunk_ids"]
        if ids and self.index is not None:
            self.index.remove_ids(faiss.IDSelectorBatch(np.asarray(ids, dtype="int64")))
        for i in ids:
            self.chunks.pop(i, None)

    def _add(self, doc: str, title: str, meta: dict, digest: str, chunks: list[Chunk]) -> None:
        ids = list(range(self.next_id, self.next_id + len(chunks)))
        self.next_id += len(chunks)
        if chunks:
            vecs = self._norm(self.emb.embed_documents([c.text for c in chunks]))
            if self.index is None:
                self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(vecs.shape[1]))
            self.index.add_with_ids(vecs, np.asarray(ids, dtype="int64"))
        for c, i in zip(chunks, ids):
            c.id = i
            self.chunks[i] = c
        self.docs[doc] = {"title": title, "meta": meta, "hash": digest, "chunk_ids": ids,
                          "indexed_at": datetime.now().timestamp()}

    def sync(self, rebuild: bool = False) -> dict:
        """Index new files, re-index changed ones, drop deleted ones. ``rebuild`` starts over."""
        if rebuild:
            self.chunks, self.docs, self.index, self.next_id = {}, {}, None, 1
        files = {}
        for p in sorted(self.kb_dir.rglob("*")):
            rel = p.relative_to(self.kb_dir)
            if p.is_file() and p.suffix.lower() in EXTENSIONS and not any(s.startswith(".") for s in rel.parts):
                files[rel.as_posix()] = p
        report = {"added": [], "updated": [], "removed": []}
        for doc in [d for d in self.docs if d not in files]:
            self._remove(doc)
            report["removed"].append(doc)
        for doc, path in files.items():
            digest, known = _sha(path), self.docs.get(doc)
            if known and known["hash"] == digest:
                continue
            title, meta, pages = load_file(path)
            chunks = chunk_pages(doc, title, pages, self.signature["chunk_size"], self.signature["chunk_overlap"])
            if known:
                self._remove(doc)
            self._add(doc, title, meta, digest, chunks)
            report["updated" if known else "added"].append(doc)
        if any(report.values()) or rebuild:
            self.version += 1
            self.save()
        return report

    def matrix(self) -> tuple[list[int], np.ndarray]:
        """All (chunk id, vector) pairs, cached per index version."""
        if self._matrix is None or self._matrix[0] != self.version:
            ids = sorted(self.chunks)
            vecs = np.vstack([self.index.reconstruct(i) for i in ids]) if ids else np.zeros((0, 1), "float32")
            self._matrix = (self.version, ids, vecs)
        return self._matrix[1], self._matrix[2]

    # ---- freshness / stale-context detection
    def _current_hash(self, path: Path) -> str:
        st = path.stat()
        key = (st.st_mtime, st.st_size)
        cached = self._hash_cache.get(str(path))
        if not cached or cached[0] != key:
            cached = (key, _sha(path))
            self._hash_cache[str(path)] = cached
        return cached[1]

    def freshness(self, now: datetime | None = None) -> dict[str, tuple[str, str]]:
        """doc -> (status, reason). Status: fresh | stale | expired | superseded | modified | missing."""
        now, out = now or datetime.now(), {}
        for doc, rec in self.docs.items():
            path, meta = self.kb_dir / doc, rec["meta"]
            if not path.exists():
                out[doc] = ("missing", "The source file no longer exists.")
            elif self._current_hash(path) != rec["hash"]:
                out[doc] = ("modified", "The file changed after it was indexed — re-index to refresh.")
            elif meta.get("superseded_by"):
                out[doc] = ("superseded", f"Superseded by {meta['superseded_by']}.")
            elif (end := _date(meta.get("valid_until"))) and end < now:
                out[doc] = ("expired", f"Validity ended on {end:%Y-%m-%d}.")
            else:
                updated = _date(meta.get("last_updated")) or datetime.fromtimestamp(path.stat().st_mtime)
                age = (now - updated).days
                if age > self.stale_days:
                    out[doc] = ("stale", f"Last updated {updated:%Y-%m-%d} ({age} days ago).")
                else:
                    out[doc] = ("fresh", f"Last updated {updated:%Y-%m-%d}.")
        return out
