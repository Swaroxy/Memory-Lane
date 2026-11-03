---
title: RAG Primer
last_updated: 2026-10-01
---

# Retrieval-Augmented Generation Primer

Retrieval-augmented generation, or RAG, answers questions by first retrieving relevant passages from a knowledge base and then composing an answer grounded in those passages.

## Chunking

Documents are split into chunks so that each passage is small enough to embed and to read. Recursive chunking tries large separators first, such as section headings and paragraphs, and falls back to sentences and words only when a piece is still too long. A small overlap between neighbouring chunks keeps sentences that straddle a boundary retrievable.

## Embeddings and vector search

An embedding model maps text to a vector so that passages with similar meaning end up close together. With normalised vectors, cosine similarity equals the inner product. FAISS is a library for fast similarity search over such vectors; a flat inner-product index gives exact results and is ideal for collections up to a few hundred thousand chunks.

## Keyword search with BM25

BM25 scores passages by how often the query terms appear, weighted by how rare each term is across the collection. It is excellent at exact names, identifiers and numbers, where embeddings can be vague.

## Hybrid retrieval and reciprocal rank fusion

Dense and keyword search fail in different ways, so strong systems run both. Reciprocal rank fusion merges the two ranked lists without needing comparable scores: each passage earns 1 divided by the sum of a constant k and its rank in each list, and the constant k is usually 60.

## Reranking

A cross-encoder reads the query and a passage together and outputs a relevance score. It is far more accurate than comparing precomputed vectors but too slow to run on everything, so it is applied only to the top candidates from hybrid retrieval.

## Preventing hallucinations

Ground every claim in a cited passage. Abstain when the best evidence is weak. Verify generated sentences against the sources and remove those that are unsupported. Flag evidence that is stale, expired or superseded so users do not trust outdated answers.
