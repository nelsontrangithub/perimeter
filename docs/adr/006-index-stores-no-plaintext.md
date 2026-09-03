# ADR-006: The index stores no plaintext

Status: Accepted

## Context

Many vector stores keep the chunk text next to the vector for convenience: one lookup
returns everything. That makes the index file a complete, readable copy of the corpus,
and the index is the artifact most likely to be copied around (to a laptop for debugging,
to a bucket for backup, to another environment for a benchmark).

## Decision

The index stores vectors (binary and int8 codes), chunk IDs, and ACL rows. Nothing else.
Chunk text lives in the document store (`memory_store.py`, `postgres_store.py`) and is
fetched by ID *through the same permission check* the index applied:
`DocumentStore.get_chunks(ids, permission_set)` refuses to return text for any chunk whose
policy does not admit the caller. Compromising the index file alone yields nothing
readable. Compromising the store requires passing its permission check. The two halves
must be compromised together, and even then the pipeline's own re-check (INV-1) is a third
gate.

Embeddings themselves are treated as sensitive but not as plaintext: inversion attacks on
quantized embeddings exist in the literature and are a known residual risk, recorded in
`docs/WALKTHROUGH.md`.

## Consequences

- Retrieval is two lookups, not one: index for IDs, store for text. At our scale the
  second lookup is a dictionary hit or a single `WHERE id = ANY(...)` query.
- Index files can be handled with less ceremony than the document store, which is useful
  operationally.
- The index cannot answer "show me what this chunk says" on its own, which is a feature.

## What would make us reverse it

Nothing product-level. A latency budget that could not afford the second lookup would be
addressed by caching permitted text in the store adapter, still behind the same check,
before ever writing text into the index file.
