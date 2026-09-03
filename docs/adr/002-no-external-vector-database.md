# ADR-002: No external vector database

Status: Accepted

## Context

Perimeter must deploy air-gapped as a single container. An external vector database is a
second process, a second failure domain, a second thing to secure, and a second copy of
the permission logic (or, more often, no permission logic and a leaky post-filter).
Managed vector stores also require a network egress that air-gapped deployments do not
have.

## Decision

The index is a flat, quantized array memory-mapped from a file on local disk. Per vector
we store a 1-bit binary code (128 bytes at 1024 dimensions) for the scan and an int8 code
(1,024 bytes) for rescoring. The scan computes Hamming distance over the permitted rows'
binary codes, takes an oversampled candidate set, and rescores those candidates with the
int8 codes against the float32 query. No float32 copy of the corpus is kept on disk. The
embedding dimension is fixed at 1024 (`embed-v4.0` with `output_dimension=1024`), which
halves storage relative to the model's 1536 default at a recall cost the benchmark gate
measures.

The design target is corpora under roughly one million chunks. Under that ceiling a
vectorized NumPy scan over memory-mapped codes is fast enough that an approximate index
would buy latency we do not need at the cost of top-k semantics we do (ADR-001).

## Consequences

- Single container, single process, no network dependency at query time other than the
  Cohere API (and none at all when the local embedder is configured).
- The OS page cache is the memory strategy. Peak RSS is bounded by the touched pages of
  the code arrays plus the Python runtime. This is a build gate.
- Index writes are append-and-rebuild, not concurrent inserts. Ingestion is a batch job.
- The scale ceiling is real and stated. Beyond it, latency degrades linearly.

## What would make us reverse it

A requirement to serve more than a few million chunks from one process, or a p95 latency
budget the flat scan cannot meet at the deployed corpus size. The first move would be an
IVF-style coarse partition over the same memory-mapped codes, keeping the filter inside
the scan of each probed partition, before considering an external service.
