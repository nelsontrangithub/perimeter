# ADR-001: ACL filtering inside the index scan, not as a post-filter

Status: Accepted

## Context

A retrieval call asks for the top `k` chunks for a query on behalf of a caller. The
corpus contains documents the caller may not see. The two common designs are:

1. Search the whole corpus, take the top `k`, then drop what the caller may not see.
2. Search the whole corpus, take the top `k`, generate an answer, then redact the answer.

Design 2 is not a security boundary at all: the model has already read the forbidden
text, and summarization leaks it. Design 1 is a real boundary but has two defects. It
breaks top-k: the caller asked for 10 and gets 2, and worse, gets no signal that 8 were
removed. And it loads unauthorized rows into the scoring path, so a bug anywhere after
the scan (a logging line, a debug dump, an off-by-one in the filter) exposes them.

## Decision

The caller's permitted set is resolved before the scan and applied *inside* it. The
index gathers only the rows the allow-list admits and computes distances for those rows
only. The top-k is therefore the top-k of the permitted set, and unpermitted rows are
never scored, never become candidates, never reach the reranker, and never have text
loaded. This is INV-2. The pipeline and the document store each re-check the policy on
the way out (INV-1), so the scan filter is one of three independent checks rather than
the only one.

An empty permitted set short-circuits to an empty result before the index is touched
(INV-4). There is no code path from "no permissions" to "unfiltered scan".

## Consequences

- `k` means what the caller thinks it means.
- The per-request cost includes building a boolean mask over the corpus from the caller's
  permission set. With a CSR-style ACL table in NumPy this is a vectorized `isin` plus a
  segmented reduction, cheap relative to the distance computation at our scale.
- Selectivity is free: a caller who can see 1% of the corpus scans 1% of it.
- The index must store ACL rows next to the vectors. That couples index rebuilds to ACL
  changes for grants (a newly granted document becomes visible on the next ACL-row
  update), which is acceptable because revocation is handled at the resolver (ADR-004)
  and at the store (INV-1), not only at the index.

## What would make us reverse it

If the corpus grew to the point where a flat scan over the permitted set could not meet
the latency budget, we would need an approximate structure (HNSW, IVF) and would have to
decide whether filtered traversal of that structure preserves top-k semantics. Most do
not without oversampling, which reintroduces exactly the "ask for 10, get 2" failure.
That is an argument for ADR-002's scale ceiling rather than for abandoning this decision.
