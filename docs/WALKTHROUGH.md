# Walkthrough

A guided tour for someone who has to explain Perimeter out loud. Read `CLAUDE.md` first
for the map; this document is the narrative behind it.

## What each layer does

**`core/`** is the domain: principals and group resolution (`principal.py`), access
policies with grants and explicit denies (`acl.py`), documents and chunks (`document.py`),
requests and results (`query.py`), the I/O ports as `typing.Protocol`s (`ports.py`), and
typed errors (`errors.py`). It imports nothing but the standard library. There is no
function in `core/` that cannot be called from a unit test with plain values.

**`index/`** is the vector index, NumPy only. `quantize.py` turns float32 vectors into
1-bit codes (for the scan) and int8 codes (for rescoring). `flat.py` stores those codes,
chunk IDs, ACL rows, and quantizer parameters in a directory of raw arrays, memory-mapped
at open, and exposes `scan_rows(query, rows, k)`: score *exactly these rows* and nothing
else. `filtered_search.py` decides which rows those are from the caller's permission set,
using a vectorised CSR mask that a property test proves equal to `AccessPolicy.admits`.

**`adapters/`** implement the ports: Cohere embeddings and rerank over httpx, an
in-memory store, a Postgres store, a deterministic local embedder for air-gapped runs,
a static resolver, and the caching ACL resolver with its TTL and invalidation hook.

**`connectors/`** answer three questions about a source: what exists, what does one
document say, who may read it. Filesystem (ACL sidecars) and Google Drive (acting as the
caller with a request-scoped token). A document whose ACL cannot be read is ingested as
readable by nobody.

**`pipeline/`** composes ports. `ingest.py` chunks, embeds, and writes; `retrieve.py` is
the orchestrator whose step order *is* the security argument.

**`server/`** is the edge: identity and token extraction (`auth.py`), the MCP tool
(`mcp.py`), FastAPI for `/health` and the admin API (`http.py`), tracing with an attribute
whitelist (`telemetry.py`), logging with a redaction filter (`logging.py`), the
composition root (`wiring.py`), and the connector registry.

**`admin/`** is the React console: overview, connectors, index health, and the permission
simulator, all through a client generated from the server's OpenAPI document.

## Why the dependency rule exists

The access-control logic is the product, so it must be testable exhaustively: thousands of
generated policy graphs per run, no database, no network, no NumPy. Putting it in a package
that imports only the standard library makes that a fact rather than a discipline, and
`tests/test_architecture.py` turns the fact into a build gate. The second benefit is
reviewability: an interviewer can read `core/acl.py` in two minutes and know exactly what
"admitted" means, with no framework in the way.

## How a request flows

1. An MCP host calls the `retrieve` tool over StreamableHTTP with `X-Perimeter-Principal`
   and optionally `X-Perimeter-Groups`. Identity comes from headers set by the trusted
   front door, never from tool arguments, because a model can change arguments.
2. `server/auth.py` parses the identity (rejecting the reserved `everyone` principal) and,
   through a pure-ASGI middleware, opens a request scope holding any connector tokens.
3. `pipeline/retrieve.py` asks the `AclResolver` for the caller's permission set. The
   caching resolver serves a fresh entry or consults upstream; on upstream failure it
   returns the empty set. **If the set is empty, retrieval returns empty here**, before
   the query is embedded (INV-4, and no API call spent).
4. The query is embedded (Cohere or the local stand-in) and the index is searched **with
   the permission set** for `k * 4` candidates. `filtered_search.permitted_rows` builds the
   allow mask minus the deny mask over the CSR ACL table and hands only those row indices
   to `scan_rows`, which gathers their binary codes, computes Hamming distance, keeps
   `k * 64` candidates, rescores them with int8 codes against the float query, and returns
   the best. Unpermitted rows are never read (INV-2).
5. The store is asked for those chunk IDs **with the permission set**; it refuses text for
   anything the policy does not admit (INV-1, second check). The orchestrator re-checks
   each returned chunk's policy (third check).
6. The reranker sees only what survived, the top `k` are taken, citations are attached,
   and a `ScopedResult` with `requested_k`, `returned`, and `candidates` goes back. The
   MCP tool serialises it; the span records counts only.
7. The request scope closes and wipes any tokens (INV-3).

## Where each invariant is enforced

| | Code | Test |
|-|------|------|
| INV-1 | `adapters/*_store.py` `get_chunks(ids, permitted)`; `pipeline/retrieve.py` re-check | `tests/invariants/test_inv1_no_unpermitted_text.py` (with a leaky index), `tests/test_acl_leak_property.py` (random graphs, both index variants) |
| INV-2 | `index/filtered_search.py` | `tests/invariants/test_inv2_filter_inside_scan.py` (scan spy + reranker spy), property equality of mask and policy in `tests/index/test_filtered_search.py` |
| INV-3 | `server/auth.py` `request_scope`, `ConnectorToken`; `server/logging.py`; `server/telemetry.py` | `tests/invariants/test_inv3_token_scope.py` |
| INV-4 | `pipeline/retrieve.py` early return; `index/filtered_search.py` empty rows | `tests/invariants/test_inv4_fail_closed.py` |
| INV-5 | `adapters/caching_acl_resolver.py` | `tests/invariants/test_inv5_cache_monotonic.py` (scripted + two Hypothesis models) |

## Trade-offs deliberately accepted

- **Flat scan, not an approximate index.** Linear in the permitted set. Under ~1M chunks
  this is inside budget by a wide margin and it keeps top-k semantics exact over the
  permitted set, which approximate structures with filtering generally do not.
- **Binary + int8 quantization.** recall@10 of 0.976 against exact float32 on the
  benchmark corpus, at 1,174 bytes per chunk. The loss is in near-tie ordering among
  same-cluster neighbours; the binary stage needs a 64x candidate net under selective
  filters (measured in commit 11, not guessed).
- **Append-and-rebuild writes.** A flush rewrites the index directory. Ingestion is a
  batch job; there is no concurrent insert path and `FlatIndex` is not thread-safe.
- **Identity is trusted from headers.** Perimeter is not an auth server (non-goal). It
  must sit behind something that is.
- **Tokens per request.** No unattended re-indexing without a scheduler that supplies a
  credential per run. That is the price of holding nothing.
- **A synthetic benchmark corpus.** Clustered Gaussian vectors, not real embeddings, and
  the local embedder is a hashing stand-in, not a semantic model. The numbers measure
  Perimeter's own path; retrieval quality on real text depends on the real embedder.

## Questions an interviewer would ask

**1. Building a permission mask over 50k rows on every query sounds expensive. Is it?**
It is a vectorised `isin` over the CSR ACL entries plus a segmented any, then
`flatnonzero`. Measured p95 is 5.7 ms end to end with all rows permitted and 3.1 ms with
10% permitted, because the scan cost tracks the permitted set, not the corpus. Caching
the mask per permission set would help the all-rows case and was not needed to meet the
budget; I would add it if the corpus grew.

**2. How do you know the vectorised filter means the same thing as the policy code?**
A Hypothesis test generates random policy lists and permission sets, evaluates
`AccessPolicy.admits` in pure Python, and asserts the CSR mask selects exactly the same
rows, both directions. If the two ever disagree the test names the counterexample.

**3. What does an attacker get from the index file alone?**
Vectors, chunk IDs, and ACL rows. No text (ADR-006). The honest residual risk is embedding
inversion: reconstructing approximate text from vectors is a real research result, and
int8/binary quantization makes it harder but not impossible. The index file should be
treated as sensitive, just not as plaintext.

**4. Your INV-5 says stale cache entries are only ever more restrictive. With explicit
denies, is that actually true?**
Not automatically, and the property tests found that. Admission is monotone in the
permission set only for grant-only policies; with a deny, being *added* to a group can
revoke access, so a cache that has not learned of the addition is less restrictive for
that document. The resolution is that the invalidation hook fires on any membership
change in either direction, and unsignalled changes are bounded by the TTL, which is why
the TTL is documented as a security parameter. Connectors without a deny concept (Drive)
get the stronger guarantee for free. ADR-004 records this.

**5. Trusting identity from headers is a hole. Anyone who can reach port 8000 is anyone.**
Yes. Perimeter is explicitly not an auth server; it must sit behind a gateway or MCP host
that authenticates and strips or overwrites those headers. The mitigations in the box are
DNS-rebinding protection with an allowed-hosts list and refusing the reserved `everyone`
principal. In a real deployment I would bind to localhost or a private network, require
mTLS from the front door, and consider a signed identity assertion instead of plain
headers.

**6. What happens at 10M chunks?**
Latency grows linearly and the budget breaks. The next step is an IVF-style coarse
partition over the same memory-mapped codes, keeping the filter inside the scan of each
probed partition, before ever considering an external service (ADR-002 says what would
make me reverse it). What I would not do is bolt on an ANN library whose filtering is a
post-filter, because that reintroduces the failure Perimeter exists to prevent.

**7. Why not a vector database with metadata filtering? They all have it now.**
Some do it as a pre-filter that preserves k, some as a post-filter that does not, and
the documentation rarely says which; the semantics under a highly selective filter are
the thing you have to test. Beyond that, the constraint was one air-gapped container, and
an external database is a second process, a second failure domain, and a second copy of
the permission logic. The honest cost of my choice is operational: no replication, no
concurrent writers, and a scale ceiling.

**8. recall@10 is 0.976, not 1.0. Where is the loss and how would you close it?**
Two stages lose. The binary Hamming stage can miss true neighbours under a selective
filter; measured recall rose from 0.9465 to 0.988 as the candidate multiplier went from
16x to 64x, so that stage is now near its ceiling. The remaining loss is int8 resolution
among near-tie neighbours; the int8-only ceiling was 0.9985 on the selective case and
0.979 on the full case. Options in order: rescore the final few with a higher-precision
code kept only for a small hot set, use per-dimension ranges from a percentile rather
than min/max, or accept the number, which is above the gate.

**9. `FlatIndex` is not thread-safe and every flush rewrites the whole index. How does
that survive production?**
It survives the deployment it was designed for: one uvicorn worker, batch ingestion,
reads that are pure NumPy over an immutable memory-mapped segment. It would not survive
concurrent ingestion from several admin sessions. The fix is a single writer thread with
a queue and atomic directory swaps, which the on-disk layout already supports; I did not
build it because nothing in the stated scope needed it, and I would rather say so than
pretend.

**10. If someone has the Postgres credentials, what stops them reading everything?**
Nothing in Perimeter. The store's SQL predicate enforces permissions for the application
path, and the Python re-check catches drift, but a direct database connection is a direct
database connection. Postgres row-level security with a per-request role would push the
check into the database itself; it is the natural next hardening step and I would do it
before exposing the database to anything but the Perimeter process.

**Bonus, the one I would ask myself: how much of the benchmark is real?**
The latency, memory, size, and API-call numbers are real measurements of the real code
path on a synthetic corpus. The recall number is real against an exact baseline on that
same synthetic corpus. What is not measured anywhere in this repository is retrieval
quality on real documents with the real embedding model, because there is no API key and
no real corpus in the build. The README says which is which.
