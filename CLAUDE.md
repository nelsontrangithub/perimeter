# Perimeter

Read this first. It is the map for an engineer joining the project on day one, and it is
kept accurate: any commit that changes a decision recorded here updates this file in the
same commit.

## Thesis

Most RAG systems retrieve everything the index can find and then hope the model behaves,
or they filter results after generation. Both leak. Post-generation filtering leaks through
summarization (the model has already read the forbidden text), and post-retrieval filtering
silently breaks top-k (you ask for 10, the index returns 10, filtering leaves you with 2).
Perimeter applies the caller's permission set *inside* the index scan: unauthorized chunks
are never scored, never become candidates, never reach the reranker, and never have their
text loaded. `k` is preserved because the top-k is computed over the permitted set, not
over the corpus. Perimeter is retrieval infrastructure exposed as an MCP tool server, and
it deploys air-gapped as a single container with no external vector database. What makes
it different from a RAG demo is that the access-control guarantee is the product: it is
stated as five numbered invariants, each with a named test that fails the build.

## Non-goals

- No multi-node distribution, sharding, or replication. Single process, single container.
- No auth server, no user database, no login. Identity arrives from the caller; Perimeter
  trusts it and enforces against it. It never issues identity.
- No agent framework, chat UI, or conversation memory. Perimeter returns scoped, cited
  chunks. What consumes them is out of scope.
- No external vector database. That constraint is the point, not a limitation to route
  around.
- No streaming LLM generation.

## Architecture map

```
perimeter/
  core/                     # pure domain: stdlib only, zero I/O
    principal.py            # Principal, PrincipalId, GroupId, effective_principals()
    acl.py                  # AccessPolicy, Grant, Deny, PermissionSet
    document.py             # Document, Chunk, ChunkId, SourceRef
    query.py                # RetrievalRequest, ScopedResult, Citation
    ports.py                # Protocols: EmbeddingModel, Reranker, VectorIndex,
                            #   DocumentStore, AclResolver, Clock
    errors.py               # typed domain errors; no bare exceptions anywhere
  index/                    # the vector index: NumPy only, no server
    quantize.py             # binary + int8 quantization, float rescoring
    flat.py                 # memory-mapped flat index, mmap'd at open
    filtered_search.py      # allow-list applied INSIDE the scan (INV-2)
  adapters/
    cohere_embeddings.py    # embed-v4.0 over HTTPS via httpx
    cohere_rerank.py        # rerank-v4.0-fast over HTTPS via httpx
    postgres_store.py
    memory_store.py
    caching_acl_resolver.py # explicit TTL + invalidation hook (ADR-004)
  connectors/
    base.py                 # Connector protocol: enumerate / fetch / acl_for
    filesystem.py
    gdrive.py
  pipeline/
    ingest.py               # chunk, embed, extract ACLs, write
    retrieve.py             # resolve principals -> filtered search -> rerank -> assemble
  server/
    mcp.py                  # MCP tool server (StreamableHTTP)
    auth.py                 # identity + connector-token extraction, request-scoped
    http.py                 # FastAPI: /health, admin API
    telemetry.py            # OpenTelemetry, record_sensitive_data=False
admin/                      # React 18 + TypeScript + Vite admin console
bench/                      # benchmark harness; emits the README results table
docs/adr/                   # architecture decision records
tests/                      # unit, property, integration, invariant, bench gates
```

**Dependency rule.** `perimeter/core/` imports nothing but the Python standard library.
No FastAPI, no NumPy, no HTTP client, no database driver, no `typing_extensions`. All I/O
is expressed as `typing.Protocol` definitions in `core/ports.py` and implemented in
`adapters/`. Outer layers may import `core`; `core` never imports an outer layer.
`tests/test_architecture.py` parses every module under `core/` and fails CI on a
violation. This rule is what makes the access-control logic testable without a network or
a database.

Layering, outermost first: `server` -> `pipeline` -> `adapters` / `index` / `connectors`
-> `core`. `index/` depends on NumPy and `core` only. `pipeline/` depends on `core` ports,
never on a concrete adapter.

## Security invariants

These are the product. Each is enforced by a named test marked `@pytest.mark.invariant`.
Run them alone with `make test-invariants`. A failure here is a security bug, not a test
flake.

| ID | Invariant | Enforcing test |
|----|-----------|----------------|
| INV-1 | No chunk text is ever returned for a chunk whose access policy does not admit the caller. | `tests/invariants/test_inv1_no_unpermitted_text.py::test_inv1_no_chunk_text_for_unpermitted_caller`, and universally over random policy graphs in `tests/test_acl_leak_property.py::test_inv1_holds_for_all_policy_graphs` |
| INV-2 | The candidate set entering the reranker is a strict subset of the caller's permitted set. Filtering happens inside the index scan, never after. | `tests/invariants/test_inv2_filter_inside_scan.py::test_inv2_reranker_input_is_subset_of_permitted` |
| INV-3 | Connector OAuth tokens never enter logs, traces, error messages, or persistent storage. They live in request scope and are dropped when the request ends. | `tests/invariants/test_inv3_token_scope.py::test_inv3_connector_token_never_persisted_or_logged` |
| INV-4 | An empty permitted set returns an empty result. It never falls back to unfiltered search. Fail closed, always. | `tests/invariants/test_inv4_fail_closed.py::test_inv4_empty_permitted_set_yields_empty_result` |
| INV-5 | A stale ACL cache entry can only ever be more restrictive than reality, never less. Revocation takes effect immediately; grants may lag. | `tests/invariants/test_inv5_cache_monotonic.py::test_inv5_stale_cache_entry_never_less_restrictive` |

Where each is enforced in code:

- INV-1: `pipeline/retrieve.py` re-checks `AccessPolicy.admits()` on every chunk before
  assembling the result, and `DocumentStore.get_chunks()` takes the permission set and
  refuses to return text for chunks it does not admit. Two independent checks; the index
  filter (INV-2) is a third.
- INV-2: `index/filtered_search.py` gathers only permitted rows before computing any
  distance. Unpermitted rows are never scored.
- INV-3: `server/auth.py` holds tokens in a request-scoped context that is cleared on
  exit; `server/telemetry.py` sets `record_sensitive_data=False`; the logging redaction
  filter in `server/logging.py` scrubs anything token-shaped.
- INV-4: `pipeline/retrieve.py` returns an empty `ScopedResult` before touching the index
  when the permitted set is empty. `index/filtered_search.py` independently returns nothing
  for an empty allow-list rather than an unfiltered scan.
- INV-5: `adapters/caching_acl_resolver.py` never serves a stale entry on upstream error
  (it returns the empty set: fail closed), and `invalidate()` evicts synchronously before
  returning. Only grants can lag, bounded by the TTL. "Revocation" includes a membership
  *addition* when explicit denies are in use, because admission is only monotone for
  grant-only policies (`tests/core/test_properties.py` proves both halves; ADR-004
  explains). The hook therefore fires on any membership change, in either direction.

## ADR index

- [ADR-001](docs/adr/001-acl-filtering-inside-the-index-scan.md): ACL filtering inside the index scan, not as a post-filter.
- [ADR-002](docs/adr/002-no-external-vector-database.md): No external vector database; flat quantized index memory-mapped from disk.
- [ADR-003](docs/adr/003-ports-and-adapters-stdlib-core.md): Ports and adapters with a stdlib-only core.
- [ADR-004](docs/adr/004-acl-cache-ttl-is-a-security-parameter.md): ACL resolution cached with an explicit TTL and an invalidation hook.
- [ADR-005](docs/adr/005-request-scoped-connector-tokens.md): Connector tokens are request-scoped and never persisted.
- [ADR-006](docs/adr/006-index-stores-no-plaintext.md): The index stores no plaintext; only vectors and chunk IDs.

## Conventions

- Full type annotations on every function, including tests. `mypy --strict` passes.
- No bare `except:` and no `except Exception:` that swallows. Catch the typed error you
  can handle; everything else propagates as a typed domain error from `core/errors.py`.
- Domain errors are typed: `PerimeterError` subclasses, never `ValueError` at a public
  boundary. Error messages never contain document text, tokens, or full permission sets.
- Structured logging through `logging` with a redaction filter installed at process start.
  Never `print`. Ruff rule `T20` enforces this outside `bench/`.
- Immutable domain objects: `@dataclass(frozen=True, slots=True)`. Sets of principals are
  `frozenset`.
- Core types are plain stdlib. Vectors cross the core boundary as `array.array('f')`,
  which NumPy wraps zero-copy.
- Conventional Commits with a body that explains why. History is documentation.
- Measured numbers only. No performance figure appears in the README or an ADR unless
  `bench/` produced it.

## Testing strategy

- **Unit tests for `core/`** with no I/O and no test doubles beyond in-memory values.
- **Property-based tests** with Hypothesis over randomly generated principal graphs and
  access policies, asserting INV-1 holds universally: for every generated corpus and every
  generated caller, the set of returned chunk IDs is a subset of the chunks the policy
  admits for that caller. Also used for principal-resolution laws (idempotence, monotonic
  growth under added memberships, cycle safety).
- **Integration tests** against the in-memory document store and a real on-disk index in a
  temp directory, driving the pipeline and the MCP server end to end.
- **Invariant tests** (`tests/invariants/`, marker `invariant`): one per INV, named in the
  table above. They run in CI as their own step.
- **Recall benchmark as a regression gate** (`tests/bench/`, marker `bench`): recall@10 of
  the quantized filtered search against an exact float32 scan must not fall below the
  budget below.

## Performance budget

Build gates, not aspirations. `tests/bench/` fails CI if any is exceeded. Measured values
live in the README and are produced only by `make bench`.

| Metric | Ceiling | Measured on |
|--------|---------|-------------|
| p95 end-to-end retrieval latency, 50k-chunk corpus, in-process (Cohere calls stubbed at zero cost) | <= 30 ms | `bench/run.py` |
| Peak RSS during a sustained query loop, 50k-chunk corpus | <= 512 MiB | `bench/run.py` |
| Index bytes per chunk (binary codes + int8 codes + ids + ACL rows) | <= 1,280 bytes | `bench/run.py` |
| recall@10 versus an exact float32 baseline | >= 0.95 | `tests/bench/test_recall_gate.py` |
| Cost per query in Cohere API calls | <= 2 (one embed, one rerank) | `bench/run.py` |

Embedding dimension is 1024 (`embed-v4.0` with `output_dimension=1024`). Index storage per
vector is 128 bytes of binary code for the scan plus 1,024 bytes of int8 code for
rescoring; there is no float32 copy on disk.

## Commands

```
make install          # uv sync --all-groups; npm ci in admin/
make test             # pytest (unit, property, integration, invariant)
make test-invariants  # only the INV-1..INV-5 tests, verbose
make lint             # ruff check + ruff format --check
make typecheck        # mypy --strict
make check            # lint + typecheck + test
make bench            # run the benchmark harness; writes bench/results.md
make bench-gate       # the CI regression gates over the budget above
make run              # perimeter serve (MCP + HTTP on :8000)
make build-admin      # vite build of the admin console
make container        # docker build -t perimeter:local .
```

Configuration is by environment: `COHERE_API_KEY` (absent means the deterministic local
embedder is used, which is what the air-gapped demo and CI run), `PERIMETER_DATA_DIR`,
`PERIMETER_DATABASE_URL` (absent means the in-memory store), `PERIMETER_ACL_TTL_SECONDS`.

## Definition of done

Before any change is committed, all of the following are true:

1. `make lint` and `make typecheck` pass.
2. `make test` passes, including every `invariant` test.
3. `make bench-gate` passes; no budget above has regressed.
4. If the change alters or contradicts any decision in this file or an ADR, this file and
   the ADR are updated in the same commit. A superseded ADR is marked superseded, never
   deleted or rewritten.
5. The commit message says why, not only what.
