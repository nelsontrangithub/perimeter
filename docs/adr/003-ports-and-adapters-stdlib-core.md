# ADR-003: Ports and adapters with a stdlib-only core

Status: Accepted

## Context

The access-control logic is the product. It must be testable exhaustively, including with
property-based tests over thousands of generated policy graphs per run, without standing
up a database, calling an embedding API, or allocating NumPy arrays. It must also be
readable by a reviewer without knowledge of any framework.

## Decision

`perimeter/core/` imports nothing but the Python standard library. It defines the domain
types (principals, policies, documents, chunks, requests, results), the typed errors, and
the I/O ports as `typing.Protocol` classes: `EmbeddingModel`, `Reranker`, `VectorIndex`,
`DocumentStore`, `AclResolver`, `Clock`. Everything that touches the outside world lives in
`adapters/`, `index/`, `connectors/`, or `server/` and implements a port.

The rule is enforced by `tests/test_architecture.py`, which parses every module under
`core/` with `ast` and fails on any import outside `sys.stdlib_module_names`. The guard has
a self-test proving it detects a violation.

Vectors cross the boundary as `array.array('f')`, a stdlib type that NumPy wraps without
copying, so the core never needs NumPy and the index never pays for a list-of-floats
conversion.

## Consequences

- Core tests run in milliseconds with no fixtures beyond in-memory values.
- Adapters are thin and boring, which is what an adapter should be.
- Some duplication at the edges: the same policy check exists as a pure function in core
  and as a vectorized mask in the index. The property tests assert they agree.
- The pipeline depends on ports, not adapters, so the retrieval orchestrator is tested
  with the in-memory store and a fake embedder and never knows the difference.

## What would make us reverse it

Nothing about the product would. If the standard library lacked something the core
genuinely needed (it does not: `dataclasses`, `typing.Protocol`, `array`, `hashlib`, and
`datetime` cover it), the answer would be to move that concern behind a port, not to
relax the rule.
