# ADR-005: Connector tokens are request-scoped and never persisted

Status: Accepted

## Context

Connectors (Google Drive, a filesystem export, others later) need credentials to
enumerate and fetch documents and to read their ACLs. The tempting design is to store a
service credential or a per-user refresh token so Perimeter can act at any time. That
makes Perimeter a credential store, which it has no business being, and it means a
compromise of Perimeter is a compromise of every connected system.

## Decision

Perimeter acts *as the caller*. Connector OAuth tokens arrive on each request (as headers
on the MCP or admin HTTP call), are placed in a request-scoped context in
`server/auth.py`, are handed to the connector for the duration of that request, and are
dropped when the request ends. They are never written to the document store, the index,
a log line, a trace attribute, or an error message. This is INV-3.

Supporting mechanisms:

- `server/telemetry.py` configures OpenTelemetry with `record_sensitive_data=False` and
  never puts headers on spans.
- A logging redaction filter is installed at process start and scrubs any value that
  looks like a bearer token, as a backstop for a mistake elsewhere.
- Token-bearing objects implement `__repr__` that hides the secret, so an accidental
  f-string cannot leak it.
- Domain errors raised while a token is in scope carry no request context beyond an
  opaque request ID.

## Consequences

- Ingestion from a connector runs as the user who triggered it and sees exactly what they
  see. There is no "Perimeter service account" with broader access.
- Background re-indexing cannot happen without a caller present. Scheduled ingestion has
  to be driven by an external scheduler that supplies a token per run. This is the price
  of holding nothing.
- Every request that touches a connector must carry credentials, which makes the admin
  API slightly more annoying to use by hand.

## What would make us reverse it

A hard requirement for unattended, scheduled re-indexing with no caller in the loop. Even
then, the preferred answer is a narrowly scoped, short-lived credential injected by the
scheduler at run time rather than anything Perimeter stores.
