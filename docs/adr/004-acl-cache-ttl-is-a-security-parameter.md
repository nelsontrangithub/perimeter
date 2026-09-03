# ADR-004: ACL cache TTL is a security parameter

Status: Accepted

## Context

Resolving a caller's effective principals (their own ID, every group they belong to
transitively, and the public principal) can be expensive: it may involve a directory
lookup or a connector API call. Callers issue bursts of queries. Caching the resolution
is necessary for latency and for the cost budget. But a cached permission set is a
statement about the past, and a permission set that is stale in the permissive direction
is a leak.

## Decision

`adapters/caching_acl_resolver.py` wraps any `AclResolver` with:

1. An explicit TTL, default 60 seconds, configured by `PERIMETER_ACL_TTL_SECONDS`.
2. An invalidation hook, `revoke(principal_id)`, which evicts synchronously before
   returning. Anything that learns of a revocation (a connector webhook, an admin action,
   a directory sync) calls it.
3. No stale-on-error. When an entry has expired and the upstream resolver fails, the
   cache returns the empty permission set. It never serves the expired entry. Combined
   with INV-4, an upstream outage means callers see nothing, not everything.

Together these give INV-5: a cached entry can only be more restrictive than reality.
Grants lag by at most the TTL. Revocations that are signalled take effect immediately.
Revocations that are *not* signalled (an upstream that offers no hook) take effect within
the TTL, which is why the TTL is a security parameter: it is the maximum exposure window
for an unsignalled revocation.

Why 60 seconds. It is short enough that an unsignalled revocation is bounded to about a
minute, which matches the propagation delay of the directory systems Perimeter fronts
(Google Workspace group changes are documented as taking up to several minutes to
propagate on their own side, so a tighter TTL buys little real-world safety). It is long
enough that a caller running a multi-step retrieval loop costs one resolution rather than
one per query. An operator who needs a tighter bound lowers it; the cost is directly
visible as resolver calls per query in the benchmark table.

## Consequences

- Latency and API cost per query stay flat under bursty load.
- Operators get one knob and it is documented as a safety bound, not a speed setting.
- A resolver outage degrades to empty results, which is the correct failure mode for a
  permission system and the wrong one for availability. This is deliberate.
- The cache is per process. With one process (ADR-002) there is no coherence problem.

## What would make us reverse it

If Perimeter ever fronted a directory that pushed changes reliably (a real-time
membership stream), the TTL could become very long and the hook would carry the safety
guarantee alone. If it fronted one that offered no hook and had a compliance requirement
tighter than the TTL could reasonably be set, caching would have to be removed for that
resolver entirely and the cost paid per query.
