"""INV-5: a stale ACL cache entry can only ever be more restrictive than reality, never
less. Revocation takes effect immediately; grants may lag.

The contract (ADR-004): every membership change that the upstream knows about is
signalled through the invalidation hook. Under that contract, whatever the cache
serves is a subset of the current truth at all times. Grants that arrive silently
lag by at most the TTL; a silent revocation is the one case the cache cannot see,
and its exposure is bounded by the TTL, which is why the TTL is a security
parameter.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from perimeter.adapters.caching_acl_resolver import CachingAclResolver
from perimeter.core.acl import PermissionSet
from perimeter.core.errors import AclResolutionError
from perimeter.core.principal import EVERYONE, GroupId, Principal, PrincipalId

pytestmark = pytest.mark.invariant

TTL = 60.0
ALICE = PrincipalId("alice")
GROUPS = [GroupId(PrincipalId(g)) for g in ("eng", "sales", "contractors", "staff")]


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> datetime:
        return datetime.fromtimestamp(self.t, tz=UTC)

    def monotonic(self) -> float:
        return self.t


class Directory:
    def __init__(self) -> None:
        self.groups: set[GroupId] = {GROUPS[0]}
        self.down = False

    def resolve(self, principal: Principal) -> PermissionSet:
        if self.down:
            raise AclResolutionError("down")
        return PermissionSet(frozenset({principal.id, EVERYONE, *self.groups}))


def test_inv5_stale_cache_entry_never_less_restrictive() -> None:
    clock, directory = Clock(), Directory()
    cache = CachingAclResolver(directory, ttl_seconds=TTL, clock=clock)
    alice = Principal(ALICE)

    served = cache.resolve(alice)
    assert GROUPS[0] in served

    # Signalled revocation: immediate, even though the entry is fresh.
    directory.groups.discard(GROUPS[0])
    cache.invalidate(ALICE)
    assert GROUPS[0] not in cache.resolve(alice)

    # Silent grant: lags (the cached entry is a subset of truth), never the reverse.
    directory.groups.add(GROUPS[1])
    served = cache.resolve(alice)
    assert GROUPS[1] not in served
    assert served.principals <= frozenset({ALICE, EVERYONE, *directory.groups})

    # The lag is bounded by the TTL.
    clock.t += TTL + 0.01
    assert GROUPS[1] in cache.resolve(alice)

    # Directory outage after expiry: empty set, never the stale entry.
    clock.t += TTL + 0.01
    directory.down = True
    assert cache.resolve(alice).is_empty
    directory.down = False

    # In-place restriction without a refetch shrinks the entry and nothing else.
    cache.resolve(alice)
    cache.restrict(ALICE, GROUPS[1])
    assert GROUPS[1] not in cache.resolve(alice)


Event = tuple[str, int]
events = st.lists(
    st.tuples(st.sampled_from(["grant", "revoke", "tick", "outage", "resolve"]), st.integers(0, 3)),
    max_size=40,
)


@given(events)
@settings(max_examples=200, deadline=None)
def test_inv5_served_set_is_subset_of_truth_under_signalled_changes(script: list[Event]) -> None:
    """Every membership change is signalled; the served set never exceeds the truth."""
    clock, directory = Clock(), Directory()
    cache = CachingAclResolver(directory, ttl_seconds=TTL, clock=clock)
    alice = Principal(ALICE)
    for kind, i in script:
        if kind == "grant":
            directory.groups.add(GROUPS[i])
            cache.invalidate(ALICE)
        elif kind == "revoke":
            directory.groups.discard(GROUPS[i])
            cache.invalidate(ALICE)
        elif kind == "tick":
            clock.t += (i + 1) * 20.0
        elif kind == "outage":
            directory.down = i % 2 == 0
        served = cache.resolve(alice)
        truth = frozenset({ALICE, EVERYONE, *directory.groups})
        assert served.principals <= truth, f"served {set(served) - truth} beyond truth"


@given(events)
@settings(max_examples=200, deadline=None)
def test_inv5_silent_changes_are_bounded_by_ttl(script: list[Event]) -> None:
    """With no signals at all, anything the cache serves was true within the last TTL."""
    clock, directory = Clock(), Directory()
    cache = CachingAclResolver(directory, ttl_seconds=TTL, clock=clock)
    alice = Principal(ALICE)
    history: list[tuple[float, frozenset[GroupId]]] = [(0.0, frozenset(directory.groups))]
    for kind, i in script:
        if kind == "grant":
            directory.groups.add(GROUPS[i])
        elif kind == "revoke":
            directory.groups.discard(GROUPS[i])
        elif kind == "tick":
            clock.t += (i + 1) * 20.0
        elif kind == "outage":
            directory.down = i % 2 == 0
        history.append((clock.t, frozenset(directory.groups)))
        served = cache.resolve(alice)
        recent_truths = [g for t, g in history if clock.t - t < TTL]
        union = frozenset({ALICE, EVERYONE}).union(*recent_truths)
        assert served.principals <= union
