"""Caching ACL resolver: TTL, invalidation hook, fail-closed on upstream error."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from perimeter.adapters.caching_acl_resolver import CachingAclResolver
from perimeter.core.acl import PermissionSet
from perimeter.core.errors import AclResolutionError
from perimeter.core.ports import AclResolver
from perimeter.core.principal import EVERYONE, GroupId, Principal, PrincipalId

ALICE = PrincipalId("alice")
ENG = GroupId(PrincipalId("eng"))
SALES = GroupId(PrincipalId("sales"))


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def now(self) -> datetime:
        return datetime.fromtimestamp(self.t, tz=UTC)

    def monotonic(self) -> float:
        return self.t


class Upstream:
    """Mutable truth. ``memberships`` is what the directory says right now."""

    def __init__(self) -> None:
        self.memberships: dict[PrincipalId, set[GroupId]] = {ALICE: {ENG}}
        self.calls = 0
        self.fail = False

    def resolve(self, principal: Principal) -> PermissionSet:
        self.calls += 1
        if self.fail:
            raise AclResolutionError("directory down")
        groups = self.memberships.get(principal.id, set()) | set(principal.groups)
        return PermissionSet(frozenset({principal.id, EVERYONE, *groups}))


@pytest.fixture
def parts() -> tuple[CachingAclResolver, Upstream, FakeClock]:
    upstream, clock = Upstream(), FakeClock()
    return CachingAclResolver(upstream, ttl_seconds=60, clock=clock), upstream, clock


def test_satisfies_port(parts: tuple[CachingAclResolver, Upstream, FakeClock]) -> None:
    assert isinstance(parts[0], AclResolver)


def test_second_call_within_ttl_is_served_from_cache(
    parts: tuple[CachingAclResolver, Upstream, FakeClock],
) -> None:
    cache, upstream, _ = parts
    a = cache.resolve(Principal(ALICE))
    b = cache.resolve(Principal(ALICE))
    assert a == b and ENG in a
    assert upstream.calls == 1
    assert cache.stats.hits == 1 and cache.stats.misses == 1


def test_entry_expires_after_ttl(parts: tuple[CachingAclResolver, Upstream, FakeClock]) -> None:
    cache, upstream, clock = parts
    cache.resolve(Principal(ALICE))
    clock.t += 59.9
    cache.resolve(Principal(ALICE))
    assert upstream.calls == 1
    clock.t += 0.2
    cache.resolve(Principal(ALICE))
    assert upstream.calls == 2


def test_upstream_error_on_miss_returns_empty_not_stale(
    parts: tuple[CachingAclResolver, Upstream, FakeClock],
) -> None:
    cache, upstream, clock = parts
    cache.resolve(Principal(ALICE))
    clock.t += 61
    upstream.fail = True
    assert cache.resolve(Principal(ALICE)).is_empty
    upstream.fail = False
    assert ENG in cache.resolve(Principal(ALICE))


def test_upstream_error_result_is_not_cached(
    parts: tuple[CachingAclResolver, Upstream, FakeClock],
) -> None:
    cache, upstream, _ = parts
    upstream.fail = True
    assert cache.resolve(Principal(ALICE)).is_empty
    upstream.fail = False
    assert not cache.resolve(Principal(ALICE)).is_empty


def test_invalidate_evicts_synchronously(
    parts: tuple[CachingAclResolver, Upstream, FakeClock],
) -> None:
    cache, upstream, _ = parts
    cache.resolve(Principal(ALICE))
    upstream.memberships[ALICE] = set()
    cache.invalidate(ALICE)
    assert ENG not in cache.resolve(Principal(ALICE))
    assert upstream.calls == 2


def test_invalidate_group_evicts_every_entry_containing_it(
    parts: tuple[CachingAclResolver, Upstream, FakeClock],
) -> None:
    cache, upstream, _ = parts
    bob = PrincipalId("bob")
    upstream.memberships[bob] = {ENG}
    cache.resolve(Principal(ALICE))
    cache.resolve(Principal(bob))
    upstream.memberships = {ALICE: set(), bob: set()}
    cache.invalidate_group(ENG)
    assert ENG not in cache.resolve(Principal(ALICE))
    assert ENG not in cache.resolve(Principal(bob))
    assert upstream.calls == 4


def test_invalidate_all(parts: tuple[CachingAclResolver, Upstream, FakeClock]) -> None:
    cache, upstream, _ = parts
    cache.resolve(Principal(ALICE))
    cache.invalidate_all()
    cache.resolve(Principal(ALICE))
    assert upstream.calls == 2


def test_restrict_shrinks_cached_entry_in_place_without_refetch(
    parts: tuple[CachingAclResolver, Upstream, FakeClock],
) -> None:
    cache, upstream, _ = parts
    cache.resolve(Principal(ALICE))
    cache.restrict(ALICE, ENG)
    served = cache.resolve(Principal(ALICE))
    assert ENG not in served and ALICE in served
    assert upstream.calls == 1


def test_forwarded_groups_are_part_of_the_cache_key(
    parts: tuple[CachingAclResolver, Upstream, FakeClock],
) -> None:
    cache, upstream, _ = parts
    cache.resolve(Principal(ALICE))
    with_sales = cache.resolve(Principal(ALICE, frozenset({SALES})))
    assert SALES in with_sales
    assert upstream.calls == 2


def test_zero_ttl_disables_caching(parts: tuple[CachingAclResolver, Upstream, FakeClock]) -> None:
    _, upstream, clock = parts
    cache = CachingAclResolver(upstream, ttl_seconds=0, clock=clock)
    cache.resolve(Principal(ALICE))
    cache.resolve(Principal(ALICE))
    assert upstream.calls == 2


def test_cache_is_bounded(parts: tuple[CachingAclResolver, Upstream, FakeClock]) -> None:
    _, upstream, clock = parts
    cache = CachingAclResolver(upstream, ttl_seconds=60, clock=clock, max_entries=2)
    for name in ("a", "b", "c"):
        cache.resolve(Principal(PrincipalId(name)))
    assert cache.stats.size == 2
    cache.resolve(Principal(PrincipalId("a")))
    assert upstream.calls == 4
