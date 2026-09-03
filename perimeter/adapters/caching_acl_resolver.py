"""ACL resolution cache with an explicit TTL and an invalidation hook (ADR-004, INV-5).

Rules, in the order they matter:

1. A fresh entry (younger than the TTL) is served without consulting upstream.
2. On a miss or an expired entry, upstream is consulted. If upstream fails, the
   expired entry is discarded and the *empty* permission set is returned. Stale
   entries are never served on error; the caller sees nothing (INV-4).
3. :meth:`invalidate` evicts synchronously and returns only when the eviction is
   done, so the next resolve for that principal reaches upstream. Anything that
   learns of a membership change (either direction: see ADR-004 on explicit
   denies) calls it.
4. :meth:`restrict` removes one group from a cached entry in place. A cached
   entry can shrink without a refetch; it can never grow without one.

The cache key includes the forwarded direct memberships, so a caller whose
front door forwards different groups on different calls does not receive
another call's resolution.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from perimeter.core.acl import PermissionSet
from perimeter.core.errors import AclResolutionError
from perimeter.core.ports import AclResolver, Clock
from perimeter.core.principal import GroupId, Principal, PrincipalId

DEFAULT_TTL_SECONDS = 60.0
DEFAULT_MAX_ENTRIES = 10_000

_Key = tuple[PrincipalId, frozenset[GroupId]]


@dataclass(slots=True)
class _Entry:
    permissions: PermissionSet
    fetched_at: float


@dataclass(slots=True)
class CacheStats:
    hits: int = 0
    misses: int = 0
    errors: int = 0
    evictions: int = 0
    size: int = 0


class CachingAclResolver:
    def __init__(
        self,
        inner: AclResolver,
        *,
        clock: Clock,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> None:
        if ttl_seconds < 0:
            raise AclResolutionError("ttl_seconds must be >= 0")
        self._inner = inner
        self._clock = clock
        self._ttl = ttl_seconds
        self._max = max(1, max_entries)
        self._entries: OrderedDict[_Key, _Entry] = OrderedDict()
        self.stats = CacheStats()

    @property
    def ttl_seconds(self) -> float:
        return self._ttl

    def resolve(self, principal: Principal) -> PermissionSet:
        key: _Key = (principal.id, principal.groups)
        now = self._clock.monotonic()
        entry = self._entries.get(key) if self._ttl > 0 else None
        if entry is not None and now - entry.fetched_at < self._ttl:
            self.stats.hits += 1
            self._entries.move_to_end(key)
            return entry.permissions
        self.stats.misses += 1
        try:
            permissions = self._inner.resolve(principal)
        except AclResolutionError:
            self.stats.errors += 1
            self._entries.pop(key, None)
            self.stats.size = len(self._entries)
            return PermissionSet.empty()
        if self._ttl > 0:
            self._entries[key] = _Entry(permissions, now)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)
                self.stats.evictions += 1
        self.stats.size = len(self._entries)
        return permissions

    # -- invalidation hook ------------------------------------------------------

    def invalidate(self, principal_id: PrincipalId) -> None:
        """Evict every entry for ``principal_id``. Returns after the eviction is done."""
        for key in [k for k in self._entries if k[0] == principal_id]:
            del self._entries[key]
            self.stats.evictions += 1
        self.stats.size = len(self._entries)

    def invalidate_group(self, group: GroupId) -> None:
        """Evict every entry whose resolution or forwarded memberships include ``group``."""
        for key in [k for k, e in self._entries.items() if group in e.permissions or group in k[1]]:
            del self._entries[key]
            self.stats.evictions += 1
        self.stats.size = len(self._entries)

    def invalidate_all(self) -> None:
        self.stats.evictions += len(self._entries)
        self._entries.clear()
        self.stats.size = 0

    def restrict(self, principal_id: PrincipalId, group: GroupId) -> None:
        """Remove ``group`` from every cached entry for ``principal_id`` without refetching."""
        for key, entry in self._entries.items():
            if key[0] == principal_id:
                entry.permissions = entry.permissions.without(group)
