"""Identity extraction, request-scoped.

Perimeter does not authenticate. A trusted front door (gateway, sidecar, the
MCP host) authenticates the user and forwards the result in headers:

    X-Perimeter-Principal: alice@example.com
    X-Perimeter-Groups:    eng, staff          (optional, comma-separated)

Perimeter trusts these values and enforces against them. A request without a
principal is refused before any retrieval work happens, and the reserved
public principal cannot be forwarded as an identity.
"""

from __future__ import annotations

from collections.abc import Mapping

from perimeter.core.errors import AuthError, InvalidPrincipalError
from perimeter.core.principal import GroupId, Principal, parse_group_id, parse_principal_id

PRINCIPAL_HEADER = "X-Perimeter-Principal"
GROUPS_HEADER = "X-Perimeter-Groups"


def _lookup(headers: Mapping[str, str], name: str) -> str | None:
    wanted = name.lower()
    for key, value in headers.items():
        if key.lower() == wanted:
            return value
    return None


def identity_from_headers(headers: Mapping[str, str] | None) -> Principal:
    """Build the caller's :class:`Principal` from forwarded headers, or raise ``AuthError``."""
    if not headers:
        raise AuthError(f"identity required: missing {PRINCIPAL_HEADER} header")
    raw_principal = _lookup(headers, PRINCIPAL_HEADER)
    if raw_principal is None or not raw_principal.strip():
        raise AuthError(f"identity required: missing {PRINCIPAL_HEADER} header")
    try:
        principal_id = parse_principal_id(raw_principal.strip())
    except InvalidPrincipalError:
        raise AuthError(f"identity rejected: malformed {PRINCIPAL_HEADER}") from None

    groups: set[GroupId] = set()
    raw_groups = _lookup(headers, GROUPS_HEADER) or ""
    for part in raw_groups.split(","):
        name = part.strip()
        if not name:
            continue
        try:
            groups.add(parse_group_id(name))
        except InvalidPrincipalError:
            raise AuthError(f"identity rejected: malformed {GROUPS_HEADER}") from None
    return Principal(id=principal_id, groups=frozenset(groups))
