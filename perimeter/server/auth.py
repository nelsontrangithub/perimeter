"""Identity extraction, request-scoped.

Perimeter does not authenticate. A trusted front door (gateway, sidecar, the
MCP host) authenticates the user and forwards the result in headers:

    X-Perimeter-Principal: alice@example.com
    X-Perimeter-Groups:    eng, staff          (optional, comma-separated)

Perimeter trusts these values and enforces against them. A request without a
principal is refused before any retrieval work happens, and the reserved
public principal cannot be forwarded as an identity.

Connector credentials travel the same way (``X-Perimeter-Token-<connector>``)
and live in a request scope that is wiped on exit (ADR-005, INV-3).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar

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


# --- connector tokens: request-scoped, never persisted (ADR-005, INV-3) -----------------

TOKEN_HEADER_PREFIX = "X-Perimeter-Token-"  # noqa: S105 - a header name, not a secret
"""``X-Perimeter-Token-<connector>: <secret>`` carries a per-connector credential."""


class ConnectorToken:
    """An opaque credential for one connector, valid for the current request only.

    The secret is reachable only through :meth:`reveal`. ``repr``, ``str``, and
    formatting never include it, equality is by identity, and :meth:`wipe`
    (called when the request scope ends) makes ``reveal`` raise.
    """

    __slots__ = ("__weakref__", "_secret")

    def __init__(self, secret: str) -> None:
        self._secret: str | None = secret

    def reveal(self) -> str:
        if self._secret is None:
            raise AuthError("connector token is no longer in scope")
        return self._secret

    def wipe(self) -> None:
        self._secret = None

    @property
    def is_live(self) -> bool:
        return self._secret is not None

    def __repr__(self) -> str:
        return "ConnectorToken([REDACTED])"

    __str__ = __repr__


def tokens_from_headers(headers: Mapping[str, str] | None) -> dict[str, ConnectorToken]:
    """Every ``X-Perimeter-Token-<connector>`` header, keyed by lower-cased connector name."""
    tokens: dict[str, ConnectorToken] = {}
    if not headers:
        return tokens
    prefix = TOKEN_HEADER_PREFIX.lower()
    for key, value in headers.items():
        lowered = key.lower()
        if not lowered.startswith(prefix):
            continue
        connector = lowered[len(prefix) :]
        if not connector or not value.strip():
            raise AuthError("connector token header is malformed")
        tokens[connector] = ConnectorToken(value.strip())
    return tokens


class RequestCredentials:
    """What one request is allowed to use. Dropped when the request ends."""

    __slots__ = ("tokens",)

    def __init__(self, tokens: dict[str, ConnectorToken]) -> None:
        self.tokens = tokens

    def wipe(self) -> None:
        for token in self.tokens.values():
            token.wipe()
        self.tokens = {}

    def __repr__(self) -> str:
        return f"RequestCredentials(connectors={sorted(self.tokens)})"


_credentials: ContextVar[RequestCredentials | None] = ContextVar(
    "perimeter_credentials", default=None
)


@contextmanager
def request_scope(headers: Mapping[str, str] | None) -> Iterator[RequestCredentials]:
    """Make the request's connector tokens available to :func:`token_for` until exit.

    On exit, whether normal or by exception, every token is wiped and the
    previous scope (if any) is restored.
    """
    credentials = RequestCredentials(tokens_from_headers(headers))
    reset = _credentials.set(credentials)
    try:
        yield credentials
    finally:
        _credentials.reset(reset)
        credentials.wipe()


def current_tokens() -> Mapping[str, ConnectorToken]:
    credentials = _credentials.get()
    return {} if credentials is None else dict(credentials.tokens)


def token_for(connector: str) -> ConnectorToken | None:
    return current_tokens().get(connector.lower())
