"""Identity extraction from forwarded headers. Perimeter trusts, it does not issue."""

from __future__ import annotations

import pytest

from perimeter.core.errors import AuthError
from perimeter.core.principal import EVERYONE, GroupId, PrincipalId
from perimeter.server.auth import GROUPS_HEADER, PRINCIPAL_HEADER, identity_from_headers


def test_principal_and_groups_are_parsed() -> None:
    p = identity_from_headers({PRINCIPAL_HEADER: "alice@example.com", GROUPS_HEADER: "eng, staff"})
    assert p.id == PrincipalId("alice@example.com")
    assert p.groups == frozenset({GroupId(PrincipalId("eng")), GroupId(PrincipalId("staff"))})


def test_header_names_are_case_insensitive() -> None:
    p = identity_from_headers({PRINCIPAL_HEADER.lower(): "alice", GROUPS_HEADER.upper(): "eng"})
    assert p.id == PrincipalId("alice")
    assert GroupId(PrincipalId("eng")) in p.groups


def test_groups_header_is_optional_and_empty_entries_are_ignored() -> None:
    assert identity_from_headers({PRINCIPAL_HEADER: "alice"}).groups == frozenset()
    assert (
        identity_from_headers({PRINCIPAL_HEADER: "alice", GROUPS_HEADER: " , ,"}).groups
        == frozenset()
    )


@pytest.mark.parametrize("headers", [None, {}, {PRINCIPAL_HEADER: ""}, {PRINCIPAL_HEADER: "  "}])
def test_missing_principal_fails_closed(headers: dict[str, str] | None) -> None:
    with pytest.raises(AuthError):
        identity_from_headers(headers)


def test_reserved_everyone_cannot_be_forwarded_as_identity() -> None:
    with pytest.raises(AuthError):
        identity_from_headers({PRINCIPAL_HEADER: str(EVERYONE)})
    with pytest.raises(AuthError):
        identity_from_headers({PRINCIPAL_HEADER: "alice", GROUPS_HEADER: str(EVERYONE)})


def test_malformed_group_rejects_whole_identity() -> None:
    with pytest.raises(AuthError):
        identity_from_headers({PRINCIPAL_HEADER: "alice", GROUPS_HEADER: "eng,bad\tgroup"})


def test_auth_error_message_does_not_echo_header_values() -> None:
    with pytest.raises(AuthError) as info:
        identity_from_headers({PRINCIPAL_HEADER: "weird\x01value"})
    assert "weird" not in str(info.value)
