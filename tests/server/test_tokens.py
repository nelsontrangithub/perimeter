"""Connector tokens: extracted per request, hidden from repr, gone when the request ends."""

from __future__ import annotations

import gc
import weakref

import pytest

from perimeter.core.errors import AuthError
from perimeter.server.auth import (
    TOKEN_HEADER_PREFIX,
    ConnectorToken,
    current_tokens,
    request_scope,
    token_for,
    tokens_from_headers,
)

SECRET = "ya29.A0ARrdaM-super-secret-token-value-1234567890"


def test_tokens_are_extracted_per_connector_case_insensitively() -> None:
    tokens = tokens_from_headers(
        {f"{TOKEN_HEADER_PREFIX}Gdrive": SECRET, "x-perimeter-token-fs": "abc"}
    )
    assert set(tokens) == {"gdrive", "fs"}
    assert tokens["gdrive"].reveal() == SECRET


def test_no_token_headers_yields_empty_mapping() -> None:
    assert tokens_from_headers(None) == {}
    assert tokens_from_headers({"X-Other": "x"}) == {}


def test_blank_token_is_rejected() -> None:
    with pytest.raises(AuthError):
        tokens_from_headers({f"{TOKEN_HEADER_PREFIX}gdrive": "   "})


def test_token_repr_str_and_format_hide_the_secret() -> None:
    token = ConnectorToken(SECRET)
    percent_formatted = "%s" % token  # noqa: UP031 - logging formats records this way
    for rendered in (
        repr(token),
        str(token),
        f"{token}",
        f"{token!r}",
        percent_formatted,
        f"{[token]}",
    ):
        assert SECRET not in rendered
        assert "REDACTED" in rendered or "ConnectorToken" in rendered


def test_token_is_not_comparable_by_secret_and_not_hashable_by_secret() -> None:
    a, b = ConnectorToken(SECRET), ConnectorToken(SECRET)
    assert a != b
    assert hash(a) != hash(SECRET)


def test_request_scope_exposes_tokens_only_inside() -> None:
    assert current_tokens() == {}
    with request_scope({f"{TOKEN_HEADER_PREFIX}gdrive": SECRET}) as creds:
        assert token_for("gdrive") is creds.tokens["gdrive"]
        assert token_for("gdrive").reveal() == SECRET  # type: ignore[union-attr]
        assert token_for("nope") is None
    assert current_tokens() == {}
    assert token_for("gdrive") is None


def test_request_scope_wipes_tokens_on_exit_even_on_error() -> None:
    with pytest.raises(RuntimeError):
        with request_scope({f"{TOKEN_HEADER_PREFIX}gdrive": SECRET}) as creds:
            token = creds.tokens["gdrive"]
            raise RuntimeError("boom")
    with pytest.raises(AuthError):
        token.reveal()
    assert current_tokens() == {}


def test_request_scopes_nest_without_leaking_outward() -> None:
    with request_scope({f"{TOKEN_HEADER_PREFIX}a": "one"}):
        with request_scope({f"{TOKEN_HEADER_PREFIX}b": "two"}):
            assert set(current_tokens()) == {"b"}
        assert set(current_tokens()) == {"a"}
    assert current_tokens() == {}


def test_token_object_is_collectable_after_scope() -> None:
    with request_scope({f"{TOKEN_HEADER_PREFIX}gdrive": SECRET}) as creds:
        ref = weakref.ref(creds.tokens["gdrive"])
    del creds
    gc.collect()
    assert ref() is None
