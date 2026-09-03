"""Structured logging with a redaction filter."""

from __future__ import annotations

import io
import json
import logging

import pytest

from perimeter.server.auth import TOKEN_HEADER_PREFIX, ConnectorToken, request_scope
from perimeter.server.logging import RedactionFilter, configure_logging, redact

SECRET = "ya29.A0ARrdaM-super-secret-token-value-1234567890"


@pytest.mark.parametrize(
    "text",
    [
        f"Authorization: Bearer {SECRET}",
        f"token={SECRET}",
        "key sk-abcdefghijklmnopqrstuvwxyz0123456789ABCD here",
        "AKIAIOSFODNN7EXAMPLEX",
        f"ya29.{'x' * 40}",
    ],
)
def test_redact_scrubs_token_shaped_values(text: str) -> None:
    out = redact(text)
    assert SECRET not in out
    assert "sk-abcdefghijklmnopqrstuvwxyz0123456789ABCD" not in out
    assert "AKIAIOSFODNN7EXAMPLEX" not in out
    assert "[REDACTED]" in out


def test_redact_leaves_ordinary_text_alone() -> None:
    text = "resolved 3 principals for request 7f3a in 12ms"
    assert redact(text) == text


def test_redact_scrubs_exact_in_scope_token_values_regardless_of_shape() -> None:
    with request_scope({f"{TOKEN_HEADER_PREFIX}fs": "short"}):
        assert "short" not in redact("the token is short indeed")


def _capture() -> tuple[logging.Logger, io.StringIO]:
    stream = io.StringIO()
    logger = logging.getLogger("perimeter.test.redaction")
    logger.handlers.clear()
    logger.propagate = False
    handler = logging.StreamHandler(stream)
    handler.addFilter(RedactionFilter())
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger, stream


def test_filter_scrubs_message_args_and_exception_text() -> None:
    logger, stream = _capture()
    token = ConnectorToken(SECRET)
    logger.info("raw %s", SECRET)
    logger.info("obj %r", token)
    logger.info(f"fstring {SECRET}")
    try:
        raise ValueError(f"failed with {SECRET}")
    except ValueError:
        logger.exception("boom")
    output = stream.getvalue()
    assert SECRET not in output
    assert output.count("[REDACTED]") >= 3


def test_configure_logging_installs_filter_and_emits_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = configure_logging(level="INFO")
    try:
        logging.getLogger("perimeter.something").info("hello %s", SECRET)
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
    err = capsys.readouterr().err
    line = json.loads(err.strip().splitlines()[-1])
    assert line["logger"] == "perimeter.something"
    assert SECRET not in err
    assert line["message"] == "hello [REDACTED]"
