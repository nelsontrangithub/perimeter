"""Shared plumbing for the Cohere adapters: one client, one error path, no leaks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

from perimeter.core.errors import PerimeterError

DEFAULT_BASE_URL = "https://api.cohere.com"
DEFAULT_TIMEOUT_SECONDS = 30.0


def make_client(
    base_url: str = DEFAULT_BASE_URL, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> httpx.Client:
    return httpx.Client(base_url=base_url, timeout=timeout)


def post_json(
    client: httpx.Client,
    path: str,
    api_key: str,
    body: Mapping[str, Any],
    error_type: type[PerimeterError],
    what: str,
) -> dict[str, Any]:
    """POST and return the JSON object, translating every failure into ``error_type``.

    The raised message names the operation and the HTTP status only. It never
    includes the request body (which holds document or query text), the response
    body (which may echo it), or the API key.
    """
    try:
        response = client.post(
            path, json=dict(body), headers={"Authorization": f"Bearer {api_key}"}
        )
    except httpx.HTTPError as exc:
        raise error_type(f"{what}: transport error ({type(exc).__name__})") from None
    if response.status_code != 200:
        raise error_type(f"{what}: HTTP {response.status_code}")
    try:
        data = response.json()
    except ValueError:
        raise error_type(f"{what}: response is not JSON") from None
    if not isinstance(data, dict):
        raise error_type(f"{what}: response is not an object")
    return data
