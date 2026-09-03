"""Connector configuration registry and the admin endpoints around it."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from perimeter.core.errors import ConnectorError
from perimeter.server.auth import TOKEN_HEADER_PREFIX
from perimeter.server.connectors import ConnectorConfig, ConnectorRegistry
from perimeter.server.http import build_app
from perimeter.server.settings import Settings
from perimeter.server.wiring import Runtime, build_runtime

pytestmark = pytest.mark.anyio

CORPUS = Path(__file__).resolve().parents[2] / "examples" / "corpus"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# --- registry ---------------------------------------------------------------------


def test_registry_persists_configs_without_secrets(tmp_path: Path) -> None:
    reg = ConnectorRegistry(tmp_path / "connectors.json")
    reg.add(ConnectorConfig(name="docs", kind="filesystem", root=str(CORPUS)))
    reg.add(ConnectorConfig(name="drive", kind="gdrive"))
    saved = json.loads((tmp_path / "connectors.json").read_text())
    assert [c["name"] for c in saved] == ["docs", "drive"]
    assert "token" not in json.dumps(saved).lower()
    again = ConnectorRegistry(tmp_path / "connectors.json")
    assert [c.name for c in again.list()] == ["docs", "drive"]


def test_registry_rejects_duplicates_bad_kinds_and_missing_root(tmp_path: Path) -> None:
    reg = ConnectorRegistry(tmp_path / "c.json")
    reg.add(ConnectorConfig(name="docs", kind="filesystem", root=str(CORPUS)))
    with pytest.raises(ConnectorError):
        reg.add(ConnectorConfig(name="docs", kind="filesystem", root=str(CORPUS)))
    with pytest.raises(ConnectorError):
        reg.add(ConnectorConfig(name="x", kind="filesystem", root=None))
    with pytest.raises(ConnectorError):
        reg.add(ConnectorConfig(name="y", kind="nope", root=None))  # type: ignore[arg-type]
    with pytest.raises(ConnectorError):
        reg.add(ConnectorConfig(name="bad name!", kind="gdrive"))


def test_registry_remove_and_build(tmp_path: Path) -> None:
    reg = ConnectorRegistry(tmp_path / "c.json")
    reg.add(ConnectorConfig(name="docs", kind="filesystem", root=str(CORPUS)))
    connector = reg.build("docs")
    assert connector.name == "filesystem"
    reg.remove("docs")
    assert reg.list() == []
    with pytest.raises(ConnectorError):
        reg.build("docs")


# --- endpoints -----------------------------------------------------------------------


@pytest.fixture
def runtime(tmp_path: Path) -> Runtime:
    return build_runtime(Settings(data_dir=tmp_path / "data"))


@pytest.fixture
async def client(runtime: Runtime) -> AsyncIterator[httpx.AsyncClient]:
    app = build_app(runtime)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000"
        ) as c:
            yield c


async def test_connectors_crud_and_ingest(client: httpx.AsyncClient, runtime: Runtime) -> None:
    assert (await client.get("/admin/api/connectors")).json() == []
    created = await client.post(
        "/admin/api/connectors", json={"name": "docs", "kind": "filesystem", "root": str(CORPUS)}
    )
    assert created.status_code == 201, created.text
    listed = (await client.get("/admin/api/connectors")).json()
    assert listed[0]["name"] == "docs" and listed[0]["last_run"] is None

    run = await client.post("/admin/api/connectors/docs/ingest")
    assert run.status_code == 200, run.text
    body = run.json()
    assert body["documents"] == 6 and body["chunks"] >= 6 and body["error"] is None
    assert runtime.index.size == body["chunks"]
    listed = (await client.get("/admin/api/connectors")).json()
    assert listed[0]["last_run"]["documents"] == 6

    again = (await client.post("/admin/api/connectors/docs/ingest")).json()
    assert again["skipped_unchanged"] == 6 and again["documents"] == 0

    deleted = await client.delete("/admin/api/connectors/docs")
    assert deleted.status_code == 204
    assert (await client.get("/admin/api/connectors")).json() == []


async def test_ingest_unknown_connector_is_404_and_bad_config_is_422(
    client: httpx.AsyncClient,
) -> None:
    assert (await client.post("/admin/api/connectors/nope/ingest")).status_code == 404
    bad = await client.post("/admin/api/connectors", json={"name": "x", "kind": "filesystem"})
    assert bad.status_code == 422
    dup = await client.post(
        "/admin/api/connectors", json={"name": "docs", "kind": "filesystem", "root": str(CORPUS)}
    )
    assert dup.status_code == 201
    dup2 = await client.post(
        "/admin/api/connectors", json={"name": "docs", "kind": "filesystem", "root": str(CORPUS)}
    )
    assert dup2.status_code == 409


async def test_gdrive_ingest_without_token_records_error_without_secrets(
    client: httpx.AsyncClient,
) -> None:
    await client.post("/admin/api/connectors", json={"name": "drive", "kind": "gdrive"})
    run = await client.post("/admin/api/connectors/drive/ingest")
    assert run.status_code == 200
    body = run.json()
    assert body["error"] and "token" in body["error"]
    run2 = await client.post(
        "/admin/api/connectors/drive/ingest",
        headers={f"{TOKEN_HEADER_PREFIX}gdrive": "ya29.secret-value"},
    )
    assert "ya29.secret-value" not in run2.text


async def test_index_health_reports_layout_and_cache(client: httpx.AsyncClient) -> None:
    await client.post(
        "/admin/api/connectors", json={"name": "docs", "kind": "filesystem", "root": str(CORPUS)}
    )
    await client.post("/admin/api/connectors/docs/ingest")
    health = (await client.get("/admin/api/index")).json()
    assert health["rows"] == 6
    assert health["dimension"] == 1024
    assert health["quantizer_fitted"] is True
    assert set(health["files"]) >= {"binary.bin", "int8.bin", "ids.json", "acl.npz", "quant.npz"}
    assert health["bytes_on_disk"] > 0
    assert health["bytes_per_chunk"] == health["bytes_on_disk"] / 6
    assert health["acl_principals"] >= 3
    assert health["rescore_multiplier"] == 64
    assert health["acl_cache"]["ttl_seconds"] == 60.0
    assert set(health["acl_cache"]) >= {"hits", "misses", "size", "ttl_seconds"}
