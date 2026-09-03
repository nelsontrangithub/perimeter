"""Permission simulator: preview the corpus as any principal."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from perimeter.server.http import build_app
from perimeter.server.settings import Settings
from perimeter.server.wiring import Runtime, build_runtime

pytestmark = pytest.mark.anyio

CORPUS = Path(__file__).resolve().parents[2] / "examples" / "corpus"
GROUPS = Path(__file__).resolve().parents[2] / "examples" / "groups.json"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def runtime(tmp_path: Path) -> Runtime:
    rt = build_runtime(Settings(data_dir=tmp_path / "data", groups_file=GROUPS))
    from perimeter.connectors.base import documents_from
    from perimeter.connectors.filesystem import FilesystemConnector

    rt.ingestor.ingest(documents_from(FilesystemConnector(CORPUS)))
    rt.index.flush()
    return rt


@pytest.fixture
async def client(runtime: Runtime) -> AsyncIterator[httpx.AsyncClient]:
    app = build_app(runtime)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000"
        ) as c:
            yield c


async def test_simulate_explains_every_document_for_a_principal(client: httpx.AsyncClient) -> None:
    body = {
        "principal": "carol@example.com",
        "groups": ["sre", "contractors"],
        "query": None,
        "k": 5,
    }
    response = await client.post("/admin/api/simulate", json=body)
    assert response.status_code == 200, response.text
    sim = response.json()
    assert sim["principal"] == "carol@example.com"
    assert set(sim["effective_principals"]) >= {
        "carol@example.com",
        "sre",
        "eng",
        "staff",
        "contractors",
        "everyone",
    }
    by_title = {d["title"]: d for d in sim["documents"]}
    assert sim["total"] == 6 and len(by_title) == 6
    assert (
        by_title["onboarding.md"]["visible"]
        and by_title["onboarding.md"]["reason"] == "granted via everyone"
    )
    assert not by_title["index-design.md"]["visible"]
    assert by_title["index-design.md"]["reason"] == "denied via contractors"
    assert by_title["oncall-runbook.md"]["visible"]  # per-file sidecar: eng, sre; no deny
    assert not by_title["q3-forecast.md"]["visible"]
    assert by_title["q3-forecast.md"]["reason"] == "no matching grant"
    assert sim["visible_count"] == 3
    assert sim["results"] is None


async def test_simulate_with_query_returns_scoped_results_only(client: httpx.AsyncClient) -> None:
    body = {"principal": "bob", "groups": ["finance"], "query": "forecast revenue", "k": 10}
    sim = (await client.post("/admin/api/simulate", json=body)).json()
    assert sim["results"] is not None
    docs = {r["citation"]["title"] for r in sim["results"]["results"]}
    assert "q3-forecast.md" in docs
    assert "index-design.md" not in docs
    assert sim["results"]["requested_k"] == 10


async def test_simulate_rejects_bad_identity_and_never_widens(client: httpx.AsyncClient) -> None:
    bad = await client.post(
        "/admin/api/simulate", json={"principal": "everyone", "groups": [], "k": 5}
    )
    assert bad.status_code == 422
    empty = await client.post("/admin/api/simulate", json={"principal": "", "groups": [], "k": 5})
    assert empty.status_code == 422


async def test_simulate_uses_the_real_resolver_including_nested_groups(
    client: httpx.AsyncClient,
) -> None:
    sim = (
        await client.post(
            "/admin/api/simulate", json={"principal": "dana", "groups": ["sre"], "k": 5}
        )
    ).json()
    assert "eng" in sim["effective_principals"] and "staff" in sim["effective_principals"]
    by_title = {d["title"]: d for d in sim["documents"]}
    assert (
        by_title["index-design.md"]["visible"]
        and by_title["index-design.md"]["reason"] == "granted via eng"
    )
