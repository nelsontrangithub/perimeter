from __future__ import annotations

import json
from pathlib import Path

import pytest

from perimeter.connectors.base import documents_from
from perimeter.connectors.filesystem import ACL_FILENAME, FilesystemConnector
from perimeter.core.acl import AccessPolicy, Deny, Grant
from perimeter.core.errors import ConnectorError
from perimeter.core.principal import EVERYONE, PrincipalId


def _write(root: Path, rel: str, text: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def _acl(root: Path, rel_dir: str, grants: list[str], denies: list[str] | None = None) -> None:
    body = {"grants": grants, "denies": denies or []}
    _write(root, f"{rel_dir}/{ACL_FILENAME}".lstrip("/"), json.dumps(body))


def test_enumerates_supported_files_recursively_sorted(tmp_path: Path) -> None:
    _write(tmp_path, "b/two.md", "two")
    _write(tmp_path, "a/one.txt", "one")
    _write(tmp_path, "a/skip.png", "binary")
    _acl(tmp_path, "", ["everyone"])
    refs = list(FilesystemConnector(tmp_path).enumerate())
    assert [r.uri for r in refs] == [
        f"file://{tmp_path / 'a/one.txt'}",
        f"file://{tmp_path / 'b/two.md'}",
    ]
    assert refs[0].title == "one.txt"
    assert refs[0].version is not None


def test_fetch_returns_text_and_rejects_paths_outside_root(tmp_path: Path) -> None:
    _write(tmp_path, "doc.md", "hello")
    outside = tmp_path.parent / "outside.md"
    outside.write_text("nope")
    conn = FilesystemConnector(tmp_path)
    assert conn.fetch(next(conn.enumerate())) == "hello"
    from perimeter.core.document import SourceRef

    with pytest.raises(ConnectorError):
        conn.fetch(SourceRef("filesystem", f"file://{outside}", "outside"))


def test_acl_comes_from_nearest_ancestor_sidecar(tmp_path: Path) -> None:
    _acl(tmp_path, "", ["everyone"])
    _acl(tmp_path, "eng", ["eng"], ["contractors"])
    _write(tmp_path, "public.md", "p")
    _write(tmp_path, "eng/design.md", "d")
    _write(tmp_path, "eng/sub/deep.md", "x")
    conn = FilesystemConnector(tmp_path)
    by_title = {r.title: r for r in conn.enumerate()}
    assert conn.acl_for(by_title["public.md"]) == AccessPolicy.from_rules([Grant(EVERYONE)])
    eng = AccessPolicy.from_rules([Grant(PrincipalId("eng")), Deny(PrincipalId("contractors"))])
    assert conn.acl_for(by_title["design.md"]) == eng
    assert conn.acl_for(by_title["deep.md"]) == eng


def test_per_file_sidecar_overrides_directory_acl(tmp_path: Path) -> None:
    _acl(tmp_path, "", ["everyone"])
    _write(tmp_path, "secret.md", "s")
    _write(tmp_path, "secret.md.acl.json", json.dumps({"grants": ["alice"]}))
    conn = FilesystemConnector(tmp_path)
    ref = next(r for r in conn.enumerate() if r.title == "secret.md")
    assert conn.acl_for(ref) == AccessPolicy.from_rules([Grant(PrincipalId("alice"))])


def test_missing_acl_is_nobody_and_malformed_acl_raises(tmp_path: Path) -> None:
    _write(tmp_path, "orphan.md", "o")
    conn = FilesystemConnector(tmp_path)
    ref = next(conn.enumerate())
    assert conn.acl_for(ref) == AccessPolicy.nobody()
    _write(tmp_path, ACL_FILENAME, "{not json")
    with pytest.raises(ConnectorError):
        conn.acl_for(ref)


def test_acl_sidecar_files_are_not_enumerated_as_documents(tmp_path: Path) -> None:
    _acl(tmp_path, "", ["everyone"])
    _write(tmp_path, "a.md", "a")
    _write(tmp_path, "a.md.acl.json", json.dumps({"grants": ["alice"]}))
    titles = [r.title for r in FilesystemConnector(tmp_path).enumerate()]
    assert titles == ["a.md"]


def test_documents_from_filesystem_end_to_end(tmp_path: Path) -> None:
    _acl(tmp_path, "", ["everyone"])
    _write(tmp_path, "a.md", "alpha")
    docs = list(documents_from(FilesystemConnector(tmp_path)))
    assert docs[0].id == f"filesystem:{(tmp_path / 'a.md').as_uri()}"
    assert docs[0].policy == AccessPolicy.public()
    assert docs[0].text == "alpha"
