"""Filesystem connector: text files under a root, ACLs from JSON sidecars.

ACL resolution for ``root/eng/design.md``, first match wins:

1. ``root/eng/design.md.acl.json``       (per-file sidecar)
2. ``root/eng/.perimeter-acl.json``      (nearest ancestor directory, up to root)
3. nobody                                (no sidecar anywhere: readable by no one)

Sidecar shape: ``{"grants": ["everyone", "eng"], "denies": ["contractors"]}``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from perimeter.core.acl import AccessPolicy, Deny, Grant
from perimeter.core.document import SourceRef
from perimeter.core.errors import ConnectorError, InvalidPrincipalError
from perimeter.core.principal import EVERYONE, PrincipalId, parse_principal_id

ACL_FILENAME = ".perimeter-acl.json"
SIDECAR_SUFFIX = ".acl.json"
DEFAULT_EXTENSIONS = (".md", ".txt", ".markdown", ".rst")


class FilesystemConnector:
    name = "filesystem"

    def __init__(self, root: Path, *, extensions: tuple[str, ...] = DEFAULT_EXTENSIONS) -> None:
        self._root = root.resolve()
        self._extensions = tuple(e.lower() for e in extensions)

    @property
    def root(self) -> Path:
        return self._root

    def enumerate(self) -> Iterator[SourceRef]:
        if not self._root.is_dir():
            raise ConnectorError("filesystem root is not a directory")
        for path in sorted(self._root.rglob("*")):
            if not path.is_file() or path.name.endswith(SIDECAR_SUFFIX):
                continue
            if path.suffix.lower() not in self._extensions:
                continue
            try:
                mtime = path.stat().st_mtime_ns
            except OSError:
                continue
            yield SourceRef(
                connector=self.name, uri=path.as_uri(), title=path.name, version=str(mtime)
            )

    def _path_for(self, ref: SourceRef) -> Path:
        if not ref.uri.startswith("file://"):
            raise ConnectorError("not a file:// uri")
        path = Path(ref.uri[len("file://") :]).resolve()
        if self._root not in path.parents:
            raise ConnectorError("path escapes the connector root")
        return path

    def fetch(self, ref: SourceRef) -> str:
        path = self._path_for(ref)
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise ConnectorError(f"cannot read file ({type(exc).__name__})") from None

    def acl_for(self, ref: SourceRef) -> AccessPolicy:
        path = self._path_for(ref)
        candidates = [path.with_name(path.name + SIDECAR_SUFFIX)]
        directory = path.parent
        while True:
            candidates.append(directory / ACL_FILENAME)
            if directory == self._root:
                break
            directory = directory.parent
        for candidate in candidates:
            if candidate.is_file():
                return _parse_acl(candidate)
        return AccessPolicy.nobody()


def _principal(raw: object) -> PrincipalId:
    if not isinstance(raw, str):
        raise ConnectorError("acl entries must be strings")
    if raw == EVERYONE:
        return EVERYONE
    try:
        return parse_principal_id(raw)
    except InvalidPrincipalError as exc:
        raise ConnectorError(f"acl entry invalid: {exc}") from None


def _parse_acl(path: Path) -> AccessPolicy:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConnectorError(f"acl sidecar unreadable ({type(exc).__name__})") from None
    if not isinstance(data, dict):
        raise ConnectorError("acl sidecar must be an object")
    grants = data.get("grants", [])
    denies = data.get("denies", [])
    if not isinstance(grants, list) or not isinstance(denies, list):
        raise ConnectorError("acl grants/denies must be lists")
    rules: list[Grant | Deny] = [Grant(_principal(g)) for g in grants]
    rules.extend(Deny(_principal(d)) for d in denies)
    return AccessPolicy.from_rules(rules)
