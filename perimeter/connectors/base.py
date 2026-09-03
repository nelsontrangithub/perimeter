"""Connector protocol: enumerate, fetch, and read the ACL of source documents.

A connector produces :class:`RawDocument` values for the ingestor. The bridge
:func:`documents_from` applies the one rule every connector shares: a document
whose ACL cannot be read is ingested as readable by nobody. Never public.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from typing import Protocol, runtime_checkable

from perimeter.core.acl import AccessPolicy
from perimeter.core.document import DocumentId, SourceRef
from perimeter.core.errors import ConnectorError
from perimeter.pipeline.ingest import RawDocument

log = logging.getLogger(__name__)


@runtime_checkable
class Connector(Protocol):
    @property
    def name(self) -> str: ...

    def enumerate(self) -> Iterator[SourceRef]: ...

    def fetch(self, ref: SourceRef) -> str: ...

    def acl_for(self, ref: SourceRef) -> AccessPolicy: ...


def document_id_for(connector: Connector, ref: SourceRef) -> DocumentId:
    return DocumentId(f"{connector.name}:{ref.uri}")


def documents_from(
    connector: Connector, *, on_skip: Callable[[SourceRef], None] | None = None
) -> Iterator[RawDocument]:
    """Every readable document from ``connector``, ACL attached, unreadable ones skipped."""
    for ref in connector.enumerate():
        try:
            policy = connector.acl_for(ref)
        except ConnectorError as exc:
            log.warning("acl unreadable; ingesting as nobody: %s (%s)", ref.uri, type(exc).__name__)
            policy = AccessPolicy.nobody()
        try:
            text = connector.fetch(ref)
        except ConnectorError as exc:
            log.warning("document unreadable; skipped: %s (%s)", ref.uri, type(exc).__name__)
            if on_skip is not None:
                on_skip(ref)
            continue
        yield RawDocument(id=document_id_for(connector, ref), source=ref, policy=policy, text=text)
