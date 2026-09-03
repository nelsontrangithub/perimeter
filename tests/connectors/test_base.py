from __future__ import annotations

from collections.abc import Iterator

from perimeter.connectors.base import Connector, documents_from
from perimeter.core.acl import AccessPolicy, Grant
from perimeter.core.document import SourceRef
from perimeter.core.errors import ConnectorError
from perimeter.core.principal import PrincipalId


class Fake:
    name = "fake"

    def __init__(self) -> None:
        self.refs = [
            SourceRef("fake", "fake://1", "One", version="a"),
            SourceRef("fake", "fake://2", "Two", version="b"),
            SourceRef("fake", "fake://3", "Three", version="c"),
        ]

    def enumerate(self) -> Iterator[SourceRef]:
        yield from self.refs

    def fetch(self, ref: SourceRef) -> str:
        if ref.uri.endswith("3"):
            raise ConnectorError("unreadable")
        return f"text of {ref.title}"

    def acl_for(self, ref: SourceRef) -> AccessPolicy:
        if ref.uri.endswith("2"):
            raise ConnectorError("acl unavailable")
        return AccessPolicy.from_rules([Grant(PrincipalId("alice"))])


def test_fake_satisfies_protocol() -> None:
    assert isinstance(Fake(), Connector)


def test_documents_from_yields_raw_documents_with_stable_ids() -> None:
    docs = list(documents_from(Fake()))
    assert [d.id for d in docs] == ["fake:fake://1", "fake:fake://2"]
    assert docs[0].text == "text of One"
    assert docs[0].source.version == "a"


def test_unreadable_acl_becomes_nobody_never_public() -> None:
    docs = {str(d.id): d for d in documents_from(Fake())}
    assert docs["fake:fake://2"].policy == AccessPolicy.nobody()


def test_unreadable_document_is_skipped_and_counted() -> None:
    skipped: list[SourceRef] = []
    docs = list(documents_from(Fake(), on_skip=skipped.append))
    assert len(docs) == 2
    assert [s.uri for s in skipped] == ["fake://3"]
