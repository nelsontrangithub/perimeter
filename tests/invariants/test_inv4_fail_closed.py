"""INV-4: an empty permitted set returns an empty result. It never falls back to
unfiltered search. Fail closed, always."""

from __future__ import annotations

import pytest

from perimeter.core.acl import AccessPolicy, PermissionSet
from perimeter.core.principal import Principal, PrincipalId
from perimeter.core.query import RetrievalRequest
from perimeter.pipeline.retrieve import Retriever
from tests.pipeline.fakes import StaticResolver, build_corpus, raw

pytestmark = pytest.mark.invariant

PUBLIC_DOCS = [raw(f"p{i}", f"apples {i} apples", AccessPolicy.public()) for i in range(5)]


class EmptyResolver:
    def resolve(self, principal: Principal) -> PermissionSet:
        return PermissionSet.empty()


@pytest.mark.parametrize(
    "resolver", [EmptyResolver(), StaticResolver(fail=True)], ids=["empty", "error"]
)
def test_inv4_empty_permitted_set_yields_empty_result(
    resolver: EmptyResolver | StaticResolver,
) -> None:
    store, index, embedder = build_corpus(PUBLIC_DOCS)
    retriever = Retriever(
        resolver=resolver, embedder=embedder, index=index, store=store, reranker=None
    )
    request = RetrievalRequest(principal=Principal(id=PrincipalId("anyone")), query="apples", k=5)
    result = retriever.retrieve(request)
    assert result.is_empty
    assert result.requested_k == 5
    assert index.search_calls == [], "the index must not be consulted at all"
    assert embedder.query_calls == 0, "no embedding call is spent on a caller with no permissions"
