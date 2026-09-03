"""Typed domain errors.

Every error raised at a public boundary of Perimeter is a subclass of
:class:`PerimeterError`. Callers catch the specific type they can handle; nothing
in the codebase catches a bare ``Exception`` and continues.

Messages are safe to log: they never contain document text, connector tokens,
or a caller's full permission set.
"""

from __future__ import annotations


class PerimeterError(Exception):
    """Base class for all Perimeter domain errors."""


class InvalidPrincipalError(PerimeterError):
    """A principal or group identifier is malformed or reserved."""


class InvalidDocumentError(PerimeterError):
    """A document, chunk, or source reference is malformed."""


class InvalidRequestError(PerimeterError):
    """A retrieval request is malformed (bad k, blank or oversized query)."""


class EmbeddingError(PerimeterError):
    """The embedding model failed. Never includes the text that was being embedded."""


class RerankError(PerimeterError):
    """The reranker failed. Never includes candidate text."""


class VectorIndexError(PerimeterError):
    """The vector index is missing, corrupt, or was asked something it cannot do."""


class StoreError(PerimeterError):
    """The document store failed."""


class AclResolutionError(PerimeterError):
    """Permissions could not be resolved. Callers must treat this as an empty set."""


class AuthError(PerimeterError):
    """The caller's forwarded identity is missing or malformed. Never echoes header values."""


class ConnectorError(PerimeterError):
    """A connector could not enumerate, fetch, or read ACLs. Never includes tokens."""
