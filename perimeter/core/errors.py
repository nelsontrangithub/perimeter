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
