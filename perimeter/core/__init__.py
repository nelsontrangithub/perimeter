"""Pure domain layer.

Dependency rule: this package imports nothing but the Python standard library.
No FastAPI, no NumPy, no HTTP client, no database driver. All I/O is expressed
as ``typing.Protocol`` definitions in :mod:`perimeter.core.ports` and implemented
in :mod:`perimeter.adapters`. ``tests/test_architecture.py`` fails CI on any
violation.
"""
