from __future__ import annotations

from pathlib import Path

import pytest

from perimeter.core.errors import InvalidRequestError
from perimeter.server.settings import Settings


def test_defaults_from_empty_environment(tmp_path: Path) -> None:
    s = Settings.from_env({}, default_data_dir=tmp_path)
    assert s.data_dir == tmp_path
    assert s.cohere_api_key is None
    assert s.database_url is None
    assert s.acl_ttl_seconds == 60.0
    assert s.embedding_dimension == 1024
    assert s.allowed_hosts == ("127.0.0.1:*", "localhost:*", "[::1]:*")


def test_values_are_read_from_environment(tmp_path: Path) -> None:
    s = Settings.from_env(
        {
            "PERIMETER_DATA_DIR": str(tmp_path / "x"),
            "COHERE_API_KEY": "k",
            "PERIMETER_DATABASE_URL": "postgresql://u@h/db",
            "PERIMETER_ACL_TTL_SECONDS": "5",
            "PERIMETER_ALLOWED_HOSTS": "perimeter.internal:*, *.corp",
        },
        default_data_dir=tmp_path,
    )
    assert s.data_dir == tmp_path / "x"
    assert s.cohere_api_key == "k"
    assert s.database_url == "postgresql://u@h/db"
    assert s.acl_ttl_seconds == 5.0
    assert s.allowed_hosts == ("perimeter.internal:*", "*.corp")


def test_invalid_ttl_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(InvalidRequestError):
        Settings.from_env({"PERIMETER_ACL_TTL_SECONDS": "-1"}, default_data_dir=tmp_path)


def test_repr_hides_api_key(tmp_path: Path) -> None:
    s = Settings.from_env({"COHERE_API_KEY": "sekret"}, default_data_dir=tmp_path)
    assert "sekret" not in repr(s)
