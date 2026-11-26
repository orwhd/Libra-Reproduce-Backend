from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_reproduce_runtime(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Ensure every test uses a clean temporary runtime & SQLite database."""
    runtime_dir = tmp_path_factory.mktemp("reproduce_runtime")
    monkeypatch.setenv("REPRODUCE_RUNTIME_DIR", str(runtime_dir))
    monkeypatch.delenv("REPRODUCE_DB_PATH", raising=False)
    os.environ["REPRODUCE_RUNTIME_DIR"] = str(runtime_dir)
    return runtime_dir
