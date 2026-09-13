"""Shared fixtures for NETRA core-flow integration tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import requests

FIXTURES = Path(__file__).resolve().parent / "fixtures"
BASE_URL = os.environ.get("NETRA_BASE_URL", "http://127.0.0.1:5000")


@pytest.fixture(scope="session")
def base_url() -> str:
    return BASE_URL.rstrip("/")


@pytest.fixture(scope="session")
def api(base_url: str):
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    class API:
        def get(self, path: str, **kwargs):
            return session.get(f"{base_url}{path}", timeout=kwargs.pop("timeout", 60), **kwargs)

        def post(self, path: str, **kwargs):
            return session.post(f"{base_url}{path}", timeout=kwargs.pop("timeout", 180), **kwargs)

        def post_json(self, path: str, payload: dict, **kwargs):
            return session.post(
                f"{base_url}{path}",
                json=payload,
                timeout=kwargs.pop("timeout", 180),
                **kwargs,
            )

    health = session.get(f"{base_url}/api/health", timeout=10)
    if health.status_code != 200:
        pytest.skip(f"Backend not healthy at {base_url}: HTTP {health.status_code}")
    body = health.json()
    if body.get("status") != "ok" or not body.get("pipeline_ready", True):
        pytest.skip(f"Backend pipeline not ready: {body}")

    return API()


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def policy_txt(fixtures_dir: Path) -> Path:
    path = fixtures_dir / "sample_policy.txt"
    assert path.exists(), f"Missing fixture: {path}"
    return path


@pytest.fixture(scope="session")
def policy_txt_2(fixtures_dir: Path) -> Path:
    path = fixtures_dir / "sample_policy_2.txt"
    assert path.exists(), f"Missing fixture: {path}"
    return path


@pytest.fixture(scope="session")
def ocr_image(fixtures_dir: Path) -> Path:
    path = fixtures_dir / "sample_ocr_doc.png"
    assert path.exists(), f"Missing fixture: {path}"
    return path
