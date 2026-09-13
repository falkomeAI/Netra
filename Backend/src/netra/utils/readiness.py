"""Readiness probes for deploy health/ready endpoints."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any


def check_ollama(url: str = "http://localhost:11434", timeout: float = 2.0) -> dict[str, Any]:
    try:
        req = urllib.request.Request(f"{url.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        models = [m.get("name", "") for m in data.get("models", [])]
        return {"ok": True, "models": models}
    except Exception as e:
        return {"ok": False, "error": str(e), "models": []}


def check_faiss_index(index_dir: Path) -> dict[str, Any]:
    index_path = Path(index_dir) / "index.faiss"
    meta_path = Path(index_dir) / "metadata.json"
    exists = index_path.exists()
    chunks = 0
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            chunks = len(meta) if isinstance(meta, list) else 0
        except Exception:
            chunks = 0
    return {
        "ok": exists,
        "index_path": str(index_path),
        "chunks": chunks,
    }


def build_readiness(
    pipeline_ready: bool,
    faiss_dir: Path,
    ollama_url: str = "http://localhost:11434",
    require_ollama: bool = True,
    require_faiss: bool = False,
) -> dict[str, Any]:
    """
    Build a readiness report.

    Ready means pipeline is loaded. Ollama/FAISS are reported as checks;
    require_* flags control whether they block overall readiness.
    """
    ollama = check_ollama(ollama_url)
    faiss = check_faiss_index(faiss_dir)

    checks = {
        "pipeline": {"ok": bool(pipeline_ready)},
        "ollama": ollama,
        "faiss": faiss,
    }
    ready = bool(pipeline_ready)
    if require_ollama:
        ready = ready and bool(ollama.get("ok"))
    if require_faiss:
        ready = ready and bool(faiss.get("ok"))

    return {
        "ready": ready,
        "status": "ready" if ready else "not_ready",
        "checks": checks,
    }
