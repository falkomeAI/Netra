from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from netra.core.base import BaseStage
from netra.core.registry import StageRegistry
from netra.utils.faiss_lock import faiss_io_lock

logger = logging.getLogger("netra.embedding")

_BACKEND_ROOT = Path(__file__).resolve().parents[3]
FAISS_INDEX_DIR = _BACKEND_ROOT / "knowledge_base" / "vector_store"

_faiss_lock = faiss_io_lock


@StageRegistry.register("embedding", variant="me5_large")
@StageRegistry.register("embedding", variant="bge_small")
@StageRegistry.register("embedding")
class EmbeddingStage(BaseStage):
    """
    Generates dense vector embeddings from text using sentence-transformer models.
    Supports multilingual-e5-large (100+ languages including Indian) and
    bge-small-en (English only, lightweight fallback).
    Maintains a persistent FAISS index for the RAG knowledge base.
    """

    def _load_model(self) -> None:
        from sentence_transformers import SentenceTransformer

        model_id = self.config.get("model_id", "intfloat/multilingual-e5-large")
        self._model = SentenceTransformer(model_id, device=self.device)
        get_dim = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self._dim: int = get_dim()
        self._query_prefix = self.config.get("params", {}).get("query_prefix", "")
        self._passage_prefix = self.config.get("params", {}).get("passage_prefix", "")

        self._index_dir = Path(
            self.config.get("params", {}).get("index_dir", str(FAISS_INDEX_DIR))
        )
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._meta_path = self._index_dir / "metadata.json"
        self._index_path = self._index_dir / "index.faiss"
        self._load_or_create_index()

    def _load_or_create_index(self) -> None:
        import faiss

        with _faiss_lock:
            if self._index_path.exists():
                existing_index = faiss.read_index(str(self._index_path))
                if existing_index.d != self._dim:
                    logger.warning(
                        "FAISS index dimension mismatch (%d vs model %d) — rebuilding index",
                        existing_index.d, self._dim,
                    )
                    self._index_path.unlink()
                    self._index = faiss.IndexFlatIP(self._dim)
                    self._metadata = []
                else:
                    self._index = existing_index
                    self._metadata: list[dict] = (
                        json.loads(self._meta_path.read_text()) if self._meta_path.exists() else []
                    )
            else:
                self._index = faiss.IndexFlatIP(self._dim)
                self._metadata = []

    def _save_index(self) -> None:
        import faiss

        with _faiss_lock:
            self._save_index_unlocked()

    def _save_index_unlocked(self) -> None:
        import faiss

        faiss.write_index(self._index, str(self._index_path))
        self._meta_path.write_text(json.dumps(self._metadata, ensure_ascii=False, indent=2))

    def _unload_model(self) -> None:
        self._save_index()
        del self._model
        self._model = None
        self._index = None
        self._metadata = []

    def _run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        text = (
            inputs.get("ocr_output", {}).get("text", "")
            or inputs.get("text", "")
        )
        if not text:
            raise ValueError("No text available for embedding")

        chunks = self._chunk_text(text)
        prefixed = [f"{self._passage_prefix}{c}" for c in chunks] if self._passage_prefix else chunks
        embeddings = self._model.encode(
            prefixed, normalize_embeddings=True, show_progress_bar=False,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)

        doc_embedding = embeddings.mean(axis=0, keepdims=True)
        doc_embedding /= np.linalg.norm(doc_embedding, axis=1, keepdims=True) + 1e-9

        doc_id = hashlib.sha256(text.encode()).hexdigest()[:16]
        image_path = inputs.get("image_path", "")

        state = str(inputs.get("state") or "").strip()
        district = str(inputs.get("district") or "").strip()

        with _faiss_lock:
            self._index.add(embeddings)
            for i, chunk in enumerate(chunks):
                self._metadata.append({
                    "doc_id": doc_id,
                    "chunk_idx": i,
                    "text": chunk,
                    "source": str(image_path),
                    "state": state,
                    "district": district,
                })
            self._save_index_unlocked()

        return {
            "doc_id": doc_id,
            "embedding": doc_embedding[0].tolist(),
            "num_chunks": len(chunks),
            "index_total": self._index.ntotal,
        }

    @staticmethod
    def _chunk_text(text: str, max_len: int = 512, overlap: int = 64) -> list[str]:
        words = text.split()
        if len(words) <= max_len:
            return [text]
        chunks: list[str] = []
        start = 0
        while start < len(words):
            end = min(start + max_len, len(words))
            chunks.append(" ".join(words[start:end]))
            start += max_len - overlap
        return chunks


@StageRegistry.register("embedding", variant="ollama")
class OllamaEmbeddingStage(BaseStage):
    """
    Generates embeddings via Ollama's /api/embed endpoint.
    Uses whatever model is pulled in Ollama (e.g. qwen2.5:3b).
    Maintains the same persistent FAISS index as EmbeddingStage.
    """

    def _load_model(self) -> None:
        import faiss
        import urllib.request

        params = self.config.get("params", {})
        self._ollama_url = params.get("ollama_url", "http://localhost:11434")
        self._ollama_model = self.config.get("ollama_model", "qwen3-embedding:0.6b")

        try:
            req = urllib.request.Request(f"{self._ollama_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5):
                pass
        except Exception as e:
            logger.warning("Ollama not reachable at %s: %s", self._ollama_url, e)

        self._dim = self._detect_dim()

        self._index_dir = Path(
            params.get("index_dir", str(FAISS_INDEX_DIR))
        )
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._meta_path = self._index_dir / "metadata.json"
        self._index_path = self._index_dir / "index.faiss"

        with _faiss_lock:
            if self._index_path.exists():
                existing = faiss.read_index(str(self._index_path))

                if existing.d != self._dim:
                    if existing.ntotal > 0:
                        logger.warning(
                            "FAISS dim mismatch (%d vs Ollama %d) — existing index has %d vectors. "
                            "Keeping existing index. Re-run ingest to rebuild with Ollama embeddings.",
                            existing.d, self._dim, existing.ntotal,
                        )
                        self._dim = existing.d
                        self._index = existing
                        self._metadata = (
                            json.loads(self._meta_path.read_text()) if self._meta_path.exists() else []
                        )
                        self._dim_mismatch = True
                    else:
                        logger.info(
                            "FAISS dim mismatch (%d vs Ollama %d) — empty index, rebuilding.",
                            existing.d, self._dim,
                        )
                        self._index_path.unlink()
                        self._index = faiss.IndexFlatIP(self._dim)
                        self._metadata: list[dict] = []
                        self._dim_mismatch = False
                else:
                    self._index = existing
                    self._metadata = (
                        json.loads(self._meta_path.read_text()) if self._meta_path.exists() else []
                    )
                    self._dim_mismatch = False
            else:
                self._index = faiss.IndexFlatIP(self._dim)
                self._metadata = []
                self._dim_mismatch = False

        self._model = True

    def _detect_dim(self) -> int:
        probe = self._ollama_embed(["hello"])
        return len(probe[0])

    def _ollama_embed(self, texts: list[str]) -> list[list[float]]:
        import urllib.request
        import json as _json

        results = []
        for text in texts:
            payload = _json.dumps({"model": self._ollama_model, "prompt": text}).encode()
            req = urllib.request.Request(
                f"{self._ollama_url}/api/embeddings",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = _json.loads(resp.read())
            results.append(result.get("embedding", []))
        return results

    def _save_index_unlocked(self) -> None:
        import faiss
        faiss.write_index(self._index, str(self._index_path))
        self._meta_path.write_text(json.dumps(self._metadata, ensure_ascii=False, indent=2))

    def _unload_model(self) -> None:
        with _faiss_lock:
            self._save_index_unlocked()
        self._model = None
        self._index = None
        self._metadata = []

    def _run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        text = (
            inputs.get("ocr_output", {}).get("text", "")
            or inputs.get("text", "")
        )
        if not text:
            raise ValueError("No text available for embedding")

        chunks = EmbeddingStage._chunk_text(text)

        all_embeddings = []
        batch_size = 8
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            embs = self._ollama_embed(batch)
            all_embeddings.extend(embs)

        embeddings = np.asarray(all_embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        embeddings /= norms

        doc_embedding = embeddings.mean(axis=0, keepdims=True)
        doc_embedding /= np.linalg.norm(doc_embedding, axis=1, keepdims=True) + 1e-9

        doc_id = hashlib.sha256(text.encode()).hexdigest()[:16]
        image_path = inputs.get("image_path", "")
        state = str(inputs.get("state") or "").strip()
        district = str(inputs.get("district") or "").strip()

        with _faiss_lock:
            self._index.add(embeddings)
            for ci, chunk in enumerate(chunks):
                self._metadata.append({
                    "doc_id": doc_id,
                    "chunk_idx": ci,
                    "text": chunk,
                    "source": str(image_path),
                    "state": state,
                    "district": district,
                })
            self._save_index_unlocked()

        return {
            "doc_id": doc_id,
            "embedding": doc_embedding[0].tolist(),
            "num_chunks": len(chunks),
            "index_total": self._index.ntotal,
        }
