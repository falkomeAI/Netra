from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from netra.core.base import BaseStage
from netra.core.registry import StageRegistry
from netra.utils.faiss_lock import faiss_io_lock
from netra.utils.location import chunk_matches_location

logger = logging.getLogger("netra.retrieval")

_BACKEND_ROOT = Path(__file__).resolve().parents[3]
FAISS_INDEX_DIR = _BACKEND_ROOT / "knowledge_base" / "vector_store"

_faiss_read_lock = faiss_io_lock


@StageRegistry.register("retrieval", variant="faiss_bge")
@StageRegistry.register("retrieval", variant="faiss")
@StageRegistry.register("retrieval")
class FAISSRetrievalStage(BaseStage):
    """
    RAG retrieval stage.  Searches the FAISS vector store built by
    the EmbeddingStage for the most similar document chunks and returns
    them as context for the LLM.
    """

    def _load_model(self) -> None:
        from sentence_transformers import SentenceTransformer

        model_id = self.config.get("model_id", "intfloat/multilingual-e5-large")
        self._model = SentenceTransformer(model_id, device=self.device)

        self._top_k: int = self.config.get("params", {}).get("top_k", 5)
        self._min_score: float = self.config.get("params", {}).get("min_score", 0.25)
        self._query_prefix = self.config.get("params", {}).get("query_prefix", "")

        self._index_dir = Path(
            self.config.get("params", {}).get("index_dir", str(FAISS_INDEX_DIR))
        )

    def _unload_model(self) -> None:
        del self._model
        self._model = None

    def _run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import faiss

        index_path = self._index_dir / "index.faiss"
        meta_path = self._index_dir / "metadata.json"

        if not index_path.exists():
            return {"context_chunks": [], "num_results": 0, "message": "No index yet"}

        with _faiss_read_lock:
            index = faiss.read_index(str(index_path))
            if index.ntotal == 0:
                return {"context_chunks": [], "num_results": 0, "message": "Index is empty"}

            metadata: list[dict] = (
                json.loads(meta_path.read_text()) if meta_path.exists() else []
            )

        query_text = (
            inputs.get("question", "")
            or inputs.get("ocr_output", {}).get("text", "")
            or inputs.get("text", "")
        )
        if not query_text:
            return {"context_chunks": [], "num_results": 0, "message": "No query text"}

        prefixed_query = f"{self._query_prefix}{query_text}" if self._query_prefix else query_text
        query_vec = self._model.encode(
            [prefixed_query], normalize_embeddings=True, show_progress_bar=False,
        )
        query_vec = np.asarray(query_vec, dtype=np.float32)

        # Over-fetch then filter by state/district so geo-scoping still has enough hits.
        fetch_k = min(max(self._top_k * 5, self._top_k), index.ntotal)
        scores, indices = index.search(query_vec, fetch_k)

        want_state = str(inputs.get("state") or "").strip()
        want_district = str(inputs.get("district") or "").strip()

        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < self._min_score:
                continue
            meta = metadata[idx] if idx < len(metadata) else {}
            if not chunk_matches_location(meta, want_state, want_district):
                continue
            results.append({
                "text": meta.get("text", ""),
                "score": float(score),
                "doc_id": meta.get("doc_id", ""),
                "source": meta.get("source", ""),
                "state": meta.get("state", ""),
                "district": meta.get("district", ""),
            })
            if len(results) >= self._top_k:
                break

        context_text = "\n---\n".join(r["text"] for r in results)

        return {
            "context_chunks": results,
            "context_text": context_text,
            "num_results": len(results),
            "filter_state": want_state,
            "filter_district": want_district,
        }


@StageRegistry.register("retrieval", variant="ollama")
class OllamaRetrievalStage(BaseStage):
    """
    RAG retrieval using Ollama embeddings (/api/embed) + FAISS search.
    Matches the OllamaEmbeddingStage for consistent vector space.
    """

    def _load_model(self) -> None:
        params = self.config.get("params", {})
        self._ollama_url = params.get("ollama_url", "http://localhost:11434")
        self._ollama_model = self.config.get("ollama_model", "qwen3-embedding:0.6b")
        self._top_k: int = params.get("top_k", 5)
        self._min_score: float = params.get("min_score", 0.25)

        self._index_dir = Path(
            params.get("index_dir", str(FAISS_INDEX_DIR))
        )
        self._model = True

    def _unload_model(self) -> None:
        self._model = None

    def _translate_if_needed(self, text: str) -> str:
        """Skip costly LLM translation — the embedding model handles multilingual queries."""
        return text

    def _ollama_embed(self, texts: list[str]) -> list[list[float]]:
        import urllib.request
        import json as _json

        results = []
        for text in texts:
            payload = _json.dumps({
                "model": self._ollama_model,
                "prompt": text,
                "keep_alive": "30m",
            }).encode()
            req = urllib.request.Request(
                f"{self._ollama_url}/api/embeddings",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = _json.loads(resp.read())
            results.append(result.get("embedding", []))
        return results

    def _run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import faiss

        index_path = self._index_dir / "index.faiss"
        meta_path = self._index_dir / "metadata.json"

        if not index_path.exists():
            return {"context_chunks": [], "num_results": 0, "message": "No index yet"}

        with _faiss_read_lock:
            index = faiss.read_index(str(index_path))
            if index.ntotal == 0:
                return {"context_chunks": [], "num_results": 0, "message": "Index is empty"}
            metadata: list[dict] = (
                json.loads(meta_path.read_text()) if meta_path.exists() else []
            )

        query_text = (
            inputs.get("question", "")
            or inputs.get("ocr_output", {}).get("text", "")
            or inputs.get("text", "")
        )
        if not query_text:
            return {"context_chunks": [], "num_results": 0, "message": "No query text"}

        query_for_embed = self._translate_if_needed(query_text)
        raw_vecs = self._ollama_embed([query_for_embed])
        query_vec = np.asarray(raw_vecs, dtype=np.float32)
        query_vec /= np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-9


        if index.d != query_vec.shape[1]:
            return {
                "context_chunks": [],
                "num_results": 0,
                "message": (
                    f"Dimension mismatch: index has dim={index.d} but Ollama "
                    f"embeddings are dim={query_vec.shape[1]}. "
                    f"Re-ingest documents with: python scripts/ingest_policies.py"
                ),
            }

        fetch_k = min(max(self._top_k * 5, self._top_k), index.ntotal)
        scores, indices = index.search(query_vec, fetch_k)

        want_state = str(inputs.get("state") or "").strip()
        want_district = str(inputs.get("district") or "").strip()

        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < self._min_score:
                continue
            meta = metadata[idx] if idx < len(metadata) else {}
            if not chunk_matches_location(meta, want_state, want_district):
                continue
            results.append({
                "text": meta.get("text", ""),
                "score": float(score),
                "doc_id": meta.get("doc_id", ""),
                "source": meta.get("source", ""),
                "state": meta.get("state", ""),
                "district": meta.get("district", ""),
            })
            if len(results) >= self._top_k:
                break

        context_text = "\n---\n".join(r["text"] for r in results)

        return {
            "context_chunks": results,
            "context_text": context_text,
            "num_results": len(results),
            "filter_state": want_state,
            "filter_district": want_district,
        }
