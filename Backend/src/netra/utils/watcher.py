from __future__ import annotations

import logging
import shutil
import time
import threading
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("netra.watcher")

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".pdf", ".txt"}


class DocumentWatcher:
    """
    Watches a directory for new documents and automatically ingests them
    into the NETRA knowledge base (embedding + knowledge graph).

    Supports: images (OCR → embed), PDFs (text extraction → embed), text files.
    Processed files are moved to ``processed/``; failures go to ``failed/``.
    """

    def __init__(
        self,
        watch_dir: str | Path,
        pipeline: Any,
        run_pipeline_fn: Callable | None = None,
        extract_pdf_fn: Callable | None = None,
        target_language: str = "hi",
        poll_interval: float = 10.0,
        on_complete: Callable[[Path, dict], None] | None = None,
    ) -> None:
        self.watch_dir = Path(watch_dir)
        self.processed_dir = self.watch_dir / "processed"
        self.failed_dir = self.watch_dir / "failed"
        self.pipeline = pipeline
        self.run_pipeline_fn = run_pipeline_fn
        self.extract_pdf_fn = extract_pdf_fn
        self.target_language = target_language
        self.poll_interval = poll_interval
        self.on_complete = on_complete

        self._running = False
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stats = {"total_processed": 0, "total_failed": 0, "last_file": None, "is_running": False}

        self.watch_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.failed_dir.mkdir(parents=True, exist_ok=True)

    @property
    def stats(self) -> dict:
        pending = len(self._scan_for_new_files())
        processed = len(list(self.processed_dir.iterdir())) if self.processed_dir.exists() else 0
        failed = len(list(self.failed_dir.iterdir())) if self.failed_dir.exists() else 0
        return {
            **self._stats,
            "pending": pending,
            "processed_count": processed,
            "failed_count": failed,
            "watch_dir": str(self.watch_dir),
        }

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._stats["is_running"] = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True, name="doc-watcher")
        self._thread.start()
        logger.info("Document watcher started: %s (poll every %.0fs)", self.watch_dir, self.poll_interval)

    def stop(self) -> None:
        self._running = False
        self._stats["is_running"] = False
        if hasattr(self, "_thread") and self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        logger.info("Document watcher stopped")

    def _watch_loop(self) -> None:
        while self._running:
            try:
                new_files = self._scan_for_new_files()
                for doc_path in new_files:
                    if not self._running:
                        break
                    logger.info("Auto-ingest: new document found — %s", doc_path.name)
                    self._process_document(doc_path)
            except Exception as e:
                logger.error("Watcher loop error: %s", e, exc_info=True)
            time.sleep(self.poll_interval)

    def _scan_for_new_files(self) -> list[Path]:
        files: list[Path] = []
        if not self.watch_dir.exists():
            return files
        for p in sorted(self.watch_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(p)
        return files

    def _process_document(self, doc_path: Path) -> bool:
        """Process a single document. Returns True on success, False on failure."""
        with self._lock:
            t0 = time.perf_counter()
            try:
                text = self._extract_text(doc_path)

                if not text or not text.strip():
                    raise ValueError(f"No text extracted from {doc_path.name}")

                inputs = {
                    "text": text,
                    "ocr_output": {"text": text, "lines": text.split("\n"), "num_lines": len(text.split("\n"))},
                    "target_language": self.target_language,
                    "source_language": self.target_language,
                }

                results = self.pipeline.run(inputs, stages=["embedding", "knowledge_graph"])
                elapsed_ms = (time.perf_counter() - t0) * 1000
                success = all(r.success for r in results.values())

                if success:
                    dest = self.processed_dir / doc_path.name
                    dest = self._unique_path(dest)
                    shutil.move(str(doc_path), str(dest))
                    self._stats["total_processed"] += 1
                    self._stats["last_file"] = doc_path.name
                    emb = results.get("embedding")
                    chunks = emb.data.get("num_chunks", 0) if emb and emb.data else 0
                    logger.info("Auto-ingest OK: %s → %d chunks in %.0f ms", doc_path.name, chunks, elapsed_ms)
                    if self.on_complete:
                        self.on_complete(doc_path, {"success": True, "elapsed_ms": elapsed_ms})
                    return True
                else:
                    failed_stages = [n for n, r in results.items() if not r.success]
                    errors = {n: r.error for n, r in results.items() if not r.success}
                    dest = self.failed_dir / doc_path.name
                    dest = self._unique_path(dest)
                    shutil.move(str(doc_path), str(dest))
                    self._stats["total_failed"] += 1
                    logger.warning("Auto-ingest partial fail: %s (stages: %s, errors: %s)", doc_path.name, failed_stages, errors)
                    if self.on_complete:
                        self.on_complete(doc_path, {"success": False, "elapsed_ms": elapsed_ms})
                    return False

            except Exception as e:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                logger.error("Auto-ingest failed: %s (%.0f ms): %s", doc_path.name, elapsed_ms, e)
                self._stats["total_failed"] += 1
                dest = self.failed_dir / doc_path.name
                dest = self._unique_path(dest)
                if doc_path.exists():
                    shutil.move(str(doc_path), str(dest))
                return False

    def _extract_text(self, doc_path: Path) -> str:
        ext = doc_path.suffix.lower()

        if ext == ".txt":
            return doc_path.read_text(encoding="utf-8", errors="replace")

        if ext == ".pdf":
            if self.extract_pdf_fn:
                return self.extract_pdf_fn(str(doc_path))
            try:
                import pdfplumber
                with pdfplumber.open(str(doc_path)) as pdf:
                    return "\n".join(page.extract_text() or "" for page in pdf.pages)
            except ImportError:
                pass
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(str(doc_path))
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            except ImportError:
                pass
            return ""

        if ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
            if self.run_pipeline_fn:
                ocr_inputs = {"image_path": str(doc_path), "source_language": self.target_language, "target_language": self.target_language}
                ocr_result = self.run_pipeline_fn(
                    ocr_inputs,
                    skip_stages=["retrieval", "llm", "nmt", "tts", "embedding", "knowledge_graph"]
                )
                ocr_stage = ocr_result.get("stages", {}).get("ocr", {})
                return ocr_stage.get("data", {}).get("text", "") if ocr_stage.get("success") else ""
            return ""

        return ""

    @staticmethod
    def _unique_path(path: Path) -> Path:
        if not path.exists():
            return path
        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        i = 1
        while True:
            candidate = parent / f"{stem}_{i}{suffix}"
            if not candidate.exists():
                return candidate
            i += 1

    def scan_and_ingest_all(self) -> dict:
        """Manually trigger ingestion of all pending files."""
        files = self._scan_for_new_files()
        results = {"processed": 0, "failed": 0, "files": []}
        for doc_path in files:
            success = self._process_document(doc_path)
            if success:
                results["processed"] += 1
                results["files"].append({"name": doc_path.name, "status": "ok"})
            else:
                results["failed"] += 1
                results["files"].append({"name": doc_path.name, "status": "error"})
        return results
