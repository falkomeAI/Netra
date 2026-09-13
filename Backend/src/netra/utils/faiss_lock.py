"""Shared lock for FAISS index + metadata read/write (ingest + retrieval)."""

from __future__ import annotations

import threading

# One lock for all FAISS file/index access across embedding + retrieval stages.
faiss_io_lock = threading.RLock()
