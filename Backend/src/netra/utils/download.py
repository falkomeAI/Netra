from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("netra.download")


class ModelDownloader:
    """
    Download models from HuggingFace Hub with progress tracking.
    Respects the configured models_dir and caches downloads.
    """

    def __init__(self, models_dir: str | Path, cache_dir: str | Path | None = None) -> None:
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def download(
        self,
        model_id: str,
        revision: str | None = None,
        allow_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
    ) -> Path:
        from huggingface_hub import snapshot_download

        local_name = model_id.replace("/", "--")
        local_dir = self.models_dir / local_name

        if local_dir.exists() and any(local_dir.iterdir()):
            logger.info("Model already downloaded: %s", local_dir)
            return local_dir

        logger.info("Downloading model: %s", model_id)

        kwargs: dict[str, Any] = {
            "repo_id": model_id,
            "local_dir": str(local_dir),
        }
        if revision:
            kwargs["revision"] = revision
        if self.cache_dir:
            kwargs["cache_dir"] = str(self.cache_dir)
        if allow_patterns:
            kwargs["allow_patterns"] = allow_patterns
        if ignore_patterns:
            kwargs["ignore_patterns"] = ignore_patterns

        snapshot_download(**kwargs)
        logger.info("Download complete: %s -> %s", model_id, local_dir)
        return local_dir

    def download_from_config(self, model_config: dict) -> Path:
        model_id = model_config.get("model_id")
        if not model_id:
            raise ValueError("model_id not found in config")
        return self.download(model_id)

    def list_downloaded(self) -> list[str]:
        if not self.models_dir.exists():
            return []
        return [d.name for d in self.models_dir.iterdir() if d.is_dir()]

    def get_model_path(self, model_id: str) -> Path | None:
        local_name = model_id.replace("/", "--")
        local_dir = self.models_dir / local_name
        return local_dir if local_dir.exists() else None
