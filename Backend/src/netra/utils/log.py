from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_format: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    log_file: str | None = None,
    max_bytes: int = 10_485_760,
    backup_count: int = 5,
) -> None:
    """Configure logging for the NETRA pipeline."""
    root = logging.getLogger("netra")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    if root.handlers:
        return

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    root.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        file_handler.setFormatter(logging.Formatter(log_format))
        root.addHandler(file_handler)

    root.info("Logging initialized — level=%s", level)
