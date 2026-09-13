from __future__ import annotations

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageResult:
    """Standardised output from every pipeline stage."""

    stage_name: str
    success: bool
    data: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    latency_ms: float = 0.0

    def __bool__(self) -> bool:
        return self.success


class BaseStage(ABC):
    """
    Abstract base for every pipeline stage.

    Subclasses implement:
        _load_model  – load / download the model into memory
        _unload_model – free GPU memory
        _run – stage-specific inference
    """

    def __init__(self, stage_name: str, config: dict, device: str = "cuda:0") -> None:
        self.stage_name = stage_name
        self.config = config
        self.device = device
        self.logger = logging.getLogger(f"netra.{stage_name}")
        self._model: Any = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def vram_mb(self) -> int:
        return self.config.get("vram_mb", 0)

    @abstractmethod
    def _load_model(self) -> None:
        ...

    @abstractmethod
    def _unload_model(self) -> None:
        ...

    @abstractmethod
    def _run(self, inputs: dict[str, Any]) -> Any:
        ...

    def load(self) -> None:
        if self._is_loaded:
            self.logger.debug("Model already loaded, skipping")
            return
        self.logger.info("Loading model: %s", self.config.get("name", self.stage_name))
        t0 = time.perf_counter()
        self._load_model()
        self._is_loaded = True
        elapsed = (time.perf_counter() - t0) * 1000
        self.logger.info("Model loaded in %.0f ms", elapsed)

    def unload(self) -> None:
        if not self._is_loaded:
            return
        self.logger.info("Unloading model: %s", self.config.get("name", self.stage_name))
        self._unload_model()
        self._model = None
        self._is_loaded = False
        self._free_gpu_cache()

    def run(self, inputs: dict[str, Any]) -> StageResult:
        t0 = time.perf_counter()
        try:
            if not self._is_loaded:
                self.load()
            result = self._run(inputs)
            latency = (time.perf_counter() - t0) * 1000
            self.logger.info("Stage '%s' completed in %.0f ms", self.stage_name, latency)
            return StageResult(
                stage_name=self.stage_name,
                success=True,
                data=result,
                latency_ms=latency,
            )
        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000
            self.logger.error("Stage '%s' failed: %s", self.stage_name, e, exc_info=True)
            return StageResult(
                stage_name=self.stage_name,
                success=False,
                error=str(e),
                latency_ms=latency,
            )

    @staticmethod
    def _free_gpu_cache() -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "unloaded"
        return f"<{self.__class__.__name__} stage={self.stage_name!r} {status}>"
