from __future__ import annotations

import logging
from typing import Any

from netra.core.base import BaseStage

logger = logging.getLogger("netra.memory")


class MemoryManager:
    """
    Manages GPU VRAM allocation across pipeline stages.
    Supports sequential (swap) and concurrent (keep-all) loading strategies.
    """

    def __init__(self, vram_budget_mb: int, strategy: str = "sequential") -> None:
        self.vram_budget_mb = vram_budget_mb
        self.strategy = strategy
        self._loaded_stages: dict[str, BaseStage] = {}
        self._vram_used_mb: int = 0

    @property
    def vram_available_mb(self) -> int:
        return self.vram_budget_mb - self._vram_used_mb

    @property
    def vram_used_mb(self) -> int:
        return self._vram_used_mb

    def can_fit(self, stage: BaseStage) -> bool:
        return stage.vram_mb <= self.vram_available_mb

    def request_load(self, stage: BaseStage) -> None:
        if stage.stage_name in self._loaded_stages:
            logger.debug("Stage '%s' already loaded", stage.stage_name)
            return

        if self.strategy == "sequential":
            self._sequential_load(stage)
        elif self.strategy == "concurrent":
            self._concurrent_load(stage)
        elif self.strategy == "lazy":
            self._lazy_load(stage)
        else:
            raise ValueError(f"Unknown loading strategy: {self.strategy}")

    def request_unload(self, stage_name: str) -> None:
        stage = self._loaded_stages.pop(stage_name, None)
        if stage:
            stage.unload()
            self._vram_used_mb -= stage.vram_mb
            logger.info(
                "Unloaded '%s' — freed %d MB — available: %d MB",
                stage_name,
                stage.vram_mb,
                self.vram_available_mb,
            )

    def unload_all(self) -> None:
        for name in list(self._loaded_stages.keys()):
            self.request_unload(name)

    def get_status(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "vram_budget_mb": self.vram_budget_mb,
            "vram_used_mb": self._vram_used_mb,
            "vram_available_mb": self.vram_available_mb,
            "loaded_stages": {
                name: stage.vram_mb for name, stage in self._loaded_stages.items()
            },
        }

    def _sequential_load(self, stage: BaseStage) -> None:
        """Unload all other stages before loading this one."""
        if not self.can_fit(stage):
            logger.info("Freeing VRAM for '%s' (needs %d MB)", stage.stage_name, stage.vram_mb)
            self.unload_all()

        if not self.can_fit(stage):
            raise MemoryError(
                f"Cannot fit stage '{stage.stage_name}' ({stage.vram_mb} MB) "
                f"into available VRAM ({self.vram_available_mb} MB)"
            )

        self._do_load(stage)

    def _concurrent_load(self, stage: BaseStage) -> None:
        """Keep all stages loaded if they fit."""
        if not self.can_fit(stage):
            self._evict_lru(stage.vram_mb)

        if not self.can_fit(stage):
            raise MemoryError(
                f"Cannot fit stage '{stage.stage_name}' ({stage.vram_mb} MB) "
                f"even after eviction. Available: {self.vram_available_mb} MB"
            )

        self._do_load(stage)

    def _lazy_load(self, stage: BaseStage) -> None:
        """Load on demand, evict oldest if needed."""
        if not self.can_fit(stage):
            self._evict_lru(stage.vram_mb)
        self._do_load(stage)

    def _do_load(self, stage: BaseStage) -> None:
        stage.load()
        self._loaded_stages[stage.stage_name] = stage
        self._vram_used_mb += stage.vram_mb
        logger.info(
            "Loaded '%s' (%d MB) — used: %d / %d MB",
            stage.stage_name,
            stage.vram_mb,
            self._vram_used_mb,
            self.vram_budget_mb,
        )

    def _evict_lru(self, needed_mb: int) -> None:
        """Evict least-recently-used stages until we have enough VRAM."""
        names = list(self._loaded_stages.keys())
        for name in names:
            if self.vram_available_mb >= needed_mb:
                break
            self.request_unload(name)
