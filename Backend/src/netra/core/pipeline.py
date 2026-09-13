from __future__ import annotations

import logging
import time
from typing import Any

from netra.config.settings import Settings
from netra.core.base import BaseStage, StageResult
from netra.core.memory import MemoryManager
from netra.core.registry import StageRegistry

logger = logging.getLogger("netra.pipeline")


class Pipeline:
    """
    Orchestrates the full NETRA pipeline.
    Loads stages from the registry, manages VRAM, and runs inference.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.memory = MemoryManager(
            vram_budget_mb=settings.get_available_vram_mb(),
            strategy=settings.loading_strategy,
        )
        self._stages: dict[str, BaseStage] = {}
        self._llm_fallback_stage: BaseStage | None = None
        self._build_stages()

    def _build_stages(self) -> None:
        is_cpu = (
            self.settings.device.target == "cpu"
            or self.settings.device.cuda_device < 0
        )
        self._device = "cpu" if is_cpu else f"cuda:{self.settings.device.cuda_device}"

        for stage_name in self.settings.active_stages:
            try:
                model_key = self.settings.model_defaults.get(stage_name)
                config = self.settings.get_model_config(stage_name, model_key)
                stage = StageRegistry.create(
                    stage_name=stage_name,
                    config=config,
                    device=self._device,
                    variant=model_key or "default",
                )
                self._stages[stage_name] = stage
                logger.info("Initialized stage: %s (model=%s)", stage_name, model_key)
            except (KeyError, ValueError) as e:
                logger.warning("Could not initialize stage '%s': %s", stage_name, e)

        LLM_FALLBACK_ORDER = ["qwen2_5_05b", "qwen2_5_3b", "smollm2"]
        if "llm" in self._stages:
            primary_key = self.settings.model_defaults.get("llm", "")
            for fallback_key in LLM_FALLBACK_ORDER:
                if fallback_key == primary_key:
                    continue
                try:
                    fb_config = self.settings.get_model_config("llm", fallback_key)
                    self._llm_fallback_stage = StageRegistry.create(
                        stage_name="llm",
                        config=fb_config,
                        device=self._device,
                        variant=fallback_key,
                    )
                    logger.info("LLM fallback ready: %s", fallback_key)
                    break
                except (KeyError, ValueError):
                    continue

    def _ensure_optional_stage(self, stage_name: str) -> bool:
        """Lazily initialize an optional stage on first use."""
        if stage_name in self._stages:
            return True
        if stage_name not in self.settings.optional_stages:
            return False
        try:
            model_key = self.settings.model_defaults.get(stage_name)
            config = self.settings.get_model_config(stage_name, model_key)
            stage = StageRegistry.create(
                stage_name=stage_name,
                config=config,
                device=self._device,
                variant=model_key or "default",
            )
            self._stages[stage_name] = stage
            logger.info("Lazy-loaded optional stage: %s (model=%s)", stage_name, model_key)
            return True
        except Exception as e:
            logger.warning("Could not initialize optional stage '%s': %s", stage_name, e)
            return False

    def run(
        self,
        inputs: dict[str, Any],
        stages: list[str] | None = None,
    ) -> dict[str, StageResult]:
        """
        Run the pipeline. Each stage receives the accumulated results
        from prior stages merged into the original inputs.
        """
        stage_names = stages or self.settings.active_stages
        results: dict[str, StageResult] = {}
        accumulated: dict[str, Any] = dict(inputs)

        t0 = time.perf_counter()
        logger.info("Pipeline run started — stages: %s", stage_names)

        CRITICAL_STAGES = {"ocr", "llm"}

        for stage_name in stage_names:
            stage = self._stages.get(stage_name)
            if stage is None:
                self._ensure_optional_stage(stage_name)
                stage = self._stages.get(stage_name)
            if stage is None:
                logger.warning("Stage '%s' not available, skipping", stage_name)
                results[stage_name] = StageResult(
                    stage_name=stage_name,
                    success=False,
                    error="Stage not initialized",
                )
                continue

            self.memory.request_load(stage)
            result = stage.run(accumulated)
            results[stage_name] = result

            if result.success and result.data is not None:
                accumulated[f"{stage_name}_output"] = result.data
            elif not result.success:
                logger.error("Stage '%s' failed: %s", stage_name, result.error)
                if stage_name in CRITICAL_STAGES:
                    if self._has_fallback(stage_name):
                        result = self._run_fallback(stage_name, accumulated)
                        results[stage_name] = result
                        if result.success and result.data is not None:
                            accumulated[f"{stage_name}_output"] = result.data
                        else:
                            break
                    else:
                        break

        total_ms = (time.perf_counter() - t0) * 1000
        logger.info("Pipeline completed in %.0f ms", total_ms)
        return results

    def run_stage(self, stage_name: str, inputs: dict[str, Any]) -> StageResult:
        stage = self._stages.get(stage_name)
        if stage is None:
            self._ensure_optional_stage(stage_name)
            stage = self._stages.get(stage_name)
        if stage is None:
            return StageResult(
                stage_name=stage_name,
                success=False,
                error=f"Stage '{stage_name}' not available",
            )
        self.memory.request_load(stage)
        return stage.run(inputs)

    def _has_fallback(self, stage_name: str) -> bool:
        fallback_map = {"ocr": "vlm", "llm": "llm_fallback"}
        fallback = fallback_map.get(stage_name)
        if fallback == "llm_fallback":
            return self._llm_fallback_stage is not None
        if fallback == "vlm":
            return self._ensure_optional_stage("vlm")
        return fallback is not None and fallback in self._stages

    def _run_fallback(self, stage_name: str, inputs: dict[str, Any]) -> StageResult:
        fallback_map = {"ocr": "vlm"}
        fallback_name = fallback_map.get(stage_name)

        if stage_name == "llm" and self._llm_fallback_stage is not None:
            logger.info("Running LLM fallback stage")
            self.memory.request_load(self._llm_fallback_stage)
            return self._llm_fallback_stage.run(inputs)

        if fallback_name:
            try:
                self._ensure_optional_stage(fallback_name)
                if fallback_name in self._stages:
                    logger.info("Running fallback: %s -> %s", stage_name, fallback_name)
                    fallback_stage = self._stages[fallback_name]
                    self.memory.request_load(fallback_stage)
                    return fallback_stage.run(inputs)
            except Exception as e:
                logger.error("Fallback %s also failed: %s", fallback_name, e)
        return StageResult(
            stage_name=stage_name,
            success=False,
            error="No fallback available",
        )

    def get_stage(self, stage_name: str) -> BaseStage | None:
        return self._stages.get(stage_name)

    def list_stages(self) -> dict[str, bool]:
        return {name: stage.is_loaded for name, stage in self._stages.items()}

    def shutdown(self) -> None:
        logger.info("Shutting down pipeline")
        self.memory.unload_all()
        self._stages.clear()

    def get_memory_status(self) -> dict[str, Any]:
        return self.memory.get_status()
