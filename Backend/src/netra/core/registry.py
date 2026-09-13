from __future__ import annotations

import logging
from typing import Type

from netra.core.base import BaseStage

logger = logging.getLogger("netra.registry")


class StageRegistry:
    """
    Registry mapping stage names to their implementation classes.
    Stages register themselves via the @register decorator or manual call.
    """

    _registry: dict[str, dict[str, Type[BaseStage]]] = {}

    @classmethod
    def register(cls, stage_name: str, variant: str = "default"):
        """Decorator to register a stage implementation."""

        def decorator(stage_cls: Type[BaseStage]) -> Type[BaseStage]:
            if stage_name not in cls._registry:
                cls._registry[stage_name] = {}
            cls._registry[stage_name][variant] = stage_cls
            logger.debug("Registered %s/%s -> %s", stage_name, variant, stage_cls.__name__)
            return stage_cls

        return decorator

    @classmethod
    def get(cls, stage_name: str, variant: str = "default") -> Type[BaseStage]:
        stage_variants = cls._registry.get(stage_name)
        if not stage_variants:
            raise KeyError(
                f"No implementations registered for stage '{stage_name}'. "
                f"Available stages: {list(cls._registry.keys())}"
            )
        impl = stage_variants.get(variant)
        if not impl:
            raise KeyError(
                f"Variant '{variant}' not found for stage '{stage_name}'. "
                f"Available variants: {list(stage_variants.keys())}"
            )
        return impl

    @classmethod
    def list_stages(cls) -> dict[str, list[str]]:
        return {stage: list(variants.keys()) for stage, variants in cls._registry.items()}

    @classmethod
    def create(
        cls,
        stage_name: str,
        config: dict,
        device: str = "cuda:0",
        variant: str = "default",
    ) -> BaseStage:
        stage_cls = cls.get(stage_name, variant)
        return stage_cls(stage_name=stage_name, config=config, device=device)
