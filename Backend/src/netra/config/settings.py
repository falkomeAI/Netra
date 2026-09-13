from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass
class DeviceConfig:
    target: str = "jetson_orin_nano"
    vram_budget_mb: int = 8192
    vram_reserved_mb: int = 2048
    cuda_device: int = 0
    enable_tensorrt: bool = True
    enable_cuda_graphs: bool = False
    memory_fraction: float = 0.85


@dataclass
class InferenceConfig:
    default_batch_size: int = 1
    max_input_length: int = 2048
    max_output_length: int = 512
    default_temperature: float = 0.7
    default_top_p: float = 0.9
    timeout_seconds: int = 30


@dataclass
class QuantizationConfig:
    default_method: str = "awq"
    default_bits: int = 4
    group_size: int = 128
    zero_point: bool = True


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    file: str = "logs/netra.log"
    max_bytes: int = 10_485_760
    backup_count: int = 5


@dataclass
class PathsConfig:
    models_dir: str = "models"
    cache_dir: str = "cache"
    output_dir: str = "output"
    temp_dir: str = "temp"
    logs_dir: str = "logs"
    knowledge_base_dir: str = "knowledge_base"
    watch_dir: str = "data/inbox"

    def resolve(self, project_root: Path) -> None:
        for attr in (
            "models_dir", "cache_dir", "output_dir", "temp_dir",
            "logs_dir", "knowledge_base_dir", "watch_dir",
        ):
            path = getattr(self, attr)
            if not os.path.isabs(path):
                setattr(self, attr, str(project_root / path))


@dataclass
class LanguagesConfig:
    supported: list[str] = field(default_factory=lambda: ["hi", "en"])
    default_source: str = "en"
    default_target: str = "hi"


class Settings:
    """Central configuration loaded from YAML files with env-var overrides."""

    def __init__(self, config_dir: str | Path | None = None) -> None:
        self.project_root = Path(__file__).resolve().parents[3]
        self.config_dir = Path(config_dir) if config_dir else self.project_root / "config"

        raw = self._load_configs()

        pipeline_cfg = raw.get("pipeline", {})
        self.pipeline_name: str = pipeline_cfg.get("name", "NETRA")
        self.pipeline_version: str = pipeline_cfg.get("version", "1.0.0")
        self.active_stages: list[str] = pipeline_cfg.get("stages", [])
        self.optional_stages: list[str] = pipeline_cfg.get("optional_stages", [])
        self.loading_strategy: str = pipeline_cfg.get("loading_strategy", "sequential")

        self.device = self._build_dataclass(DeviceConfig, raw.get("device", {}))
        self.inference = self._build_dataclass(InferenceConfig, raw.get("inference", {}))
        self.quantization = self._build_dataclass(QuantizationConfig, raw.get("quantization", {}))
        self.logging = self._build_dataclass(LoggingConfig, raw.get("logging", {}))
        self.paths = self._build_dataclass(PathsConfig, raw.get("paths", {}))
        self.paths.resolve(self.project_root)
        self.languages = self._build_dataclass(LanguagesConfig, raw.get("languages", {}))

        self.models: dict[str, Any] = raw.get("models", {})
        self.model_defaults: dict[str, str] = raw.get("defaults", {})
        self.prompts: dict[str, Any] = raw.get("prompts", {})

        self._apply_env_overrides()

    def _load_configs(self) -> dict:
        merged: dict = {}
        load_order = ["default.yaml", "models.yaml", "prompts.yaml", "bhashini.yaml"]

        for filename in load_order:
            filepath = self.config_dir / filename
            if filepath.exists():
                with open(filepath) as f:
                    data = yaml.safe_load(f) or {}
                merged = _deep_merge(merged, data)

        env_config = os.environ.get("NETRA_CONFIG")
        if env_config and Path(env_config).exists():
            device_yaml = Path(env_config)
        else:
            device_yaml = self._auto_detect_device_config()

        if device_yaml and device_yaml.exists():
            with open(device_yaml) as f:
                data = yaml.safe_load(f) or {}
            merged = _deep_merge(merged, data)

        target = merged.get("device", {}).get("target", "")
        is_gpu = target in ("cuda", "jetson_orin_nano", "jetson") or "jetson" in target
        if "jetson_defaults" in merged and is_gpu:
            jd = merged.pop("jetson_defaults", {})
            current_defaults = merged.get("defaults", {})
            merged["defaults"] = _deep_merge(current_defaults, jd)

        return merged

    def _auto_detect_device_config(self) -> Path | None:
        """Pick jetson.yaml or cpu.yaml based on hardware detection."""
        from netra.utils.device import _detect_jetson

        is_jetson, model = _detect_jetson()
        if is_jetson:
            chosen = self.config_dir / "jetson.yaml"
            if chosen.exists():
                return chosen

        chosen = self.config_dir / "cpu.yaml"
        if chosen.exists():
            return chosen
        return None

    @staticmethod
    def _build_dataclass(cls, data: dict):
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def _apply_env_overrides(self) -> None:
        env_map = {
            "NETRA_DEVICE_TARGET": ("device", "target"),
            "NETRA_CUDA_DEVICE": ("device", "cuda_device", int),
            "NETRA_VRAM_BUDGET_MB": ("device", "vram_budget_mb", int),
            "NETRA_LOG_LEVEL": ("logging", "level"),
            "NETRA_MODELS_DIR": ("paths", "models_dir"),
            "NETRA_LOADING_STRATEGY": (None, "loading_strategy"),
            "NETRA_DEFAULT_LLM": (None, "model_defaults.llm"),
            "NETRA_DEFAULT_OCR": (None, "model_defaults.ocr"),
            "NETRA_DEFAULT_TARGET_LANG": ("languages", "default_target"),
        }

        for env_key, target in env_map.items():
            value = os.environ.get(env_key)
            if value is None:
                continue

            if len(target) == 3:
                section, attr, cast = target
            else:
                section, attr = target
                cast = str

            value = cast(value)

            if section is None:
                if "." in attr:
                    parts = attr.split(".")
                    obj = getattr(self, parts[0])
                    obj[parts[1]] = value
                else:
                    setattr(self, attr, value)
            else:
                setattr(getattr(self, section), attr, value)

    def get_model_config(self, stage: str, model_key: str | None = None) -> dict:
        if model_key is None:
            model_key = self.model_defaults.get(stage)
        stage_models = self.models.get(stage, {})
        if model_key and model_key in stage_models:
            return stage_models[model_key]
        raise ValueError(
            f"Model '{model_key}' not found for stage '{stage}'. "
            f"Available: {list(stage_models.keys())}"
        )

    def get_prompt(self, prompt_key: str) -> dict[str, str]:
        if prompt_key in self.prompts:
            return self.prompts[prompt_key]
        raise ValueError(
            f"Prompt '{prompt_key}' not found. Available: {list(self.prompts.keys())}"
        )

    def get_available_vram_mb(self) -> int:
        return self.device.vram_budget_mb - self.device.vram_reserved_mb

    def list_models(self, stage: str) -> list[str]:
        return list(self.models.get(stage, {}).keys())


@lru_cache(maxsize=1)
def get_settings(config_dir: str | None = None) -> Settings:
    return Settings(config_dir)
