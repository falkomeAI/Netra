from __future__ import annotations

import logging
import platform
import subprocess
from dataclasses import dataclass

logger = logging.getLogger("netra.device")


@dataclass
class DeviceInfo:
    platform: str
    architecture: str
    cuda_available: bool
    cuda_version: str | None
    gpu_name: str | None
    gpu_memory_total_mb: int
    gpu_memory_free_mb: int
    is_jetson: bool
    jetson_model: str | None
    tensorrt_available: bool

    def summary(self) -> str:
        lines = [
            f"Platform      : {self.platform}",
            f"Architecture  : {self.architecture}",
            f"CUDA          : {self.cuda_version or 'N/A'}",
            f"GPU           : {self.gpu_name or 'None'}",
            f"VRAM Total    : {self.gpu_memory_total_mb} MB",
            f"VRAM Free     : {self.gpu_memory_free_mb} MB",
            f"Jetson        : {self.jetson_model or 'No'}",
            f"TensorRT      : {'Yes' if self.tensorrt_available else 'No'}",
        ]
        return "\n".join(lines)


def get_device_info() -> DeviceInfo:
    cuda_available = False
    cuda_version = None
    gpu_name = None
    gpu_mem_total = 0
    gpu_mem_free = 0

    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_version = torch.version.cuda
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            gpu_mem_free = (
                torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_allocated(0)
            ) // (1024 * 1024)
    except ImportError:
        logger.warning("PyTorch not installed — cannot detect CUDA")

    is_jetson, jetson_model = _detect_jetson()

    trt_available = False
    try:
        import tensorrt  # noqa: F401
        trt_available = True
    except ImportError:
        pass

    return DeviceInfo(
        platform=platform.system(),
        architecture=platform.machine(),
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        gpu_name=gpu_name,
        gpu_memory_total_mb=gpu_mem_total,
        gpu_memory_free_mb=gpu_mem_free,
        is_jetson=is_jetson,
        jetson_model=jetson_model,
        tensorrt_available=trt_available,
    )


def _detect_jetson() -> tuple[bool, str | None]:
    try:
        result = subprocess.run(
            ["cat", "/proc/device-tree/model"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        model_str = result.stdout.strip().rstrip("\x00")
        if "NVIDIA" in model_str and ("Jetson" in model_str or "Orin" in model_str):
            return True, model_str
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    try:
        result = subprocess.run(
            ["cat", "/etc/nv_tegra_release"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return True, "NVIDIA Jetson (Tegra)"
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    return False, None
