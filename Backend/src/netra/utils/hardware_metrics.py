"""Hardware utilization and temperature sampling for performance benchmarks.

Collects CPU/GPU utilization, temperature, and memory during pipeline runs.
CPU metrics come from /proc/stat and /sys/class/thermal (Linux).
GPU metrics come from nvidia-smi (NVIDIA) or tegrastats (Jetson).
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("netra.hardware_metrics")


@dataclass
class HardwareSample:
    timestamp: float
    cpu_util_pct: float | None = None
    cpu_temp_c: float | None = None
    ram_used_mb: float | None = None
    ram_total_mb: float | None = None
    gpu_util_pct: float | None = None
    gpu_temp_c: float | None = None
    gpu_mem_used_mb: float | None = None
    gpu_mem_total_mb: float | None = None


@dataclass
class MetricStats:
    min: float | None = None
    max: float | None = None
    avg: float | None = None
    samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── CPU jiffies (kept as module-level state for delta tracking) ──

_prev_jiffies: tuple[float, float] | None = None
_prev_jiffies_time: float = 0.0


def _read_cpu_jiffies() -> tuple[float, float] | None:
    """Read aggregate (idle, total) jiffies from /proc/stat."""
    try:
        with open("/proc/stat", encoding="utf-8") as f:
            parts = f.readline().split()
        if parts[0] != "cpu":
            return None
        values = [float(v) for v in parts[1:]]
        idle = values[3] + (values[4] if len(values) > 4 else 0.0)
        total = sum(values)
        return idle, total
    except (OSError, ValueError, IndexError):
        return None


def read_cpu_util_pct() -> float | None:
    """Non-blocking CPU utilization using delta from last call.

    First call returns None (no delta yet). Subsequent calls return
    utilization since the previous call without any sleep().
    """
    global _prev_jiffies, _prev_jiffies_time

    current = _read_cpu_jiffies()
    if current is None:
        return None

    if _prev_jiffies is None or (time.monotonic() - _prev_jiffies_time) > 10.0:
        _prev_jiffies = current
        _prev_jiffies_time = time.monotonic()
        return None

    idle_delta = current[0] - _prev_jiffies[0]
    total_delta = current[1] - _prev_jiffies[1]

    _prev_jiffies = current
    _prev_jiffies_time = time.monotonic()

    if total_delta <= 0:
        return None
    return round(max(0.0, min(100.0, (1.0 - idle_delta / total_delta) * 100.0)), 1)


def read_cpu_temp_c() -> float | None:
    """Read CPU package temperature from Linux thermal zones.

    Filters to zones whose type contains 'cpu', 'x86', 'acpitz',
    'coretemp', or 'k10temp'. Ignores GPU thermal zones.
    """
    thermal_root = Path("/sys/class/thermal")
    if not thermal_root.exists():
        return None

    cpu_labels = {"cpu", "x86", "acpitz", "coretemp", "k10temp", "soc", "tboard"}
    gpu_labels = {"gpu", "nvidia"}
    temps: list[float] = []

    for zone in sorted(thermal_root.glob("thermal_zone*")):
        temp_file = zone / "temp"
        type_file = zone / "type"
        try:
            raw = int(temp_file.read_text().strip())
            label = type_file.read_text().strip().lower() if type_file.exists() else ""
            temp_c = raw / 1000.0
            if temp_c < 1.0 or temp_c > 120.0:
                continue
            if any(g in label for g in gpu_labels):
                continue
            if any(c in label for c in cpu_labels):
                temps.append(temp_c)
            elif not label and 20.0 <= temp_c <= 110.0:
                temps.append(temp_c)
        except (OSError, ValueError):
            continue

    return round(max(temps), 1) if temps else None


def read_ram_usage() -> tuple[float | None, float | None]:
    """Read system RAM usage from /proc/meminfo. Returns (used_mb, total_mb)."""
    try:
        info: dict[str, int] = {}
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    info[key] = int(parts[1])
        total_kb = info.get("MemTotal", 0)
        available_kb = info.get("MemAvailable", info.get("MemFree", 0))
        if total_kb == 0:
            return None, None
        used_mb = round((total_kb - available_kb) / 1024.0, 1)
        total_mb = round(total_kb / 1024.0, 1)
        return used_mb, total_mb
    except (OSError, ValueError, KeyError):
        return None, None


def read_gpu_metrics() -> dict[str, float | None]:
    """Read GPU utilization, temperature, and memory from nvidia-smi.

    Returns dict with keys: util_pct, temp_c, mem_used_mb, mem_total_mb.
    """
    out: dict[str, float | None] = {
        "util_pct": None, "temp_c": None,
        "mem_used_mb": None, "mem_total_mb": None,
    }
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return out
        line = result.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            return out

        def _parse(s: str) -> float | None:
            if s in ("[N/A]", "N/A", ""):
                return None
            try:
                return float(s)
            except ValueError:
                return None

        out["util_pct"] = _parse(parts[0])
        out["temp_c"] = _parse(parts[1])
        out["mem_used_mb"] = _parse(parts[2])
        out["mem_total_mb"] = _parse(parts[3])
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return out


def snapshot_hardware() -> HardwareSample:
    """Take a single point-in-time hardware sample."""
    cpu_util = read_cpu_util_pct()
    cpu_temp = read_cpu_temp_c()
    ram_used, ram_total = read_ram_usage()
    gpu = read_gpu_metrics()
    return HardwareSample(
        timestamp=time.time(),
        cpu_util_pct=cpu_util,
        cpu_temp_c=cpu_temp,
        ram_used_mb=ram_used,
        ram_total_mb=ram_total,
        gpu_util_pct=gpu["util_pct"],
        gpu_temp_c=gpu["temp_c"],
        gpu_mem_used_mb=gpu["mem_used_mb"],
        gpu_mem_total_mb=gpu["mem_total_mb"],
    )


def _stats_for(values: list[float]) -> MetricStats:
    if not values:
        return MetricStats()
    return MetricStats(
        min=round(min(values), 1),
        max=round(max(values), 1),
        avg=round(sum(values) / len(values), 1),
        samples=len(values),
    )


_METRIC_FIELDS = [
    "cpu_util_pct", "cpu_temp_c", "ram_used_mb",
    "gpu_util_pct", "gpu_temp_c", "gpu_mem_used_mb",
]


def summarize_samples(samples: list[HardwareSample]) -> dict[str, Any]:
    """Compute min/max/avg statistics from a list of hardware samples."""
    def collect(attr: str) -> list[float]:
        return [getattr(s, attr) for s in samples if getattr(s, attr) is not None]

    summary: dict[str, Any] = {}
    for f in _METRIC_FIELDS:
        summary[f] = _stats_for(collect(f)).to_dict()
    summary["sample_count"] = len(samples)
    return summary


@dataclass
class MetricsSampler:
    """Background thread that collects hardware samples at a fixed interval."""

    interval_s: float = 0.5
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _samples: list[HardwareSample] = field(default_factory=list, init=False, repr=False)

    def start(self) -> None:
        self._stop.clear()
        with self._lock:
            self._samples = []
        self._thread = threading.Thread(target=self._loop, daemon=True, name="hw-sampler")
        self._thread.start()

    def stop(self) -> list[HardwareSample]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        with self._lock:
            return list(self._samples)

    def _loop(self) -> None:
        while not self._stop.is_set():
            sample = snapshot_hardware()
            with self._lock:
                self._samples.append(sample)
            self._stop.wait(self.interval_s)


def system_info() -> dict[str, Any]:
    """Collect static system information + one baseline hardware snapshot."""
    try:
        from netra.utils.device import get_device_info
        info = get_device_info()
        device_dict = {
            "platform": info.platform,
            "architecture": info.architecture,
            "cuda_available": info.cuda_available,
            "cuda_version": info.cuda_version,
            "gpu_name": info.gpu_name,
            "gpu_memory_total_mb": info.gpu_memory_total_mb,
            "gpu_memory_free_mb": info.gpu_memory_free_mb,
            "is_jetson": info.is_jetson,
            "jetson_model": info.jetson_model,
        }
    except ImportError:
        import platform as _platform
        device_dict = {
            "platform": _platform.system(),
            "architecture": _platform.machine(),
        }

    cpu_count = os.cpu_count()
    ram_used, ram_total = read_ram_usage()
    baseline = snapshot_hardware()

    return {
        **device_dict,
        "cpu_count": cpu_count,
        "ram_total_mb": ram_total,
        "baseline": asdict(baseline),
    }
