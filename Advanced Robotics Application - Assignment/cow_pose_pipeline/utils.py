from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def human_bytes(num_bytes: float) -> float:
    return round(float(num_bytes) / (1024.0 * 1024.0), 3)


def cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def collect_env_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cwd": os.getcwd(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        info["torch"] = "not_installed"
    return info


def get_gpu_memory_nvidia_smi() -> Optional[int]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        first_line = output.strip().splitlines()[0]
        return int(first_line)
    except Exception:
        return None


def get_cpu_memory_mb() -> Optional[float]:
    try:
        import psutil

        process = psutil.Process(os.getpid())
        return human_bytes(process.memory_info().rss)
    except Exception:
        return None


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return out, elapsed
