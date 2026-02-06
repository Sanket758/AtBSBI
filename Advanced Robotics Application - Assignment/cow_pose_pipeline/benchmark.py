from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from .data import list_images_from_split
from .utils import (
    cuda_sync,
    get_cpu_memory_mb,
    get_gpu_memory_nvidia_smi,
    human_bytes,
    save_json,
)


def _sample_images(images: list[Path], max_images: int) -> list[Path]:
    if not images:
        return []
    if max_images >= len(images):
        return images
    return random.sample(images, max_images)


def benchmark_pytorch(
    model,
    data_yaml: str | Path,
    split: str = "val",
    imgsz: int = 640,
    device: str | int | None = None,
    batch: int = 1,
    warmup: int = 3,
    runs: int = 30,
    max_images: int = 200,
    half: Optional[bool] = None,
    out_json: Optional[str | Path] = None,
) -> dict:
    images = list_images_from_split(data_yaml, split=split)
    images = _sample_images(images, max_images)
    if not images:
        raise RuntimeError("No images found for benchmarking.")

    def _predict_inference(**kwargs):
        try:
            import torch

            ctx = torch.no_grad()
        except Exception:
            from contextlib import nullcontext

            ctx = nullcontext()
        with ctx:
            return model.predict(**kwargs)

    if half is None:
        try:
            import torch

            half = bool(torch.cuda.is_available())
        except Exception:
            half = False

    # Warmup
    for i in range(min(warmup, len(images))):
        _predict_inference(
            source=str(images[i]),
            imgsz=imgsz,
            device=device,
            batch=batch,
            verbose=False,
            half=half,
        )

    cuda_sync()
    gpu_mem_before = get_gpu_memory_nvidia_smi()
    cpu_mem_before = get_cpu_memory_mb()

    cuda_sync()
    import time

    n = min(runs, len(images))
    start = time.perf_counter()
    for i in range(n):
        _predict_inference(
            source=str(images[i]),
            imgsz=imgsz,
            device=device,
            batch=batch,
            verbose=False,
            half=half,
        )
    cuda_sync()
    total = time.perf_counter() - start

    gpu_mem_after = get_gpu_memory_nvidia_smi()
    cpu_mem_after = get_cpu_memory_mb()

    latency_ms = (total / max(n, 1)) * 1000.0
    fps = max(n, 1) / max(total, 1e-9)

    result = {
        "backend": "pytorch",
        "latency_ms": round(latency_ms, 3),
        "fps": round(fps, 3),
        "num_images": n,
        "imgsz": imgsz,
        "batch": batch,
        "half": bool(half),
        "cpu_mem_mb": None,
        "gpu_mem_mb": None,
    }

    if cpu_mem_before is not None and cpu_mem_after is not None:
        result["cpu_mem_mb"] = round(cpu_mem_after - cpu_mem_before, 3)

    if gpu_mem_before is not None and gpu_mem_after is not None:
        result["gpu_mem_mb"] = int(gpu_mem_after - gpu_mem_before)

    if out_json:
        save_json(out_json, result)

    return result


def benchmark_onnx(
    onnx_path: str | Path,
    data_yaml: str | Path,
    split: str = "val",
    imgsz: int = 640,
    warmup: int = 3,
    runs: int = 30,
    max_images: int = 200,
    out_json: Optional[str | Path] = None,
) -> dict:
    import numpy as np

    try:
        import onnxruntime as ort
    except Exception as exc:
        raise RuntimeError("onnxruntime is not installed.") from exc

    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("opencv-python is required for ONNX benchmarking.") from exc

    onnx_path = Path(onnx_path).expanduser().resolve()
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    available = ort.get_available_providers()
    providers = [p for p in providers if p in available] or ["CPUExecutionProvider"]

    session = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = session.get_inputs()[0].name

    images = list_images_from_split(data_yaml, split=split)
    images = _sample_images(images, max_images)
    if not images:
        raise RuntimeError("No images found for benchmarking.")

    def preprocess(img_path: Path) -> np.ndarray:
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        return img

    # Warmup
    for i in range(min(warmup, len(images))):
        session.run(None, {input_name: preprocess(images[i])})

    import time

    gpu_mem_before = get_gpu_memory_nvidia_smi()
    cpu_mem_before = get_cpu_memory_mb()

    start = time.perf_counter()
    n = min(runs, len(images))
    for i in range(n):
        session.run(None, {input_name: preprocess(images[i])})
    total = time.perf_counter() - start

    gpu_mem_after = get_gpu_memory_nvidia_smi()
    cpu_mem_after = get_cpu_memory_mb()

    latency_ms = (total / max(n, 1)) * 1000.0
    fps = max(n, 1) / max(total, 1e-9)

    result = {
        "backend": "onnx",
        "providers": ",".join(providers),
        "latency_ms": round(latency_ms, 3),
        "fps": round(fps, 3),
        "num_images": n,
        "imgsz": imgsz,
        "cpu_mem_mb": None,
        "gpu_mem_mb": None,
    }

    if cpu_mem_before is not None and cpu_mem_after is not None:
        result["cpu_mem_mb"] = round(cpu_mem_after - cpu_mem_before, 3)

    if gpu_mem_before is not None and gpu_mem_after is not None:
        result["gpu_mem_mb"] = int(gpu_mem_after - gpu_mem_before)

    if out_json:
        save_json(out_json, result)

    return result
