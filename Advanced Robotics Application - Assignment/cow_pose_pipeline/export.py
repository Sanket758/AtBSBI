from __future__ import annotations

from pathlib import Path
from typing import Optional

from ultralytics import YOLO

from .utils import ensure_dir, save_json


def export_onnx(
    weights: str | Path,
    out_dir: str | Path = "experiments/exports",
    imgsz: int = 640,
    dynamic: bool = False,
    simplify: bool = False,
) -> Path:
    weights = Path(weights).expanduser().resolve()
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    out_dir = ensure_dir(out_dir)

    model = YOLO(str(weights))
    export_args = {
        "format": "onnx",
        "imgsz": imgsz,
        "dynamic": dynamic,
        "simplify": simplify,
    }

    exported = model.export(**export_args)

    # Ultralytics returns a list or path depending on version
    if isinstance(exported, (list, tuple)) and exported:
        onnx_path = Path(exported[0])
    else:
        onnx_path = Path(exported)

    onnx_path = onnx_path.expanduser().resolve()

    info = {
        "weights": str(weights),
        "onnx_path": str(onnx_path),
        "imgsz": imgsz,
        "dynamic": dynamic,
        "simplify": simplify,
    }
    save_json(out_dir / f"{weights.stem}_onnx_export.json", info)

    return onnx_path
