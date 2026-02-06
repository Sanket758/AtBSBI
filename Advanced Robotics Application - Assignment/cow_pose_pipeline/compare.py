from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from ultralytics import YOLO

from .benchmark import benchmark_pytorch
from .data import resolve_data_yaml
from .utils import ensure_dir, save_json, timestamp


def _model_params(model) -> int:
    try:
        return int(sum(p.numel() for p in model.model.parameters()))
    except Exception:
        return 0


def _extract_pose_metrics(metrics) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    if hasattr(metrics, "pose"):
        pose = metrics.pose
        for key in ["map", "map50", "map75", "mp", "mr"]:
            if hasattr(pose, key):
                out[f"pose_{key}"] = float(getattr(pose, key))

    if hasattr(metrics, "results_dict"):
        for k, v in metrics.results_dict.items():
            if isinstance(v, (int, float)):
                out[k] = float(v)

    return out


def compare_models(
    models: Iterable[str],
    data_yaml: str | Path,
    dataset_dir: Optional[str | Path] = None,
    imgsz: int = 640,
    device: str | int | None = None,
    batch: int = 1,
    project: str | Path = "experiments",
    name: Optional[str] = None,
    max_images: int = 200,
) -> Path:
    data_yaml = resolve_data_yaml(data_yaml, dataset_dir=dataset_dir, out_dir=Path(project) / "configs")

    if name is None:
        name = f"compare_{timestamp()}"

    project = Path(project)
    ensure_dir(project)

    run_dir = project / name
    ensure_dir(run_dir)

    rows = []

    for model_id in models:
        model = YOLO(model_id)
        metrics = model.val(data=str(data_yaml), imgsz=imgsz, device=device, batch=batch)

        metrics_dict = _extract_pose_metrics(metrics)
        bench = benchmark_pytorch(
            model=model,
            data_yaml=str(data_yaml),
            imgsz=imgsz,
            device=device,
            batch=1,
            max_images=max_images,
        )

        params = _model_params(model)
        weight_path = Path(model_id)
        weight_size = weight_path.stat().st_size if weight_path.exists() else None

        row: Dict[str, Any] = {
            "model_id": model_id,
            "params": params,
            "weights_size_mb": round(weight_size / (1024 * 1024), 3) if weight_size else None,
            "imgsz": imgsz,
            **metrics_dict,
            **bench,
        }
        rows.append(row)

    csv_path = run_dir / "model_compare.csv"
    json_path = run_dir / "model_compare.json"

    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    save_json(json_path, {"rows": rows})

    return csv_path
