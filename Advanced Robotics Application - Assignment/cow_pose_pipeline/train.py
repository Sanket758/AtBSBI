from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from ultralytics import YOLO

from .data import resolve_data_yaml
from .utils import collect_env_info, ensure_dir, save_json, timestamp

DEFAULT_AUG = {
    "degrees": 10.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 2.0,
    "fliplr": 0.5,
    "flipud": 0.0,
    "mosaic": 0.5,
    "mixup": 0.05,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
}


def train_model(
    data_yaml: str | Path,
    dataset_dir: Optional[str | Path] = None,
    model_id: str = "yolov8m-pose.pt",
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 640,
    device: str | int | None = None,
    project: str | Path = "experiments",
    name: Optional[str] = None,
    patience: int = 50,
    workers: int = 4,
    seed: int = 42,
    amp: bool = True,
    cache: bool = False,
    resume: bool = False,
    pretrained: bool = True,
    aug: Optional[Dict[str, Any]] = None,
) -> Path:
    data_yaml = resolve_data_yaml(data_yaml, dataset_dir=dataset_dir, out_dir=Path(project) / "configs")

    if name is None:
        name = f"pose/{timestamp()}"

    project = Path(project)
    ensure_dir(project)

    run_dir = project / name

    model = YOLO(model_id)
    aug_params = DEFAULT_AUG.copy()
    if aug:
        aug_params.update(aug)

    train_args = dict(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=str(project),
        name=name,
        patience=patience,
        workers=workers,
        seed=seed,
        amp=amp,
        cache=cache,
        resume=resume,
        pretrained=pretrained,
    )
    train_args.update(aug_params)

    results = model.train(**train_args)

    manifest = {
        "model_id": model_id,
        "run_dir": str(run_dir),
        "train_args": train_args,
        "env": collect_env_info(),
    }
    save_json(run_dir / "run_manifest.json", manifest)

    # Best weights path
    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        weights = run_dir / "weights" / "last.pt"
    return weights


def _parse_aug(aug_str: Optional[str]) -> Optional[Dict[str, Any]]:
    if not aug_str:
        return None
    try:
        return json.loads(aug_str)
    except Exception as exc:
        raise ValueError("--aug must be a valid JSON string") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO Pose with experiment tracking.")
    parser.add_argument("--data", required=True, help="Path to dataset YAML")
    parser.add_argument("--dataset-dir", default=None, help="Override dataset root directory")
    parser.add_argument("--model", default="yolov8m-pose.pt", help="Model id or weights path")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="Device id(s), e.g. 0 or 0,1")
    parser.add_argument("--project", default="experiments")
    parser.add_argument("--name", default=None)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--aug", default=None, help="JSON string for augmentation overrides")

    args = parser.parse_args()

    amp = True
    if args.no_amp:
        amp = False
    elif args.amp:
        amp = True

    pretrained = True
    if args.no_pretrained:
        pretrained = False

    aug = _parse_aug(args.aug)

    weights = train_model(
        data_yaml=args.data,
        dataset_dir=args.dataset_dir,
        model_id=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        seed=args.seed,
        amp=amp,
        cache=args.cache,
        resume=args.resume,
        pretrained=pretrained,
        aug=aug,
    )

    print(f"Best weights: {weights}")


if __name__ == "__main__":
    main()
