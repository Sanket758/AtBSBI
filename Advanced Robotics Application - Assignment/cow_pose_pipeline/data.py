from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from .utils import ensure_dir


def resolve_data_yaml(
    data_yaml: str | Path,
    dataset_dir: Optional[str | Path] = None,
    out_dir: Optional[str | Path] = None,
) -> Path:
    data_yaml = Path(data_yaml).expanduser().resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError("Dataset YAML must contain a mapping at the top level.")

    base_dir = data_yaml.parent
    desired_path = None

    if dataset_dir:
        desired_path = Path(dataset_dir).expanduser().resolve()
    else:
        raw_path = data.get("path", "")
        if raw_path:
            desired_path = Path(str(raw_path)).expanduser().resolve()

    if desired_path is None or not desired_path.exists():
        # Fallback to the YAML directory if the declared path does not exist.
        desired_path = base_dir.resolve()

    data["path"] = str(desired_path)

    if out_dir is None:
        out_dir = base_dir / "_resolved"
    out_dir = ensure_dir(out_dir)

    out_path = out_dir / f"{data_yaml.stem}_resolved.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    return out_path


def list_images_from_split(data_yaml: str | Path, split: str = "val") -> list[Path]:
    data_yaml = Path(data_yaml).expanduser().resolve()
    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    base_path = Path(data.get("path", data_yaml.parent)).expanduser().resolve()
    split_path = Path(data.get(split, ""))
    if not split_path.is_absolute():
        split_path = base_path / split_path

    if not split_path.exists():
        raise FileNotFoundError(f"Split path does not exist: {split_path}")

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [p for p in split_path.rglob("*") if p.suffix.lower() in exts]
    images.sort()
    return images
