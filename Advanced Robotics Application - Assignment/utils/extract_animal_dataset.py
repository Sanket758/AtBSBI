#!/usr/bin/env python3
"""
Extract single-class animal dataset from Animal-3D dataset.
Converts annotations to YOLO Pose format for training.

Usage:
    python extract_animal_dataset.py --root ./Animal-3D --animal cow --output ./cow_dataset
    python extract_animal_dataset.py --root ./Animal-3D --animal horse --output ./horse_dataset
    python extract_animal_dataset.py --root ./Animal-3D --generate-mapping-only
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List
import re


# Animal-3D uses 26 keypoints for quadrupeds
keypoint_names = [
    "L_Eye",  # 0
    "R_Eye",  # 1
    "Chin",  # 2
    "R_F_Hoof",  # 3
    "L_F_Hoof",  # 4
    "R_B_Hoof",  # 5
    "L_B_Hoof",  # 6
    "Tail_Base",  # 7
    "R_Shoulder",  # 8
    "L_Shoulder",  # 9
    "R_Hip",  # 10
    "L_Hip",  # 11
    "Spine_Mid",  # 12
    "Withers",  # 13
    "R_F_Knee",  # 14
    "L_F_Knee",  # 15
    "R_B_Knee",  # 16  (Hock)
    "L_B_Knee",  # 17  (Hock)
    "Throat",  # 18
    "Tail_Tip",  # 19
    "L_EarBase",  # 20
    "R_EarBase",  # 21
    "Mouth_Corner",  # 22
    "Nose_Tip",  # 23
    "Nose_Bridge",  # 24
    "Tail_Mid",  # 25
]


def get_animal_from_path(img_path: str) -> str:
    """Extract animal name from image path suffix like 'xxx_cow.jpg'"""
    filename = os.path.basename(img_path)
    match = re.search(r"_([a-z]+)\.(jpg|jpeg|png)$", filename, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return None


def convert_keypoints_to_yolo(
    keypoints_2d: List[List[float]], bbox: List[float], img_width: int, img_height: int
) -> str:
    """
    Convert Animal-3D keypoints to YOLO Pose format.

    YOLO Pose format per line:
    class_id center_x center_y width height kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v ...

    All coordinates are normalized to [0, 1].
    """
    # Animal-3D bbox is COCO-style: [x, y, w, h]
    x, y, w, h = bbox
    x_min = x
    y_min = y
    x_max = x + w
    y_max = y + h

    # Clamp to image bounds for safety
    x_min = max(0.0, min(float(img_width), x_min))
    y_min = max(0.0, min(float(img_height), y_min))
    x_max = max(0.0, min(float(img_width), x_max))
    y_max = max(0.0, min(float(img_height), y_max))

    box_width = max(0.0, x_max - x_min)
    box_height = max(0.0, y_max - y_min)
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0

    # Normalize bbox
    norm_cx = max(0, min(1, center_x / img_width))
    norm_cy = max(0, min(1, center_y / img_height))
    norm_w = max(0, min(1, box_width / img_width))
    norm_h = max(0, min(1, box_height / img_height))

    yolo_line = f"0 {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}"

    for kp in keypoints_2d:
        x, y, v = kp[0], kp[1], kp[2]
        norm_x = max(0, min(1, x / img_width))
        norm_y = max(0, min(1, y / img_height))
        yolo_v = 2 if v > 0 else 0
        yolo_line += f" {norm_x:.6f} {norm_y:.6f} {yolo_v}"

    return yolo_line


def create_dataset_yaml(output_dir: Path, animal_name: str, num_keypoints: int):
    """Create YOLO dataset configuration YAML file."""
    kp_names = ", ".join(keypoint_names)
    yaml_content = f"""# YOLO Pose Dataset Configuration
# Auto-generated for {animal_name} pose estimation

path: {output_dir.absolute()}
train: images/train
val: images/val

# Keypoints
kpt_shape: [{num_keypoints}, 3]  # [num_keypoints, (x, y, visibility)]

# Classes
names:
  0: {animal_name}

# Keypoint names (for reference)
# {kp_names}
"""
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"Created dataset config: {yaml_path}")
    return yaml_path


def process_annotations(
    data: List[Dict],
    animal_name: str,
    root_dir: Path,
    output_dir: Path,
    split: str,
    source_split: str,  # The split name in source data (train/test)
    mapping: Dict,
) -> int:
    """Process annotations for a specific animal and split."""
    processed = 0

    images_dir = output_dir / "images" / split
    labels_dir = output_dir / "labels" / split
    obj_dir = output_dir / "obj_files" / split

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    obj_dir.mkdir(parents=True, exist_ok=True)

    for item in data:
        img_path = item.get("img_path", "")
        detected_animal = get_animal_from_path(img_path)

        if detected_animal != animal_name.lower():
            continue

        img_width = item.get("width", 0)
        img_height = item.get("height", 0)

        if img_width == 0 or img_height == 0:
            print(f"Warning: Invalid dimensions for {img_path}, skipping")
            continue

        keypoints_2d = item.get("keypoint_2d", [])
        bbox = item.get("bbox", [])

        if len(keypoints_2d) == 0 or len(bbox) != 4:
            print(f"Warning: Missing keypoints or bbox for {img_path}, skipping")
            continue

        img_filename = os.path.basename(img_path)
        base_name = os.path.splitext(img_filename)[0]

        folder_id = (
            img_path.split("/")[2] if len(img_path.split("/")) > 2 else "unknown"
        )
        if folder_id not in mapping:
            mapping[folder_id] = {"animal": detected_animal, "count": 0, "examples": []}
        mapping[folder_id]["count"] += 1
        if len(mapping[folder_id]["examples"]) < 3:
            mapping[folder_id]["examples"].append(img_filename)

        # Find image (handle nested structure like images/images/train/...)
        src_img = root_dir / img_path
        if not src_img.exists():
            # Try images/images/... structure (from zip extraction)
            alt_path = img_path.replace("images/", "")  # Remove leading "images/"
            alt_src_img = root_dir / "images" / "images" / alt_path
            if alt_src_img.exists():
                src_img = alt_src_img
            else:
                # Also try just images/images/<rest>
                alt_src_img2 = root_dir / "images" / alt_path
                if alt_src_img2.exists():
                    src_img = alt_src_img2

        if src_img.exists():
            dst_img = images_dir / img_filename
            shutil.copy2(src_img, dst_img)
        else:
            print(f"Warning: Image not found: {src_img}")
            continue

        # Convert and save YOLO annotation
        yolo_annotation = convert_keypoints_to_yolo(
            keypoints_2d, bbox, img_width, img_height
        )
        label_path = labels_dir / f"{base_name}.txt"
        with open(label_path, "w") as f:
            f.write(yolo_annotation + "\n")

        # Copy OBJ file if exists (use source_split for original data structure)
        src_obj = (
            root_dir
            / "obj_files"
            / "obj_files"
            / source_split  # Use source_split (train/test) not output split (train/val)
            / folder_id
            / f"{base_name}.obj"
        )
        if src_obj.exists():
            dst_obj = obj_dir / f"{base_name}.obj"
            shutil.copy2(src_obj, dst_obj)

        processed += 1

    return processed


def main():
    parser = argparse.ArgumentParser(
        description="Extract single-class animal dataset from Animal-3D"
    )
    parser.add_argument(
        "--root", type=str, required=True, help="Root directory of Animal-3D dataset"
    )
    parser.add_argument(
        "--animal",
        type=str,
        default=None,
        help="Animal name to extract (e.g., cow, horse, cat, dog)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for extracted dataset",
    )
    parser.add_argument(
        "--generate-mapping-only",
        action="store_true",
        help="Only generate folder-to-animal mapping without extracting data",
    )

    args = parser.parse_args()
    root_dir = Path(args.root).resolve()

    # Load annotations
    train_json = root_dir / "train.json"
    test_json = root_dir / "test.json"

    if not train_json.exists() or not test_json.exists():
        print(f"Error: Cannot find train.json or test.json in {root_dir}")
        return 1

    print(f"Animal-3D Root: {root_dir}")
    print("-" * 50)

    print("Loading annotations...")
    with open(train_json, "r") as f:
        train_data = json.load(f)["data"]
    print(f"  Loaded {len(train_data)} training annotations")

    with open(test_json, "r") as f:
        test_data = json.load(f)["data"]
    print(f"  Loaded {len(test_data)} test annotations")

    mapping = {}

    # Generate mapping only mode
    if args.generate_mapping_only:
        print("\nGenerating folder-to-animal mapping...")
        for item in train_data + test_data:
            img_path = item.get("img_path", "")
            detected_animal = get_animal_from_path(img_path)
            if detected_animal:
                folder_id = (
                    img_path.split("/")[2]
                    if len(img_path.split("/")) > 2
                    else "unknown"
                )
                if folder_id not in mapping:
                    mapping[folder_id] = {"animal": detected_animal, "count": 0}
                mapping[folder_id]["count"] += 1

        mapping_path = root_dir / "folder_animal_mapping.json"
        with open(mapping_path, "w") as f:
            json.dump(mapping, f, indent=2)
        print(f"\nMapping saved to: {mapping_path}")

        print("\nFolder Mapping Summary:")
        print("-" * 50)
        animal_counts = {}
        for folder_id, info in mapping.items():
            animal = info["animal"]
            animal_counts[animal] = animal_counts.get(animal, 0) + info["count"]
            print(f"  {folder_id}: {animal} ({info['count']} images)")

        print("\nTotal per animal:")
        for animal, count in sorted(animal_counts.items(), key=lambda x: -x[1]):
            print(f"  {animal}: {count}")

        return 0

    # Extraction mode
    if not args.animal:
        print("Error: --animal is required for extraction mode")
        return 1

    animal_name = args.animal.lower()
    output_dir = (
        Path(args.output) if args.output else Path(f"./{animal_name}_yolo_dataset")
    )
    output_dir = output_dir.resolve()

    print(f"Target Animal: {animal_name}")
    print(f"Output Directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing training data for '{animal_name}'...")
    train_count = process_annotations(
        train_data, animal_name, root_dir, output_dir, "train", "train", mapping
    )
    print(f"  Extracted {train_count} training samples")

    print(f"\nProcessing test/validation data for '{animal_name}'...")
    val_count = process_annotations(
        test_data, animal_name, root_dir, output_dir, "val", "test", mapping
    )
    print(f"  Extracted {val_count} validation samples")

    num_keypoints = len(keypoint_names)
    create_dataset_yaml(output_dir, animal_name, num_keypoints)

    mapping_path = output_dir / "folder_animal_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"\nFolder mapping saved to: {mapping_path}")

    print("\n" + "=" * 50)
    print("EXTRACTION COMPLETE")
    print("=" * 50)
    print(f"  Animal: {animal_name}")
    print(f"  Training samples: {train_count}")
    print(f"  Validation samples: {val_count}")
    print(f"  Total: {train_count + val_count}")
    print(f"  Keypoints: {num_keypoints}")
    print(f"  Output: {output_dir}")
    print("\nDataset structure:")
    print(f"  {output_dir}/")
    print(f"    ├── dataset.yaml")
    print(f"    ├── folder_animal_mapping.json")
    print(f"    ├── images/")
    print(f"    │   ├── train/  ({train_count} images)")
    print(f"    │   └── val/    ({val_count} images)")
    print(f"    ├── labels/")
    print(f"    │   ├── train/  ({train_count} labels)")
    print(f"    │   └── val/    ({val_count} labels)")
    print(f"    └── obj_files/")
    print(f"        ├── train/")
    print(f"        └── val/")

    print("\nTo train YOLOv8 Pose:")
    print(
        f"  yolo pose train data={output_dir}/dataset.yaml model=yolov8n-pose.pt epochs=100"
    )

    return 0


if __name__ == "__main__":
    exit(main())
