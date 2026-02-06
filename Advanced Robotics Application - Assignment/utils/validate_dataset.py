#!/usr/bin/env python3
"""
Validate YOLO Pose dataset integrity before training.
Checks images, labels, format, and visualizes samples.
"""

import os
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def validate_dataset(dataset_dir: str):
    """Validate YOLO Pose dataset integrity."""
    dataset_dir = Path(dataset_dir)

    print("=" * 60)
    print("YOLO POSE DATASET VALIDATION")
    print("=" * 60)
    print(f"Dataset: {dataset_dir}")
    print()

    errors = []
    warnings = []
    stats = {
        "train": {"images": 0, "labels": 0, "obj_files": 0, "valid_labels": 0},
        "val": {"images": 0, "labels": 0, "obj_files": 0, "valid_labels": 0},
    }

    # Check dataset.yaml exists
    yaml_path = dataset_dir / "dataset.yaml"
    if not yaml_path.exists():
        errors.append("dataset.yaml not found!")
    else:
        print("✓ dataset.yaml found")
        with open(yaml_path, "r") as f:
            content = f.read()
            if "kpt_shape" in content:
                print("✓ kpt_shape defined in config")
            else:
                warnings.append("kpt_shape not found in dataset.yaml")

    print()

    # Check each split
    for split in ["train", "val"]:
        print(f"--- Checking {split.upper()} split ---")

        images_dir = dataset_dir / "images" / split
        labels_dir = dataset_dir / "labels" / split
        obj_dir = dataset_dir / "obj_files" / split

        # Get file lists
        images = set(
            f.stem
            for f in images_dir.glob("*.*")
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        )
        labels = set(f.stem for f in labels_dir.glob("*.txt"))
        obj_files = set(f.stem for f in obj_dir.glob("*.obj"))

        stats[split]["images"] = len(images)
        stats[split]["labels"] = len(labels)
        stats[split]["obj_files"] = len(obj_files)

        print(f"  Images:    {len(images)}")
        print(f"  Labels:    {len(labels)}")
        print(f"  OBJ files: {len(obj_files)}")

        # Check matching
        images_without_labels = images - labels
        labels_without_images = labels - images

        if images_without_labels:
            errors.append(
                f"{split}: {len(images_without_labels)} images without labels"
            )
            print(f"  ✗ {len(images_without_labels)} images missing labels")
        else:
            print(f"  ✓ All images have labels")

        if labels_without_images:
            errors.append(
                f"{split}: {len(labels_without_images)} labels without images"
            )
            print(f"  ✗ {len(labels_without_images)} orphan labels")
        else:
            print(f"  ✓ All labels have images")

        # Validate label format
        print(f"  Validating label format...")
        valid_count = 0
        invalid_labels = []

        for label_file in labels_dir.glob("*.txt"):
            try:
                with open(label_file, "r") as f:
                    lines = f.readlines()

                for line_num, line in enumerate(lines, 1):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        invalid_labels.append(
                            f"{label_file.name}:{line_num} - too few values"
                        )
                        continue

                    # class_id + 4 bbox + keypoints (26 keypoints * 3 values each = 78)
                    expected_min = 5  # class_id + bbox
                    expected_with_kpts = 5 + 26 * 3  # 83 values for 26 keypoints

                    if len(parts) != expected_with_kpts:
                        if len(parts) < expected_min:
                            invalid_labels.append(
                                f"{label_file.name}:{line_num} - {len(parts)} values (expected {expected_with_kpts})"
                            )
                            continue

                    # Validate bbox values are in [0, 1]
                    class_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])

                    if not (
                        0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1
                    ):
                        invalid_labels.append(
                            f"{label_file.name}:{line_num} - bbox out of range"
                        )
                        continue

                    # Validate keypoints
                    kpts = parts[5:]
                    valid_kpts = True
                    for i in range(0, len(kpts), 3):
                        if i + 2 < len(kpts):
                            kx, ky, kv = (
                                float(kpts[i]),
                                float(kpts[i + 1]),
                                float(kpts[i + 2]),
                            )
                            if not (0 <= kx <= 1 and 0 <= ky <= 1 and kv in [0, 1, 2]):
                                valid_kpts = False
                                break

                    if not valid_kpts:
                        invalid_labels.append(
                            f"{label_file.name}:{line_num} - invalid keypoint values"
                        )
                        continue

                    valid_count += 1

            except Exception as e:
                invalid_labels.append(f"{label_file.name} - parse error: {e}")

        stats[split]["valid_labels"] = valid_count

        if invalid_labels:
            print(f"  ✗ {len(invalid_labels)} invalid label entries")
            for inv in invalid_labels[:5]:
                print(f"      - {inv}")
            if len(invalid_labels) > 5:
                print(f"      ... and {len(invalid_labels) - 5} more")
            errors.extend(invalid_labels[:3])
        else:
            print(f"  ✓ All {valid_count} label entries valid")

        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_images = stats["train"]["images"] + stats["val"]["images"]
    total_labels = stats["train"]["labels"] + stats["val"]["labels"]
    total_objs = stats["train"]["obj_files"] + stats["val"]["obj_files"]

    print(f"Total images:    {total_images}")
    print(f"Total labels:    {total_labels}")
    print(f"Total OBJ files: {total_objs}")
    print(f"Train/Val split: {stats['train']['images']}/{stats['val']['images']}")
    print()

    if errors:
        print(f"❌ ERRORS: {len(errors)}")
        for err in errors[:10]:
            print(f"  - {err}")
        return False
    elif warnings:
        print(f"⚠️  WARNINGS: {len(warnings)}")
        for warn in warnings:
            print(f"  - {warn}")
        print("\n✓ Dataset is VALID (with warnings)")
        return True
    else:
        print("✅ Dataset is VALID and ready for training!")
        return True


def visualize_samples(dataset_dir: str, num_samples: int = 4, output_path: str = None):
    """Visualize random samples with annotations."""
    dataset_dir = Path(dataset_dir)

    # Keypoint names for reference
    KEYPOINT_NAMES = [
        "L_Eye",
        "R_Eye",
        "Chin",
        "R_F_Hoof",
        "L_F_Hoof",
        "R_B_Hoof",
        "L_B_Hoof",
        "Tail_Base",
        "R_Shoulder",
        "L_Shoulder",
        "R_Hip",
        "L_Hip",
        "Spine_Mid",
        "Withers",
        "R_F_Knee",
        "L_F_Knee",
        "R_B_Knee",
        "L_B_Knee",
        "Throat",
        "Tail_Tip",
        "L_EarBase",
        "R_EarBase",
        "Mouth_Corner",
        "Nose_Tip",
        "Nose_Bridge",
        "Tail_Mid",
    ]

    # Skeleton connections for visualization
    SKELETON = [
        # --- Head Mesh ---
        (23, 24),  # Nose Tip -> Nose Bridge
        (24, 22),  # Nose Bridge -> Mouth Corner
        (22, 2),  # Mouth Corner -> Chin
        (23, 2),  # Nose Tip -> Chin
        (24, 1),  # Nose Bridge -> R Eye
        (24, 0),  # Nose Bridge -> L Eye
        (1, 21),  # R Eye -> R Ear
        (0, 20),  # L Eye -> L Ear
        (18, 21),  # Throat -> R Ear
        (18, 20),  # Throat -> L Ear
        (18, 2),  # Throat -> Chin
        # --- Spine Chain ---
        (18, 13),  # Throat -> Withers
        (13, 12),  # Withers -> Spine Mid
        (12, 7),  # Spine Mid -> Tail Base
        (7, 25),  # Tail Base -> Tail Mid
        (25, 19),  # Tail Mid -> Tail Tip
        # --- Front Legs (Anchored at Spine_Mid 12) ---
        (12, 8),  # Spine Mid -> R Shoulder
        (8, 14),  # R Shoulder -> R Knee
        (14, 3),  # R Knee -> R Hoof
        (12, 9),  # Spine Mid -> L Shoulder
        (9, 15),  # L Shoulder -> L Knee
        (15, 4),  # L Knee -> L Hoof
        # --- Back Legs (Anchored at Tail_Base 7) ---
        (7, 10),  # Tail Base -> R Hip
        (10, 16),  # R Hip -> R Hock
        (16, 5),  # R Hock -> R Hoof
        (7, 11),  # Tail Base -> L Hip
        (11, 17),  # L Hip -> L Hock
        (17, 6),  # L Hock -> L Hoof
        # --- Cross Connections ---
        (8, 9),  # Shoulder -> Shoulder
        (10, 11),  # Hip -> Hip
    ]

    # Get sample images from train split
    images_dir = dataset_dir / "images" / "train"
    labels_dir = dataset_dir / "labels" / "train"

    image_files = list(images_dir.glob("*.jpg"))
    if len(image_files) == 0:
        image_files = list(images_dir.glob("*.png"))

    # Select random samples
    np.random.seed(42)
    sample_indices = np.random.choice(
        len(image_files), min(num_samples, len(image_files)), replace=False
    )
    samples = [image_files[i] for i in sample_indices]

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()

    for idx, (ax, img_path) in enumerate(zip(axes, samples)):
        # Load image
        img = Image.open(img_path)
        img_w, img_h = img.size

        ax.imshow(img)
        ax.set_title(img_path.name, fontsize=10)

        # Load label
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            with open(label_path, "r") as f:
                line = f.readline().strip()

            parts = line.split()
            cx, cy, w, h = map(float, parts[1:5])

            # Draw bounding box
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            box_w = w * img_w
            box_h = h * img_h

            rect = patches.Rectangle(
                (x1, y1), box_w, box_h, linewidth=2, edgecolor="lime", facecolor="none"
            )
            ax.add_patch(rect)

            # Draw keypoints
            kpts = parts[5:]
            keypoints = []
            for i in range(0, len(kpts), 3):
                if i + 2 < len(kpts):
                    kx = float(kpts[i]) * img_w
                    ky = float(kpts[i + 1]) * img_h
                    kv = float(kpts[i + 2])
                    keypoints.append((kx, ky, kv))

            # Draw skeleton
            for i, j in SKELETON:
                if i < len(keypoints) and j < len(keypoints):
                    kp1, kp2 = keypoints[i], keypoints[j]
                    if kp1[2] > 0 and kp2[2] > 0:
                        ax.plot(
                            [kp1[0], kp2[0]],
                            [kp1[1], kp2[1]],
                            "c-",
                            linewidth=1.5,
                            alpha=0.7,
                        )

            # Draw keypoint dots
            for i, (kx, ky, kv) in enumerate(keypoints):
                if kv > 0:
                    color = "red" if i < 7 else ("yellow" if i < 9 else "cyan")
                    ax.plot(kx, ky, "o", color=color, markersize=5)

        ax.axis("off")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to: {output_path}")

    plt.show()
    return fig


if __name__ == "__main__":
    import sys

    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "./cow_yolo_dataset"

    # Validate
    is_valid = validate_dataset(dataset_dir)

    if is_valid:
        print("\n" + "=" * 60)
        print("GENERATING SAMPLE VISUALIZATION...")
        print("=" * 60)
        visualize_samples(
            dataset_dir,
            num_samples=4,
            output_path=f"{dataset_dir}/validation_samples.png",
        )
