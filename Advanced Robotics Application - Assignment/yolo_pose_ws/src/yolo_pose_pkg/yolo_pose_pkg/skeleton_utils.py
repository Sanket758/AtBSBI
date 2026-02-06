"""
skeleton_utils.py
Cow skeleton visualization utilities for 26-keypoint pose estimation
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

# ============================================
# COW SKELETON DEFINITION (26 Keypoints)
# ============================================

KEYPOINT_NAMES = [
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
    "R_B_Knee",  # 16
    "L_B_Knee",  # 17
    "Throat",  # 18
    "Tail_Tip",  # 19
    "L_EarBase",  # 20
    "R_EarBase",  # 21
    "Mouth_Corner",  # 22
    "Nose_Tip",  # 23
    "Nose_Bridge",  # 24
    "Tail_Mid",  # 25
]

# Skeleton connections (joint pairs)
SKELETON = [
    # Head
    (23, 24),
    (24, 22),
    (22, 2),
    (23, 2),
    (24, 1),
    (24, 0),
    (1, 21),
    (0, 20),
    (18, 21),
    (18, 20),
    (18, 2),
    # Spine
    (18, 13),
    (13, 12),
    (12, 7),
    (7, 25),
    (25, 19),
    # Front legs
    (12, 8),
    (8, 14),
    (14, 3),
    (12, 9),
    (9, 15),
    (15, 4),
    # Back legs
    (7, 10),
    (10, 16),
    (16, 5),
    (7, 11),
    (11, 17),
    (17, 6),
    # Cross connections
    (8, 9),
    (10, 11),
]

# Color scheme by body part (BGR format)
COLORS = {
    "head": (255, 100, 100),  # Light blue
    "spine": (100, 255, 100),  # Light green
    "front_leg_r": (100, 100, 255),  # Light red
    "front_leg_l": (255, 100, 255),  # Light magenta
    "back_leg_r": (255, 255, 100),  # Light cyan
    "back_leg_l": (100, 255, 255),  # Light yellow
    "tail": (200, 200, 200),  # Light gray
}

# Map skeleton edges to body parts
EDGE_COLORS = {
    # Head edges
    (23, 24): "head",
    (24, 22): "head",
    (22, 2): "head",
    (23, 2): "head",
    (24, 1): "head",
    (24, 0): "head",
    (1, 21): "head",
    (0, 20): "head",
    (18, 21): "head",
    (18, 20): "head",
    (18, 2): "head",
    # Spine edges
    (18, 13): "spine",
    (13, 12): "spine",
    (12, 7): "spine",
    # Tail edges
    (7, 25): "tail",
    (25, 19): "tail",
    # Front right leg
    (12, 8): "front_leg_r",
    (8, 14): "front_leg_r",
    (14, 3): "front_leg_r",
    # Front left leg
    (12, 9): "front_leg_l",
    (9, 15): "front_leg_l",
    (15, 4): "front_leg_l",
    # Back right leg
    (7, 10): "back_leg_r",
    (10, 16): "back_leg_r",
    (16, 5): "back_leg_r",
    # Back left leg
    (7, 11): "back_leg_l",
    (11, 17): "back_leg_l",
    (17, 6): "back_leg_l",
    # Cross connections
    (8, 9): "spine",
    (10, 11): "spine",
}

# Keypoint colors by body part
KEYPOINT_COLORS = {
    # Head
    0: "head",
    1: "head",
    2: "head",
    18: "head",
    20: "head",
    21: "head",
    22: "head",
    23: "head",
    24: "head",
    # Spine
    12: "spine",
    13: "spine",
    # Tail
    7: "tail",
    19: "tail",
    25: "tail",
    # Front right leg
    3: "front_leg_r",
    8: "front_leg_r",
    14: "front_leg_r",
    # Front left leg
    4: "front_leg_l",
    9: "front_leg_l",
    15: "front_leg_l",
    # Back right leg
    5: "back_leg_r",
    10: "back_leg_r",
    16: "back_leg_r",
    # Back left leg
    6: "back_leg_l",
    11: "back_leg_l",
    17: "back_leg_l",
}


def draw_cow_skeleton(
    image: np.ndarray,
    keypoints: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    conf_threshold: float = 0.3,
    line_thickness: int = 2,
    point_radius: int = 4,
    draw_labels: bool = False,
    alpha: float = 0.8,
) -> np.ndarray:
    """
    Draw cow skeleton on image with color-coded body parts.

    Args:
        image: BGR image (H, W, 3)
        keypoints: (26, 2) or (26, 3) array of keypoints
                   If (26, 3), third column is confidence
        confidence: Optional (26,) array of confidence scores
        conf_threshold: Minimum confidence to draw keypoint
        line_thickness: Skeleton line thickness
        point_radius: Keypoint circle radius
        draw_labels: Whether to draw keypoint index labels
        alpha: Transparency for overlay (0-1)

    Returns:
        Annotated image
    """
    output = image.copy()
    overlay = image.copy()

    # Extract confidence if embedded in keypoints
    if keypoints.shape[1] == 3:
        confidence = keypoints[:, 2]
        keypoints = keypoints[:, :2]

    if confidence is None:
        confidence = np.ones(26)

    # Convert to int coordinates
    kpts = keypoints.astype(int)

    # Draw skeleton edges first (so points appear on top)
    for i, j in SKELETON:
        if confidence[i] >= conf_threshold and confidence[j] >= conf_threshold:
            pt1 = tuple(kpts[i])
            pt2 = tuple(kpts[j])

            # Skip if points are at origin (invalid)
            if pt1[0] <= 0 or pt1[1] <= 0 or pt2[0] <= 0 or pt2[1] <= 0:
                continue

            body_part = EDGE_COLORS.get((i, j), "spine")
            color = COLORS[body_part]

            cv2.line(overlay, pt1, pt2, color, line_thickness, cv2.LINE_AA)

    # Draw keypoints
    for idx in range(26):
        if confidence[idx] < conf_threshold:
            continue

        x, y = kpts[idx]
        if x <= 0 or y <= 0:
            continue

        body_part = KEYPOINT_COLORS.get(idx, "spine")
        color = COLORS[body_part]

        # Filled circle with border
        cv2.circle(overlay, (x, y), point_radius, color, -1, cv2.LINE_AA)
        cv2.circle(overlay, (x, y), point_radius, (0, 0, 0), 1, cv2.LINE_AA)

        # Optional label
        if draw_labels:
            cv2.putText(
                overlay,
                str(idx),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # Blend overlay with original
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    return output


def draw_detection_info(
    image: np.ndarray,
    bbox: Tuple[float, float, float, float],
    det_confidence: float,
    inference_time_ms: float,
    class_name: str = "cow",
) -> np.ndarray:
    """
    Draw bounding box and detection info on image.

    Args:
        image: BGR image
        bbox: (x1, y1, x2, y2) bounding box
        det_confidence: Detection confidence
        inference_time_ms: Inference time in milliseconds
        class_name: Class name to display

    Returns:
        Annotated image
    """
    output = image.copy()
    x1, y1, x2, y2 = map(int, bbox)

    # Draw bounding box
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw label background
    label = f"{class_name}: {det_confidence:.2f}"
    (label_w, label_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )
    cv2.rectangle(
        output, (x1, y1 - label_h - 10), (x1 + label_w + 5, y1), (0, 255, 0), -1
    )
    cv2.putText(
        output,
        label,
        (x1 + 2, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    # Draw FPS info
    fps = 1000.0 / inference_time_ms if inference_time_ms > 0 else 0
    fps_text = f"Inference: {inference_time_ms:.1f}ms ({fps:.1f} FPS)"
    cv2.putText(
        output,
        fps_text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return output


def extract_keypoints_from_results(results) -> List[dict]:
    """
    Extract keypoints from YOLO results object.

    Args:
        results: Ultralytics YOLO results object

    Returns:
        List of detection dicts with keys:
        - 'bbox': (x1, y1, x2, y2)
        - 'confidence': float
        - 'keypoints': (26, 2) array
        - 'keypoint_conf': (26,) array
    """
    detections = []

    if results[0].keypoints is None:
        return detections

    boxes = results[0].boxes
    keypoints = results[0].keypoints

    for i in range(len(boxes)):
        det = {
            "bbox": boxes.xyxy[i].cpu().numpy(),
            "confidence": float(boxes.conf[i].cpu().numpy()),
            "keypoints": keypoints.xy[i].cpu().numpy(),
            "keypoint_conf": (
                keypoints.conf[i].cpu().numpy()
                if keypoints.conf is not None
                else np.ones(26)
            ),
        }
        detections.append(det)

    return detections
