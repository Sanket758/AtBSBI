#!/usr/bin/env python3
"""
cow_pose_node.py
ROS2 node for cow pose estimation using custom-trained YOLOv8 model (26 keypoints)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import time
from pathlib import Path

from ultralytics import YOLO

# Import custom utilities
from .skeleton_utils import (
    draw_cow_skeleton,
    draw_detection_info,
    extract_keypoints_from_results,
    KEYPOINT_NAMES,
)

# Try to import custom message (will be available after building)
try:
    from yolo_pose_pkg.msg import CowPoseKeypoints, CowPoseArray

    CUSTOM_MSG_AVAILABLE = True
    CUSTOM_MSG_IMPORT_ERROR = ""
except ImportError as e:
    CUSTOM_MSG_AVAILABLE = False
    CUSTOM_MSG_IMPORT_ERROR = str(e)


class CowPoseNode(Node):
    """
    ROS2 Node for cow pose estimation.

    Subscribes to camera images, runs YOLO pose inference,
    publishes annotated images and structured keypoint data.
    """

    def __init__(self):
        super().__init__("cow_pose_node")

        # ============================================
        # PARAMETERS
        # ============================================
        self.declare_parameter("model_path", "yolo26n_with_kpt.pt")
        self.declare_parameter("camera_topic", "/camera/image_raw")
        self.declare_parameter(
            "device", "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.declare_parameter("conf_threshold", 0.5)
        self.declare_parameter("iou_threshold", 0.45)
        self.declare_parameter("draw_labels", False)
        self.declare_parameter("publish_rate_limit", 30.0)  # Max FPS

        # Get parameters
        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        camera_topic = (
            self.get_parameter("camera_topic").get_parameter_value().string_value
        )
        self.camera_topic = camera_topic
        self.device = self.get_parameter("device").get_parameter_value().string_value
        self.conf_threshold = (
            self.get_parameter("conf_threshold").get_parameter_value().double_value
        )
        self.iou_threshold = (
            self.get_parameter("iou_threshold").get_parameter_value().double_value
        )
        self.draw_labels = (
            self.get_parameter("draw_labels").get_parameter_value().bool_value
        )
        self.rate_limit = (
            self.get_parameter("publish_rate_limit").get_parameter_value().double_value
        )

        # ============================================
        # LOGGING
        # ============================================
        self.get_logger().info("=" * 50)
        self.get_logger().info("COW POSE ESTIMATION NODE")
        self.get_logger().info("=" * 50)
        self.get_logger().info(f"Model: {model_path}")
        self.get_logger().info(f"Device: {self.device}")
        self.get_logger().info(f"Camera Topic: {camera_topic}")
        self.get_logger().info(f"Confidence Threshold: {self.conf_threshold}")
        self.get_logger().info(f"IoU Threshold: {self.iou_threshold}")
        self.get_logger().info(f"Keypoints: 26 (Custom Cow Skeleton)")

        # ============================================
        # MODEL LOADING
        # ============================================
        try:
            self.get_logger().info("Loading YOLO model...")
            self.model = YOLO(model_path)

            # Warmup inference
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model(dummy, device=self.device, verbose=False)

            self.get_logger().info("Model loaded and warmed up!")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

        # ============================================
        # CV BRIDGE
        # ============================================
        self.bridge = CvBridge()

        # ============================================
        # QOS PROFILE (for sensor data)
        # ============================================
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ============================================
        # SUBSCRIBERS
        # ============================================
        self.subscription = self.create_subscription(
            Image, camera_topic, self.image_callback, sensor_qos
        )

        # ============================================
        # PUBLISHERS
        # ============================================
        # Annotated image
        self.pub_annotated = self.create_publisher(Image, "/cow_pose/annotated", 10)

        # Keypoint data (if custom message available)
        if CUSTOM_MSG_AVAILABLE:
            self.pub_keypoints = self.create_publisher(
                CowPoseArray, "/cow_pose/keypoints", 10
            )
            self.get_logger().info(
                "Custom message available: Publishing to /cow_pose/keypoints"
            )
        else:
            self.pub_keypoints = None
            self.get_logger().warn(
                f"Custom message not available ({CUSTOM_MSG_IMPORT_ERROR}). "
                "Build and source the workspace again."
            )

        # ============================================
        # RATE LIMITING
        # ============================================
        self.last_publish_time = 0.0
        self.min_publish_interval = 1.0 / self.rate_limit

        # ============================================
        # STATISTICS
        # ============================================
        self.frame_count = 0
        self.total_inference_time = 0.0
        self.first_frame_received = False
        self.no_image_warned = False
        self.start_time = time.time()
        self.stream_watchdog = self.create_timer(5.0, self._stream_watchdog_callback)

        self.get_logger().info("=" * 50)
        self.get_logger().info("Node ready. Waiting for images...")
        self.get_logger().info("=" * 50)

    def image_callback(self, msg: Image):
        """Process incoming camera image."""

        if not self.first_frame_received:
            self.first_frame_received = True
            self.get_logger().info(
                f"First image received on {self.camera_topic} "
                f"({msg.width}x{msg.height}, encoding={msg.encoding})"
            )

        # Rate limiting
        current_time = time.time()
        if current_time - self.last_publish_time < self.min_publish_interval:
            return

        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"CvBridge error: {e}")
            return

        # ============================================
        # INFERENCE
        # ============================================
        start_time = time.perf_counter()

        results = self.model(
            cv_image,
            device=self.device,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        inference_time = (time.perf_counter() - start_time) * 1000  # ms

        # Update statistics
        self.frame_count += 1
        self.total_inference_time += inference_time

        # ============================================
        # EXTRACT DETECTIONS
        # ============================================
        detections = extract_keypoints_from_results(results)

        # ============================================
        # VISUALIZE
        # ============================================
        annotated = cv_image.copy()

        for det in detections:
            # Draw skeleton
            kpts_with_conf = np.column_stack([det["keypoints"], det["keypoint_conf"]])
            annotated = draw_cow_skeleton(
                annotated,
                kpts_with_conf,
                conf_threshold=self.conf_threshold,
                draw_labels=self.draw_labels,
            )

            # Draw bbox and info
            annotated = draw_detection_info(
                annotated, det["bbox"], det["confidence"], inference_time
            )

        # If no detections, still show FPS
        if len(detections) == 0:
            fps = 1000.0 / inference_time if inference_time > 0 else 0
            cv2.putText(
                annotated,
                f"No cows detected | {inference_time:.1f}ms ({fps:.1f} FPS)",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        # ============================================
        # PUBLISH ANNOTATED IMAGE
        # ============================================
        try:
            output_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            output_msg.header = msg.header
            self.pub_annotated.publish(output_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish annotated image: {e}")

        # ============================================
        # PUBLISH KEYPOINT DATA
        # ============================================
        if self.pub_keypoints is not None and CUSTOM_MSG_AVAILABLE:
            self.publish_keypoints(
                msg.header, detections, cv_image.shape, inference_time
            )

        self.last_publish_time = current_time

    def _stream_watchdog_callback(self):
        """Emit a clear warning if no camera frames are arriving."""
        if self.first_frame_received or self.no_image_warned:
            return

        elapsed = time.time() - self.start_time
        self.get_logger().warn(
            f"No images received on {self.camera_topic} after {elapsed:.1f}s. "
            "Verify Gazebo is running and ros_gz_bridge is active for /camera/image_raw."
        )
        self.no_image_warned = True

    def publish_keypoints(
        self, header: Header, detections: list, img_shape: tuple, inference_time: float
    ):
        """Publish structured keypoint data."""

        array_msg = CowPoseArray()
        array_msg.header = header
        array_msg.num_detections = len(detections)
        array_msg.total_inference_time_ms = inference_time

        for det in detections:
            kp_msg = CowPoseKeypoints()
            kp_msg.header = header
            kp_msg.confidence = det["confidence"]
            kp_msg.bbox = det["bbox"].flatten().tolist()

            kp_msg.keypoint_x = det["keypoints"][:, 0].tolist()
            kp_msg.keypoint_y = det["keypoints"][:, 1].tolist()
            kp_msg.keypoint_conf = det["keypoint_conf"].tolist()
            kp_msg.keypoint_visible = [
                2 if c > self.conf_threshold else 0 for c in det["keypoint_conf"]
            ]

            kp_msg.image_width = img_shape[1]
            kp_msg.image_height = img_shape[0]
            kp_msg.inference_time_ms = inference_time

            array_msg.detections.append(kp_msg)

        self.pub_keypoints.publish(array_msg)

    def destroy_node(self):
        """Cleanup on shutdown."""
        if self.frame_count > 0:
            avg_time = self.total_inference_time / self.frame_count
            self.get_logger().info(f"Processed {self.frame_count} frames")
            self.get_logger().info(f"Average inference time: {avg_time:.2f}ms")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CowPoseNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
