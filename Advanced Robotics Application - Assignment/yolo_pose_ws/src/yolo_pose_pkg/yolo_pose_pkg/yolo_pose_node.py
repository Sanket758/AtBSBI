import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import sys
import numpy as np
import torch


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class YoloPoseNode(Node):
    def __init__(self):
        super().__init__('yolo_pose_node')
        
        # Parameters
        self.declare_parameter('model_path', 'yolov8n-pose.pt')
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('device', DEVICE) # 'cpu' or 'cuda:0'
        
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        device = self.get_parameter('device').get_parameter_value().string_value

        self.get_logger().info(f'Initializing YOLO Pose Node...')
        self.get_logger().info(f'Model: {model_path}')
        self.get_logger().info(f'Topic: {camera_topic}')
        self.get_logger().info(f'Device: {device}')

        # Load YOLO Model
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            self.get_logger().error(f'Failed to load YOLO model: {e}')
            sys.exit(1)

        self.bridge = CvBridge()

        # Subscribers and Publishers
        self.subscription = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            10
        )

        # Publisher for annotated image
        self.publisher_ = self.create_publisher(Image, '/yolo_pose/annotated', 10)
        
        
        self.get_logger().info('Yolo Pose Node ready. Waiting for images...')

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV Image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CvBridge conversion error: {e}')
            return

        # Run Inference
        # verbose=False reduces console spam
        results = self.model(cv_image, verbose=False, device=self.get_parameter('device').get_parameter_value().string_value)

        # Annotate Frame
        # results[0].plot() returns the image with boxes and keypoints drawn
        annotated_frame = results[0].plot()

        # Resize before publishing for faster transport.
        # annotated_frame = cv2.resize(annotated_frame, (640, 480))

        # Publish Result
        try:
            output_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
            output_msg.header = msg.header # Preserve timestamp and frame_id
            self.publisher_.publish(output_msg)
        except Exception as e:
            self.get_logger().error(f'Publishing error: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = YoloPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
