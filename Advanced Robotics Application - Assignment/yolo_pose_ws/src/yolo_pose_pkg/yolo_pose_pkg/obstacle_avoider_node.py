#!/usr/bin/env python3
"""Simple obstacle avoidance controller for the Spot model."""

import math
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


class ObstacleAvoider(Node):
    STATE_FORWARD = 0
    STATE_TURN = 1

    def __init__(self):
        super().__init__("obstacle_avoider")

        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("forward_speed", 0.18)
        self.declare_parameter("turn_speed", 0.45)
        self.declare_parameter("min_distance", 1.00)
        self.declare_parameter("clear_distance", 1.25)
        self.declare_parameter("front_angle_deg", 35.0)
        self.declare_parameter("scan_timeout_s", 0.75)

        scan_topic = self.get_parameter("scan_topic").value
        cmd_topic = self.get_parameter("cmd_topic").value
        self.forward_speed = float(self.get_parameter("forward_speed").value)
        self.turn_speed = float(self.get_parameter("turn_speed").value)
        self.min_distance = float(self.get_parameter("min_distance").value)
        self.clear_distance = float(self.get_parameter("clear_distance").value)
        self.front_angle_rad = math.radians(
            float(self.get_parameter("front_angle_deg").value)
        )
        self.scan_timeout = float(self.get_parameter("scan_timeout_s").value)

        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, scan_topic, self.scan_callback, qos_profile_sensor_data
        )

        self.state = self.STATE_FORWARD
        self.turn_direction = 1.0
        self.last_scan_time = time.time()
        self.watchdog = self.create_timer(0.20, self.watchdog_callback)

        self.get_logger().info("=" * 50)
        self.get_logger().info("OBSTACLE AVOIDER NODE")
        self.get_logger().info("=" * 50)
        self.get_logger().info(f"Scan topic: {scan_topic}")
        self.get_logger().info(f"Cmd topic: {cmd_topic}")
        self.get_logger().info(
            f"Forward speed: {self.forward_speed:.2f} m/s | Turn speed: {self.turn_speed:.2f} rad/s"
        )
        self.get_logger().info(
            f"Distance thresholds: min={self.min_distance:.2f}m clear={self.clear_distance:.2f}m"
        )
        self.get_logger().info("State: FORWARD")

    def scan_callback(self, msg: LaserScan):
        self.last_scan_time = time.time()

        valid = []
        for i, rng in enumerate(msg.ranges):
            if not math.isfinite(rng):
                continue
            if rng < msg.range_min or rng > msg.range_max:
                continue
            angle = msg.angle_min + i * msg.angle_increment
            valid.append((angle, rng))

        if not valid:
            self.publish_stop()
            return

        front = [r for a, r in valid if abs(a) <= self.front_angle_rad]
        left = [r for a, r in valid if 0.35 <= a <= 1.4]
        right = [r for a, r in valid if -1.4 <= a <= -0.35]

        front_min = min(front) if front else float("inf")
        left_min = min(left) if left else float("inf")
        right_min = min(right) if right else float("inf")

        cmd = Twist()

        if self.state == self.STATE_FORWARD:
            if front_min < self.min_distance:
                self.state = self.STATE_TURN
                self.turn_direction = 1.0 if left_min > right_min else -1.0
                self.get_logger().info(
                    f"State: FORWARD -> TURN (front={front_min:.2f}m, "
                    f"turn={'left' if self.turn_direction > 0 else 'right'})"
                )
                cmd.angular.z = self.turn_direction * self.turn_speed
            else:
                cmd.linear.x = self.forward_speed

        elif self.state == self.STATE_TURN:
            if front_min > self.clear_distance:
                self.state = self.STATE_FORWARD
                self.get_logger().info(
                    f"State: TURN -> FORWARD (front={front_min:.2f}m clear)"
                )
                cmd.linear.x = self.forward_speed
            else:
                cmd.angular.z = self.turn_direction * self.turn_speed

        self.cmd_pub.publish(cmd)

    def watchdog_callback(self):
        if time.time() - self.last_scan_time > self.scan_timeout:
            self.publish_stop()

    def publish_stop(self):
        self.cmd_pub.publish(Twist())

    def destroy_node(self):
        self.publish_stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoider()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
