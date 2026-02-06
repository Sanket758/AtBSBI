"""Launch Gazebo simulation + bridge + cow pose node."""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    GroupAction,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _default_model_path() -> str:
    """Prefer the user's custom cow model if present in workspace root."""
    ws_root = Path.home() / "Education/BSBI/Advanced Robotics Application/yolo_pose_ws"
    preferred = ws_root / "yolo26n_with_kpt.pt"
    fallback = ws_root / "yolo26s-pose.pt"
    if preferred.exists():
        return str(preferred)
    return str(fallback)


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("yolo_pose_pkg")
    world_file = os.path.join(pkg_share, "worlds", "farm_world.sdf")
    models_path = os.path.join(pkg_share, "models")
    rviz_config = os.path.join(pkg_share, "config", "cow_pose.rviz")

    existing_gz_path = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    if existing_gz_path:
        gz_resource_path = f"{models_path}:{existing_gz_path}"
    else:
        gz_resource_path = models_path

    set_gz_model_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=gz_resource_path,
    )

    declare_model_path = DeclareLaunchArgument(
        "model_path",
        default_value=_default_model_path(),
        description="Path to YOLO pose model",
    )
    declare_device = DeclareLaunchArgument(
        "device",
        default_value="cuda:0",
        description="Inference device (cuda:0 or cpu)",
    )
    declare_conf = DeclareLaunchArgument(
        "conf_threshold",
        default_value="0.5",
        description="Detection confidence threshold",
    )
    declare_iou = DeclareLaunchArgument(
        "iou_threshold",
        default_value="0.45",
        description="NMS IoU threshold",
    )
    declare_use_sim = DeclareLaunchArgument(
        "use_sim",
        default_value="true",
        description="Launch Gazebo simulation",
    )
    declare_use_rviz = DeclareLaunchArgument(
        "use_rviz",
        default_value="true",
        description="Launch RViz",
    )
    declare_headless = DeclareLaunchArgument(
        "headless",
        default_value="false",
        description="Run Gazebo server headless (-s); set true only when GUI is not needed",
    )
    declare_use_nav = DeclareLaunchArgument(
        "use_nav",
        default_value="false",
        description="Run obstacle avoidance controller for Spot robot",
    )
    declare_nav_forward_speed = DeclareLaunchArgument(
        "nav_forward_speed",
        default_value="0.18",
        description="Forward speed for obstacle avoider (m/s)",
    )
    declare_nav_turn_speed = DeclareLaunchArgument(
        "nav_turn_speed",
        default_value="0.45",
        description="Turn speed for obstacle avoider (rad/s)",
    )
    declare_nav_min_distance = DeclareLaunchArgument(
        "nav_min_distance",
        default_value="1.00",
        description="Minimum front obstacle distance before turning (m)",
    )

    gazebo_headless = ExecuteProcess(
        cmd=["gz", "sim", "-r", "-s", world_file],
        output="screen",
        condition=IfCondition(LaunchConfiguration("headless")),
    )

    gazebo_gui = ExecuteProcess(
        cmd=["gz", "sim", "-r", world_file],
        output="screen",
        condition=UnlessCondition(LaunchConfiguration("headless")),
    )

    # Use directed GZ->ROS bridges only to avoid topic echo loops.
    ros_gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image",
            "/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            "/spot/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image",
            "/spot/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            "/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
            "/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
        ],
        output="screen",
    )

    sim_stack = GroupAction(
        actions=[
            gazebo_headless,
            gazebo_gui,
            ros_gz_bridge,
        ],
        condition=IfCondition(LaunchConfiguration("use_sim")),
    )

    cow_pose_node = TimerAction(
        period=3.0,
        actions=[
            Node(
                package="yolo_pose_pkg",
                executable="cow_pose_node",
                name="cow_pose_node",
                output="screen",
                parameters=[
                    {
                        "model_path": LaunchConfiguration("model_path"),
                        "camera_topic": "/camera/image_raw",
                        "device": LaunchConfiguration("device"),
                        "conf_threshold": LaunchConfiguration("conf_threshold"),
                        "iou_threshold": LaunchConfiguration("iou_threshold"),
                        "draw_labels": False,
                        "publish_rate_limit": 30.0,
                    }
                ],
            )
        ],
    )

    obstacle_avoider_node = TimerAction(
        period=4.0,
        actions=[
            Node(
                package="yolo_pose_pkg",
                executable="obstacle_avoider_node",
                name="obstacle_avoider",
                output="screen",
                parameters=[
                    {
                        "scan_topic": "/scan",
                        "cmd_topic": "/cmd_vel",
                        "forward_speed": LaunchConfiguration("nav_forward_speed"),
                        "turn_speed": LaunchConfiguration("nav_turn_speed"),
                        "min_distance": LaunchConfiguration("nav_min_distance"),
                        "clear_distance": 1.25,
                        "front_angle_deg": 35.0,
                    }
                ],
                condition=IfCondition(LaunchConfiguration("use_nav")),
            )
        ],
    )

    static_world_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_static_tf",
        arguments=["0", "0", "0", "0", "0", "0", "world", "map"],
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_rviz")),
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_rviz")),
    )

    return LaunchDescription(
        [
            set_gz_model_path,
            declare_model_path,
            declare_device,
            declare_conf,
            declare_iou,
            declare_use_sim,
            declare_use_rviz,
            declare_headless,
            declare_use_nav,
            declare_nav_forward_speed,
            declare_nav_turn_speed,
            declare_nav_min_distance,
            sim_stack,
            cow_pose_node,
            obstacle_avoider_node,
            static_world_tf,
            rviz,
        ]
    )
