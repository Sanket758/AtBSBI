from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_webcam',
            default_value='false',
            description='Whether to launch the standard ROS 2 webcam node (cam2image)'
        ),
        DeclareLaunchArgument(
            'model',
            default_value='yolov8n-pose.pt',
            description='Path to the YOLO model file'
        ),
        
        # Standard ROS 2 webcam node (for testing)
        Node(
            package='image_tools',
            executable='cam2image',
            name='cam2image',
            output='screen',
            parameters=[{'width': 640, 'height': 480, 'frequency': 30.0}],
            condition=IfCondition(LaunchConfiguration('use_webcam')),
            remappings=[
                ('/image', '/camera/image_raw'),
            ]
        ),

        # Our YOLO Pose Node
        Node(
            package='yolo_pose_pkg',
            executable='yolo_pose_node',
            name='yolo_pose_node',
            output='screen',
            parameters=[
                {'model_path': LaunchConfiguration('model')},
                {'camera_topic': '/camera/image_raw'}
            ]
        ),

        # Moving to rviz2 for faster better image handling
        Node(
            package='image_transport',
            executable='republish',
            arguments=['raw', 'compressed'],
            remappings=[
                ('in', '/yolo_pose/annotated'),
                ('out', '/yolo_pose/annotated'),
            ],
            output='screen'
        ),

    ])
