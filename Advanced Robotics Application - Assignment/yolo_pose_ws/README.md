# yolo_pose_ws

ROS2 Jazzy workspace for cow keypoint pose estimation in Gazebo Harmonic.

## Build

```bash
cd ~/Education/BSBI/Advanced\ Robotics\ Application/yolo_pose_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Run full simulation + bridge + pose node + RViz

```bash
cd ~/Education/BSBI/Advanced\ Robotics\ Application/yolo_pose_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch yolo_pose_pkg cow_pose_sim.launch.py \
  "model_path:=$PWD/yolo26n_with_kpt.pt" \
  device:=cuda:0 \
  headless:=false \
  use_nav:=false
```

Set `headless:=true` for server-only mode (faster, no Gazebo window).
Set `use_nav:=true` to enable autonomous robot motion.

## Key topics

- `/camera/image_raw` (from Gazebo camera rig)
- `/spot/camera/image_raw` (camera mounted on moving `spot_micro`)
- `/cow_pose/annotated` (pose overlay image)
- `/cow_pose/keypoints` (custom message, when import succeeds)
- `/scan` (lidar from `spot_micro`)
- `/cmd_vel` (obstacle avoidance velocity command to `spot_micro`)

## Quick checks

```bash
ros2 topic hz /camera/image_raw
ros2 topic hz /cow_pose/annotated
ros2 topic echo /cow_pose/keypoints --once
```

## Notes

- The world is now fully local (`model://cow`, `model://camera_rig`, `model://spot_micro`) and does not require Fuel downloads.
- If you run `gz sim` directly (outside ROS launch), export model path first:
  `export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:$PWD/src/yolo_pose_pkg/models`
