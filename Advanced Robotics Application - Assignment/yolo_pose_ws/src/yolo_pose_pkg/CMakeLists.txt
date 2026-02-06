cmake_minimum_required(VERSION 3.8)
project(yolo_pose_pkg)

# Compiler settings
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ============================================
# FIND DEPENDENCIES
# ============================================
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# ============================================
# PYTHON PACKAGE INSTALL
# ============================================
# Install the python package into the same site-packages directory used by
# rosidl_generator_py so `yolo_pose_pkg` and `yolo_pose_pkg.msg` coexist.
ament_get_python_install_dir(PYTHON_INSTALL_DIR)
install(
  DIRECTORY yolo_pose_pkg/
  DESTINATION ${PYTHON_INSTALL_DIR}/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

# ============================================
# GENERATE CUSTOM MESSAGES
# ============================================
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/CowPoseKeypoints.msg"
  "msg/CowPoseArray.msg"
  DEPENDENCIES std_msgs
)

# ============================================
# INSTALL PYTHON EXECUTABLES
# ============================================
install(PROGRAMS
  scripts/yolo_pose_node
  scripts/cow_pose_node
  scripts/obstacle_avoider_node
  DESTINATION lib/${PROJECT_NAME}
)

# ============================================
# INSTALL LAUNCH FILES
# ============================================
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
  PATTERN "__pycache__" EXCLUDE
)

# ============================================
# INSTALL CONFIG FILES
# ============================================
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# ============================================
# INSTALL WORLD FILES
# ============================================
install(DIRECTORY worlds/
  DESTINATION share/${PROJECT_NAME}/worlds
)

# ============================================
# INSTALL MODEL FILES
# ============================================
install(DIRECTORY models/
  DESTINATION share/${PROJECT_NAME}/models
)

# ============================================
# INSTALL RESOURCE INDEX MARKER
# ============================================
install(FILES resource/${PROJECT_NAME}
  DESTINATION share/ament_index/resource_index/packages
)

# ============================================
# TESTING
# ============================================
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# ============================================
# EXPORT
# ============================================
ament_export_dependencies(rosidl_default_runtime)
ament_package()
