# CowPoseKeypoints.msg
# Custom message for 26-keypoint cow pose estimation

std_msgs/Header header

# Detection info
float32 confidence              # Detection confidence (0-1)
float32[4] bbox                 # Bounding box [x1, y1, x2, y2] in pixels

# 26 Keypoints - each with x, y, confidence
# Keypoint order matches training:
# 0:L_Eye, 1:R_Eye, 2:Chin, 3:R_F_Hoof, 4:L_F_Hoof, 5:R_B_Hoof, 6:L_B_Hoof,
# 7:Tail_Base, 8:R_Shoulder, 9:L_Shoulder, 10:R_Hip, 11:L_Hip, 12:Spine_Mid,
# 13:Withers, 14:R_F_Knee, 15:L_F_Knee, 16:R_B_Knee, 17:L_B_Knee, 18:Throat,
# 19:Tail_Tip, 20:L_EarBase, 21:R_EarBase, 22:Mouth_Corner, 23:Nose_Tip,
# 24:Nose_Bridge, 25:Tail_Mid

float32[26] keypoint_x          # X coordinates (pixels)
float32[26] keypoint_y          # Y coordinates (pixels)
float32[26] keypoint_conf       # Confidence per keypoint (0-1)
int8[26] keypoint_visible       # Visibility: 0=not visible, 1=occluded, 2=visible

# Metadata
int32 image_width
int32 image_height
float32 inference_time_ms       # For performance tracking