# CowPoseArray.msg
# Array of cow detections (for multiple cows in frame)

std_msgs/Header header
CowPoseKeypoints[] detections
int32 num_detections
float32 total_inference_time_ms