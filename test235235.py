import numpy as np
import cv2
import json

# Load the JSON file
with open(r"analyzie_jsons\val.json", "r") as file:
    data = json.load(file)

# Extract camera intrinsics
intrinsics = data["camera_data"]["intrinsics"]
camera_matrix = np.array([
    [intrinsics["fx"], 0, intrinsics["cx"]],
    [0, intrinsics["fy"], intrinsics["cy"]],
    [0, 0, 1]
])
dist_coeffs = np.zeros(5)  # Assuming no distortion

# Extract 3D points (local cuboid or world frame)
# Replace this with actual 3D points if available
object_data = data["objects"][0]
local_to_world_matrix = np.array(object_data["local_to_world_matrix"])
# Example: Generate 3D points for a cuboid (replace with actual points)
object_3d_points = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
    [0,0,0],
])  # Replace with actual cuboid points

# Transform 3D points to world frame (if needed)
object_3d_points_world = np.dot(local_to_world_matrix[:3, :3], object_3d_points.T).T + local_to_world_matrix[:3, 3]

# Extract 2D points
object_2d_points = np.array(object_data["projected_cuboid"])

print("3D Points Shape:", object_3d_points_world.shape)
print("2D Points Shape:", object_2d_points.shape)

# Solve PnP
success, rotation_vector, translation_vector = cv2.solvePnP(
    object_3d_points_world,  # 3D points in the world frame
    object_2d_points,        # Corresponding 2D points in the image
    camera_matrix,           # Camera intrinsic matrix
    dist_coeffs              # Distortion coefficients
)

if success:
    # Convert rotation vector to quaternion
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    quaternion = cv2.RQDecomp3x3(rotation_matrix)[0]
    print("Translation Vector:", translation_vector.ravel())
    print("Rotation Quaternion:", quaternion)
else:
    print("PnP solution failed.")