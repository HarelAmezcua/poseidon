import os
import json
import numpy as np
from itertools import permutations
from src.cuboid import Cuboid3d
from src.cuboid_pnp_solver import CuboidPNPSolver
from src.auxiliar import parse_arguments, load_config_files

def main():
    options = parse_arguments()
    config, camera_info = load_config_files(options.config, options.camera)

    dimensions = [6.4513998031616211, 14.860799789428711, 4.3368000984191895]
    folder_path = r"C:\github\poseidon\dataset\test_frame_images"  # Assuming folder path is passed as an argument

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as f:
                data = json.load(f)

            camera_matrix = np.array([
                [data["camera_data"]["intrinsics"]["fx"], 0, data["camera_data"]["intrinsics"]["cx"]],
                [0, data["camera_data"]["intrinsics"]["fy"], data["camera_data"]["intrinsics"]["cy"]],
                [0, 0, 1]
            ])

            for obj in data["objects"]:
                if obj["class"] == "Ketchup":
                    points = obj["projected_cuboid"]
                    
                    pnp_solver = CuboidPNPSolver("Ketchup", cuboid3d=Cuboid3d(list(dimensions)))
                    pnp_solver.set_camera_intrinsic_matrix(camera_matrix)
                    pnp_solver.set_dist_coeffs(np.zeros(5))

                    location, quaternion, projected_points = pnp_solver.solve_pnp(points)
                    print(f"File: {file_name}, Dimensions: {dimensions} -> Location: {np.array(quaternion)}")

if __name__ == "__main__":
    main()
