# This file is a modified version of the original DOPE code from the following repository:
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import numpy as np
from PIL import Image
import os
import json

# Importing the necessary classes from the common2 package
from common2.cuboid import Cuboid3d
from common2.cuboid_pnp_solver import CuboidPNPSolver
from common2.detector import ModelData, ObjectDetector
from common2.utils import Draw

class DopeNode(object):
    """ROS node that listens to image topic, runs DOPE, and publishes DOPE results"""

    def __init__(
        self,
        config,   # config yaml loaded eg dict
        weight,   # path to weight file
        parallel, # was it trained using DDP
        class_name,
    ):
        self.input_is_rectified = config["input_is_rectified"]
        self.downscale_height = config["downscale_height"]

        self.config_detect = lambda: None
        self.config_detect.mask_edges = 1
        self.config_detect.mask_faces = 1
        self.config_detect.vertex = 1
        self.config_detect.threshold = 0.00001
        self.config_detect.softmax = 1000
        self.config_detect.thresh_angle = config["thresh_angle"]
        self.config_detect.thresh_map = config["thresh_map"]
        self.config_detect.sigma = config["sigma"]
        self.config_detect.thresh_points = config["thresh_points"]

        # load network model, create PNP solver
        self.model = ModelData(
            name=class_name,
            net_path=weight,
            parallel=parallel
        )
        self.model.load_net_model()
        print("Model Loaded")

        try:
            self.draw_color = tuple(config["draw_colors"][class_name])
        except:
            self.draw_color = (0, 255, 0)

        self.dimension = tuple(config["dimensions"][class_name])
        self.class_id = config["class_ids"][class_name]

        self.pnp_solver = CuboidPNPSolver(
            class_name, cuboid3d=Cuboid3d(config["dimensions"][class_name])
        )
        self.class_name = class_name

        print("Ctrl-C to stop")

    def image_callback(
        self,
        img,
        camera_info,
        img_name,  # this is the name of the img file to save, it needs the .png at the end
        output_folder,  # folder where to put the output
        weight,
        debug=False
    ):
        # Update camera matrix and distortion coefficients
        if self.input_is_rectified:
            P = np.matrix(
                camera_info["projection_matrix"]["data"], dtype="float64"
            ).copy()
            P.resize((3, 4))
            camera_matrix = P[:, :3]
            dist_coeffs = np.zeros((4, 1))
        else:
            # TODO
            camera_matrix = np.matrix(camera_info.K, dtype="float64")
            camera_matrix.resize((3, 3))
            dist_coeffs = np.matrix(camera_info.D, dtype="float64")
            dist_coeffs.resize((len(camera_info.D), 1))

        # Downscale image if necessary
        height, width, _ = img.shape
        scaling_factor = float(self.downscale_height) / height
        if scaling_factor < 1.0:
            camera_matrix[:2] *= scaling_factor
            img = cv2.resize(
                img, (int(scaling_factor * width), int(scaling_factor * height))
            )

        self.pnp_solver.set_camera_intrinsic_matrix(camera_matrix)
        self.pnp_solver.set_dist_coeffs(dist_coeffs)

        # Copy and draw image
        img_copy = img.copy()
        im = Image.fromarray(img_copy)
        draw = Draw(im)

        # dictionary for the final output
        dict_out = {"camera_data": {}, "objects": []}

        # Detect object
        results, belief_imgs = ObjectDetector.detect_object_in_image(
            self.model.net, self.pnp_solver, img, self.config_detect,
            grid_belief_debug=True
        )
        
        #belief_imgs.show()

        # Publish pose and overlay cube on image
        for _, result in enumerate(results):
            if result["location"] is None:
                continue

            loc = result["location"]
            ori = result["quaternion"]

            dict_out["objects"].append(
                {
                    "class": self.class_name,
                    "location": np.array(loc).tolist(),
                    "quaternion_xyzw": np.array(ori).tolist(),
                    "projected_cuboid": np.array(result["projected_points"]).tolist(),
                }
            )

            # Draw the cube
            if None not in result["projected_points"]:
                points2d = []
                for pair in result["projected_points"]:
                    points2d.append(tuple(pair))
                draw.draw_cube(points2d, self.draw_color)

        # Save image
        print("Returbn Image as cv2 image")
        im = np.array(im)  
        return im