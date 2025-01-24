import os
import cv2
import yaml
import argparse
from common2.utils import loadimages_inference, loadweights

def parse_arguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--outf", default=r"D:\Github\my-pose-estimator\output",
                        help="Where to store the output images and inference results.")
    parser.add_argument("--data", default=r"D:\Github\my-pose-estimator\other\test_frame_images",
                        help="Folder for data images to load.")
    parser.add_argument("--config", default=r"D:\Github\my-pose-estimator\other\config\config_pose.yaml",
                        help="Path to inference config file.")
    parser.add_argument("--camera", default=r"D:\Github\my-pose-estimator\other\config\camera_info.yaml",
                        help="Path to camera info file.")
    parser.add_argument("--weights", "-w", default=r"D:\Github\my-pose-estimator\other\weights",
                        help="Path to weights or folder containing weights.")
    parser.add_argument("--parallel", action="store_true",
                        help="Specify if weights were trained using DDP.")
    parser.add_argument("--exts", nargs="+", type=str, default=["png"],
                        help="Extensions for images to use (e.g., png jpg).")
    parser.add_argument("--object", default="cracker", help="Name of class to run detections on.")
    parser.add_argument("--debug", action="store_true",
                        help="Generates debugging information.")

    return parser.parse_args()


def load_config_files(config_path, camera_path):
    """Load configuration and camera info files."""
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    with open(camera_path, 'r') as f:
        camera_info = yaml.load(f, Loader=yaml.FullLoader)
    return config, camera_info


def prepare_output_folder(output_folder):
    """Create the output folder if it doesn't exist."""
    os.makedirs(output_folder, exist_ok=True)


def load_images_and_weights(data_path, exts, weights_path):
    """Load inference images and model weights."""
    imgs, imgsname = loadimages_inference(data_path, extensions=exts)
    if not imgs or not imgsname:
        raise FileNotFoundError("No input images found. Check --data and --exts flags.")

    weights = loadweights(weights_path)
    if not weights:
        raise FileNotFoundError("No weights found. Check --weights flag.")
    return imgs, imgsname, weights


def process_images(dope_node, imgs, imgsname, camera_info, output_folder, weight, debug):
    """Run inference on a set of images."""
    for i, (img_path, img_name) in enumerate(zip(imgs, imgsname)):
        print(f"Processing frame {i + 1} of {len(imgs)}: {img_name}")
        frame = cv2.imread(img_path)[..., ::-1].copy()  # Convert BGR to RGB
        dope_node.image_callback(
            img=frame,
            camera_info=camera_info,
            img_name=img_name,
            output_folder=output_folder,
            weight=weight,
            debug=debug
        )
