import os
import cv2
import yaml
import argparse
import glob
from pygrabber.dshow_graph import FilterGraph
import numpy as np

def parse_arguments(base_dir: str = ""): 
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--outf", default=os.path.join(base_dir, 'output'),
                        help="Where to store the output images and inference results.")
    parser.add_argument("--config", default= os.path.join(base_dir,'other','config','config_pose.yaml'),
                        help="Path to inference config file.")
    parser.add_argument("--camera", default=os.path.join(base_dir,'other','config','camera_info.yaml'),
                        help="Path to camera info file.")
    parser.add_argument("--weights", "-w", default=os.path.join(base_dir,'other','weights'),
                        help="Path to weights or folder containing weights.")
    parser.add_argument("--parallel", action="store_true",
                        help="Specify if weights were trained using DDP.")
    parser.add_argument("--exts", nargs="+", type=str, default=["png"],
                        help="Extensions for images to use (e.g., png jpg).")
    parser.add_argument("--object", default="Ketchup", help="Name of class to run detections on.")
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


def load_weight(weights_path):
    """Load inference images and model weights."""
    files = glob.glob(os.path.join(weights_path, "*.pth"), recursive=True)
    if not files:
        raise FileNotFoundError(f"No .pth file found in: {weights_path}")
    return files[0]

def process_images(dope_node, img, camera_info, output_folder, weight, debug):
    """Run inference on a set of images."""        
    frame = img[..., ::-1].copy()  # Convert BGR to RGB
    img, (rvec,tvec) = dope_node.image_callback(
        img=frame,
        camera_info=camera_info,
        img_name="frame",
        output_folder=output_folder,
        weight=weight,
        debug=debug
    )

    #print("rvec:", rvec)
    #print("tvec:", tvec)

    # Convert from rgb to bgr
    img = img[..., ::-1].copy()
    return img, (rvec,tvec)

def get_camera():
    """Find and return the '5Mega Webcam' camera device."""
    
    graph = FilterGraph()
    devices = graph.get_input_devices()
    
    for camera_index, device_name in enumerate(devices):
        if device_name == '5Mega Webcam':
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open camera {device_name} at index {camera_index}")                            
            return cap
    
    raise RuntimeError("5Mega Webcam not found in available devices")