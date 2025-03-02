import os
import cv2
import yaml
import argparse
from common2.utils import loadimages_inference, loadweights
import glob

def parse_arguments(base_dir: str = ""): 
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--outf", default=os.path.join(base_dir, 'output'),
                        help="Where to store the output images and inference results.")
    parser.add_argument("--config", default= os.path.join(base_dir,'other','config','config_pose.yaml'),
                        help="Path to inference config file.")
    parser.add_argument("--camera", default=os.path.join(base_dir,'other','config','camera_info.yaml'),
                        help="Path to camera info file.")
    parser.add_argument("--weights", "-w", default=os.path.join(base_dir,' other','weights'),
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


def load_weight(weights_path):
    """Load inference images and model weights."""
    files = glob.glob(os.path.join(weights_path, "*.pth"), recursive=True)
    if not files:
        raise FileNotFoundError(f"No .pth file found in: {weights_path}")
    return files[0]


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
