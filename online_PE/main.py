import sys
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_dir)
import cv2
import numpy as np
import os
from common2.dope_node import DopeNode
import src.utils as ut

def main():
    opt = ut.parse_arguments(base_dir)
    # Load configurations and prepare the output folder
    config, camera_info = ut.load_config_files(opt.config, opt.camera)
    ut.prepare_output_folder(opt.outf)    

    # Load images and weights
    try:
        weight = ut.load_weight(opt.weights)        
    except FileNotFoundError as e:
        print(e)
        return        
    dope_node = DopeNode(config, weight, opt.parallel, opt.object)

    imgs = []
    imgsname = []
            
    ut.process_images(dope_node, imgs, imgsname, camera_info, opt.outf, weight, opt.debug)

main()