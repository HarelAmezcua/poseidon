import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common2.dope_node import DopeNode
from common2.auxiliar import parse_arguments, load_config_files, prepare_output_folder, load_images_and_weights, process_images

def main():
    opt = parse_arguments()

    # Load configurations and prepare the output folder
    config, camera_info = load_config_files(opt.config, opt.camera)
    prepare_output_folder(opt.outf)
    print(opt.outf)

    # Load images and weights
    try:
        imgs, imgsname, weights = load_images_and_weights(opt.data, opt.exts, opt.weights)
    except FileNotFoundError as e:
        print(e)
        return

    print(f"Found {len(weights)} weights and {len(imgs)} images for processing.")

    # Run inference for each weight
    for w_i, weight in enumerate(weights):
        print(f"Using weight {w_i + 1} of {len(weights)}: {weight}")
        dope_node = DopeNode(config, weight, opt.parallel, opt.object)
        process_images(dope_node, imgs, imgsname, camera_info, opt.outf, weight, opt.debug)
        print("------")

main()