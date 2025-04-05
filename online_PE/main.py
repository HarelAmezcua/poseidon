import sys
import os
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_dir)
import cv2
import numpy as np
from src.dope_node_online import DopeNode
import src.auxiliar_v2 as ut

def main():
    opt = ut.parse_arguments(base_dir)

    cap = ut.get_camera()

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

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(os.path.join(opt.outf, 'output.mp4'), fourcc, 20.0, (640, 480))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        processed_frame = ut.process_images(dope_node, frame, camera_info, opt.outf, weight, opt.debug)

        # Write the frame to the video file
        out.write(processed_frame)

        # Display the frame
        cv2.imshow('Processed Frame', processed_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and video writer, and close all OpenCV windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()