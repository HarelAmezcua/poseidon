import sys
import os
import cv2
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dope_node_online import DopeNode
import src.auxiliar_v2 as ut
import time

camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1))  # Assuming no lens distortion

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    opt = ut.parse_arguments(base_dir)
    
    # Solicitar la ruta del video
    video_path = input("Ingrese la ruta del video a procesar: ")
    if not os.path.exists(video_path):
        print("Error: El archivo de video no existe.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: No se pudo abrir el archivo de video.")
        return

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

    # Obtener las dimensiones del video de entrada
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(os.path.join(opt.outf, 'output.mp4'), fourcc, fps, (frame_width, frame_height))

    prev_time = time.time()  # Initialize the previous time

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fin del video o error al leer el frame.")
            break

        # Measure the time before processing the frame
        start_time = time.time()

        processed_frame, (rvec, tvec) = ut.process_images(dope_node, frame, camera_info, opt.outf, weight, opt.debug)

        if rvec is not None and tvec is not None and len(rvec) > 0 and len(tvec) > 0:
            tvec = tvec / 100
            cv2.drawFrameAxes(processed_frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)  # Draw axis for the first marker

        # Write the frame to the video file
        out.write(processed_frame)

        # Display the frame
        cv2.imshow('Processed Frame', processed_frame)

        # Measure the time after processing the frame
        end_time = time.time()

        # Calculate and print the time difference (dt)
        dt = end_time - prev_time
        print(f"Time difference (dt) between frames: {dt:.4f} seconds")
        prev_time = end_time

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video and video writer, and close all OpenCV windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()