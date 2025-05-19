import cv2
import numpy as np

def main():
    # Open the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return


    # Define the camera matrix and distortion coefficients (calibrate your camera for accurate results)
    camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((5, 1))  # Assuming no lens distortion

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize the frame to speed up processing (optional)

        # Detect ArUco markers
        corners, ids, _ = detector.detectMarkers(frame)

        if ids is not None:                    

            # Estimate pose of each marker
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.095, camera_matrix, dist_coeffs)

            for rvec, tvec in zip(rvecs, tvecs):

                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)  # Draw axis for the first marker

                # Print the transformation matrix
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                transformation_matrix = np.hstack((rotation_matrix, tvec.T))
                transformation_matrix = np.vstack((transformation_matrix, [0, 0, 0, 1]))
                print("Transformation Matrix:\n", transformation_matrix)

        # Display the frame
        cv2.imshow('Aruco Marker Detection', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()