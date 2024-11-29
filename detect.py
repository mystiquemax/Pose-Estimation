import cv2
import numpy as np
from utils import get_aruco_dict

ARUCO_DICT = get_aruco_dict()

# calibration data for camera on mobile robot
# MATRIX_COEFFS_PATH = "../data/calibration_data/calibration_matrix.npy"
# DIST_COEFFS_PATH = "../data/calibration_data/distortion_coefficients.npy"

# calibration data for Mech-Eye camera
MATRIX_COEFFS_PATH = "data/calibration_data/calibration_matrix_MechEye.npy"
DIST_COEFFS_PATH = "data/calibration_data/distortion_coefficients_MechEye.npy"

def detect_all_markers(image, tag_type):
    # verify that the supplied ArUCo tag exists and is supported by OpenCV
    if ARUCO_DICT.get(tag_type, None) is None:
        print(f"[INFO] ArUCo tag of '{tag_type}' is not supported")
        return [], None, []
    # load the ArUCo dictionary, grab the ArUCo parameters, and detect the markers
    arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[tag_type])
    arucoParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

    # Return type of cv2.aruco.detectMarkers: Each detected marker includes the position of its corners in the image (in their original order) and the id of the marker
    # Parameters of detectMarkers: the image containing arUcos, arucotype, parameters
    (corners, ids, rejected) = detector.detectMarkers(image)

    return corners, ids, rejected

# not used 
def visualize_detected_markers(image, corners, ids):
    if len(corners) > 0:
        # flatten the ArUco IDs list
        ids = ids.flatten()
        # loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
            # extract the marker corners (which are always returned in
            # top-left, top-right, bottom-right, and bottom-left order)
            corners = markerCorner.reshape((4, 2)) # 4 rows and 2 columns 
                                                   #
                                                   # All corners -> top-left, top-right, bottom-right and bottom-left
                                                   # [x0,y0]
                                                   # [x1,y1]
                                                   # [x2,y2]
                                                   # [x3,y3]
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            # draw the bounding box of the ArUCo detection (green colored border)
            # cv2.line draws a line on image beginning from top/bottomLeft/Right ending at top/bottomRight/Left in green with thickens 2
            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
            # compute and draw the center (x, y)-coordinates of the ArUco marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
            # draw the ArUco marker ID on the image using FONT_HERSHEY_SIMPLEX font
            cv2.putText(image, str(markerID),
                        (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
            print("[INFO] ArUco marker ID: {}".format(markerID))
            # show the output image
            cv2.imshow("Image", image)
            cv2.waitKey(0)


def estimate_poses(corners, ids, matrix_coefficients, distortion_coefficients, marker_size):
    rvecs, tvecs = [], []
    if len(corners) > 0:
    
        marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)

        for i in range(0, len(ids)):
            # Estimate pose of each marker (individuallly) and return the values rvec and tvec
            _, rvec, tvec = cv2.solvePnP(marker_points, corners[i], matrix_coefficients, distortion_coefficients)
            rvecs.append(rvec)
            tvecs.append(tvec)

    return rvecs, tvecs


def visualize_estimated_poses(image, corners, ids, matrix_coefficients, distortion_coefficients, rvecs, tvecs, marker_length):
    for i in range(0, len(ids)):
        # Draw a square around the markers
        cv2.aruco.drawDetectedMarkers(image, corners)
        # Draw red, green and blue axis on the arUco marker. last parameters is the length of the axis
        cv2.drawFrameAxes(image, matrix_coefficients, distortion_coefficients, rvecs[i], tvecs[i], marker_length/2)
        # write ID on it
        curr_con = corners[i].reshape((4, 2)) # create from the 1D array an array with 4 rows and 2 columns (4 points (x,y) for each corner)
        (topLeft, topRight, bottomRight, bottomLeft) = curr_con
        topRight = (int(topRight[0]), int(topRight[1]))
        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
        topLeft = (int(topLeft[0]), int(topLeft[1]))
        cv2.putText(image, str(ids[i][0]),
                    (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

    # the function draws an axis on the picture (for better coordination) #origin = [0,0,0] => axis begin #axes=[[[-0.0, -0.0, 1.0]]] => z-axis points towards the camera                    
    cv2.drawFrameAxes(image, matrix_coefficients, distortion_coefficients, np.zeros([1,1,3]), np.array([[[-0.0, -0.0, 1.0]]]),
                       marker_length/2)
    cv2.imshow('Estimated Poses', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return image


def get_marker_poses(image_path, tag_type, show=False):
    image = cv2.imread(image_path)
    # 3x3 matrix
    matrix_coeffs = np.load(MATRIX_COEFFS_PATH)
    # 5 parameters
    distortion_coeffs = np.load(DIST_COEFFS_PATH)
    
    # corners: corners of the arUCo in the 2D image
    # ids: ids of all the arUCos / list probably 
    # rejected: rejected shapes that were found and considered (for being arucos) but did not contain a valid marker // used for debugging purpose
    corners, ids, rejected = detect_all_markers(image, tag_type)

    # rvecs and tvecs are extrinsic parameters. Rotation and translation vectors, which translates a coordinates of a 3D point to a coordinate system.
    # estimates poses of the arUcos
    rvecs, tvecs = estimate_poses(corners, ids, matrix_coeffs, distortion_coeffs, 0.105)

    if show:
        visualize_estimated_poses(image, corners, ids, matrix_coeffs, distortion_coeffs, rvecs, tvecs, 0.105)
        

    # makes a dictionary containing [[id_0, rvec_0, tvec_0], ... ,[id _n,rvec_n, tvec_n]]
    out = []
    for i in range(len(ids)):
        out.append({
            "id": ids[i][0],
            "rvec": rvecs[i],
            "tvec": tvecs[i]
        })

    return out


if __name__ == "__main__":
    # Mech Eye scans 
    result = get_marker_poses(image_path="data/Mech_Eye/rgb_image_00000.png",
                        tag_type="DICT_4X4_100",
                        show=True)
    print(result)

