import cv2
import numpy as np

def get_aruco_dict():
    ARUCO_DICT = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
        "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
        "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
        "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
        "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
    }
    return ARUCO_DICT


def save_calibration_data():
    # camera on mobile robot
    matrix = np.array([[845.1212278795325, 0., 658.2016571460368],
                      [0., 847.4904976312075, 514.4071735156942],
                      [ 0., 0., 1.]])
    distotion = np.array([[-0.2555195243217436, 0.19506876207592883, 0.0005719910672726553, -0.0008880216165534071, -0.181005072717146]])
    
    # Mech-Eye camera
    matrix_MechEye = np.array([[2421.874517126708, 0., 957.9707124630149],
                               [0., 2422.340102924828, 614.8579217015205],
                               [ 0., 0., 1.]])

    distotion_MechEye = np.array([[0.0, 0.0, 0.0, 0.0]])

    np.save("data/calibration_data/calibration_matrix.npy", matrix, allow_pickle=False)
    np.save("data/calibration_data/distortion_coefficients.npy", distotion, allow_pickle=False)

    np.save("data/calibration_data/calibration_matrix_MechEye.npy", matrix_MechEye, allow_pickle=False)
    np.save("data/calibration_data/distortion_coefficients_MechEye.npy", distotion_MechEye, allow_pickle=False)


if __name__ == "__main__":
    save_calibration_data()
