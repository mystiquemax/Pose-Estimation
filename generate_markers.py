import numpy as np
import cv2
from utils import get_aruco_dict

ARUCO_DICT = get_aruco_dict()
TYPE = "DICT_4X4_100"
ID = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
OUT_DIR = "data/markers"


def generate(TYPE, ID, OUT_DIR, visualize=False, save=True):
    arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[TYPE])
    tag = np.zeros((300, 300, 1), dtype="uint8")
    cv2.aruco.drawMarker(arucoDict, ID, 300, tag, 1)
    if save:
       cv2.imwrite(f"{OUT_DIR}/{TYPE}-{ID}.png", tag)
    if visualize:
       cv2.imshow("ArUCo Tag", tag)
       cv2.waitKey(0)


def print_info(TYPE):
    arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[TYPE])
    print(arucoDict)

if __name__ == "__main__":
    for id in ID:
        print_info(TYPE)
        generate(TYPE, id, OUT_DIR)
