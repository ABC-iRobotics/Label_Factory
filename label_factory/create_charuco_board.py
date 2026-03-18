import os
import numpy as np
import cv2
import yaml
import subprocess

def create_and_save_new_board(ARUCO_DICT,SQUARES_VERTICALLY,SQUARES_HORIZONTALLY,SQUARE_LENGTH,MARKER_LENGTH,LENGTH_PX,MARGIN_PX,SAVE_NAME):
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    size_ratio = SQUARES_HORIZONTALLY / SQUARES_VERTICALLY
    img = board.generateImage((LENGTH_PX, int(LENGTH_PX*size_ratio)), marginSize=10)
    cv2.imwrite(SAVE_NAME, img)


def main():
    path = subprocess.check_output("ros2 pkg prefix label_factory",shell = True, text = True)
    path = path.split("/install",1)[0]
    with open(os.path.join(os.getcwd(),'label_factory/config.yaml'), 'r') as file:
        config = yaml.safe_load(file)
    # ------------------------------
    ARUCO_DICT = cv2.aruco.DICT_4X4_1000                                # dictionary used for the aruco markers
    SQUARES_VERTICALLY = 37                                              # number of squares horizontally
    SQUARES_HORIZONTALLY = 25                                            # number of squares horizontally
    SQUARE_LENGTH = 0.035                                                # square length in meter
    MARKER_LENGTH = 0.02                                                # marker length in meter
    LENGTH_PX = 640                                                     # total length of the page in pixels
    MARGIN_PX = 20                                                      # size of the margin in pixels
    SAVE_NAME = str(config['path_and_name_of_the_charuco_image'])       # path and name of the image
    # ------------------------------
    create_and_save_new_board(ARUCO_DICT,SQUARES_VERTICALLY,SQUARES_HORIZONTALLY,SQUARE_LENGTH,MARKER_LENGTH,LENGTH_PX,MARGIN_PX,SAVE_NAME)
    
if __name__ == '__main__':
    main()
