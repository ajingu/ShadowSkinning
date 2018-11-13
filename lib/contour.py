import cv2
import numpy as np

from lib.blob import extract_human_blob
from lib.binarization import to_binary_inv_image
import matplotlib.pyplot as plt


def calc_contour_area(contour, hierarchy, index):
    if hierarchy[0][index][3] > -1:
        return 0
    area = cv2.contourArea(contour)
    return area


vfunc = np.vectorize(calc_contour_area)


def find_human_contour(src):
    # human_blob = extract_human_blob(src)
    # plt.gray()
    # plt.title("blob")
    # plt.imshow(human_blob)
    # plt.show()
    # if human_blob is None:
    #    return None
    binary_inv = to_binary_inv_image(src)
    binary_inv = cv2.bilateralFilter(binary_inv, 9, 75, 10)
    binary_inv[-1, :] = 0
    plt.gray()
    plt.title("blob")
    plt.imshow(binary_inv)
    plt.show()
    _, contours, hierarchy = cv2.findContours(binary_inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # dst = cv2.drawContours(src, contours, 2, (0, 0, 255))
    print(hierarchy)
    # cv2.imshow("contour", dst)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    max_index = -1
    max_area = 0
    human_contour = None

    for i, contour in enumerate(contours):
        # skip inner blobs
        if hierarchy[0][i][3] > -1:
            continue

        area = cv2.contourArea(contour)
        if max_area < area:
            max_index = i
            max_area = area
            human_contour = contour

    print(max_index)

    # skip blobs which include inner blobs
    if max_index == -1:
        return None

    if hierarchy[0][max_index][2] > -1:
        return None

    return human_contour


def draw_contour(img, contour):
    cv2.drawContours(img, [contour], 0, (0, 0, 255), 1)
