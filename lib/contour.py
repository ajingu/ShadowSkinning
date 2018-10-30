import cv2

from lib.binarization import to_binary_image
from lib.blob import extract_human_blob


def _find_contours_and_hierarchy(src):
    binary = to_binary_image(src)
    _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy


def find_human_contour(src):
    dst = extract_human_blob(src)
    contours, hierarchy = _find_contours_and_hierarchy(dst)

    max_area = 0
    human_contour = None

    for i, contour in enumerate(contours):
        # remove outer triangle
        if hierarchy[0][i][2] > -1:
            continue

        area = cv2.contourArea(contour)
        if max_area < area:
            max_area = area
            human_contour = contour

    return human_contour


def draw_contour(img, contour):
    cv2.drawContours(img, [contour], 0, (0, 0, 255), 1)
