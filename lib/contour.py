import cv2

from lib.blob import extract_human_blob


def find_human_contour(src):
    human_blob = extract_human_blob(src)
    _, contours, hierarchy = cv2.findContours(human_blob, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

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
