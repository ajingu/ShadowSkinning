import cv2


def find_contours_and_hierarchy(src):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy


def find_human_contour(contours, hierarchy):
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
