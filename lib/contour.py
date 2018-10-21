import cv2


def find_contours_and_hierarchy(src):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy


def draw_contour_except_outer_rectangle(img, contours, hierarchy):
    for i in range(len(contours)):
        if hierarchy[0][i][2] == -1:
            cv2.drawContours(img, contours, i, (0, 0, 255), 1)
    return img
