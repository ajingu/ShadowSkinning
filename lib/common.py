import cv2


def draw_triangle(img, pt1, pt2, pt3):
    cv2.line(img, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.line(img, pt2, pt3, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.line(img, pt3, pt1, (0, 255, 0), 1, cv2.LINE_AA)


def calculate_squared_distance(pt1, pt2):
    return (pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2
