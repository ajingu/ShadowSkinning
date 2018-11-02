import cv2


def draw_triangle(img, pt1, pt2, pt3):
    cv2.line(img, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.line(img, pt2, pt3, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.line(img, pt3, pt1, (0, 255, 0), 1, cv2.LINE_AA)


def draw_circle(img, point, color=(0, 0, 255)):
    cv2.circle(img, point, 3, color, thickness=3, lineType=8, shift=0)



