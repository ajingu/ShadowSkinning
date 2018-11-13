import sys

import cv2
import matplotlib.pyplot as plt

from lib.contour import find_human_contour, draw_contour

if __name__ == "__main__":
    src = cv2.imread("./watch_test_images/2018-11-11-00-09-26_360.jpg")
    dst = src.copy()
    human_contour = find_human_contour(src)
    if human_contour is None:
        print("Not a human contour has detected in the image.")
        sys.exit(0)

    draw_contour(dst, human_contour)

    plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    plt.show()
