import sys

import cv2

from lib.contour import find_human_contour, draw_contour

if __name__ == "__main__":
    src = cv2.imread("./watch_test_images/2018-11-11-00-09-26_360.jpg")
    src = src.transpose(1, 0, 2)[::-1]
    dst = src.copy()
    human_contour = find_human_contour(src)
    if human_contour is None:
        print("Not a human contour has detected in the image.")
        sys.exit(0)

    draw_contour(dst, human_contour)

    cv2.imshow("contour", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
