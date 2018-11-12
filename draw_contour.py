import sys

import cv2
import matplotlib.pyplot as plt

from lib.contour import find_human_contour, draw_contour

if __name__ == "__main__":
    src = cv2.imread("./images/Shadow_500.jpg")
    #src = src.transpose(1, 0, 2)[::-1]
    dst = src.copy()
    human_contour = find_human_contour(src)
    if human_contour is None:
        print("Not a human contour has detected in the image.")
        sys.exit(0)

    draw_contour(dst, human_contour)

    plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    plt.show()
    #cv2.imshow("contour", dst)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
