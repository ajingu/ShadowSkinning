import sys

import cv2

from lib.contour import find_human_contour
from lib.triangulation import SimpleTriangulation

if __name__ == "__main__":
    src = cv2.imread("./images/shadow.jpg")
    dst = src.copy()
    human_contour = find_human_contour(src)
    if human_contour is None:
        print("Not a human contour has detected in the image.")
        sys.exit(0)

    triangulation = SimpleTriangulation(src, human_contour)
    triangulation.draw_triangles(dst)

    cv2.imshow("mesh_division", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
