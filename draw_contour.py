import cv2

from lib.contour import find_contours_and_hierarchy, draw_contour_except_outer_rectangle

if __name__ == "__main__":
    src = cv2.imread("./images/shadow.jpg")
    dst = src.copy()
    contours, hierarchy = find_contours_and_hierarchy(src)
    dst = draw_contour_except_outer_rectangle(dst, contours, hierarchy)

    # cv2.imwrite("./images/contour.png", dst)
    cv2.imshow("contour", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
