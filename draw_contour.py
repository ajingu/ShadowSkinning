import cv2

from lib.contour import find_human_contour, draw_contour

if __name__ == "__main__":
    src = cv2.imread("./images/shadow.jpg")
    dst = src.copy()
    human_contour = find_human_contour(src)
    draw_contour(dst, human_contour)

    # cv2.imwrite("./images/contour.png", dst)
    cv2.imshow("contour", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
