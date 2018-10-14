import cv2 as cv

if __name__ == "__main__":
    src = cv.imread("./images/shadow.jpg")
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    _, contours, hierarchy = cv.findContours(thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    dst = src.copy()

    # draw contour
    for i in range(len(contours)):
        if hierarchy[0][i][2] == -1:
            dst = cv.drawContours(dst, contours, i, (0, 0, 255), 1)

    cv.imshow("dst", dst)
    cv.waitKey(0)
    cv.destroyAllWindows()
