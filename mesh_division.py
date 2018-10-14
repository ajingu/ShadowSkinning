import cv2 as cv

if __name__ == "__main__":
    src = cv.imread("./images/shadow.jpg")
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    _, contours, hierarchy = cv.findContours(thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    dst = src.copy()

    srcRect = (0, 0, src.shape[1], src.shape[0])

    for i in range(len(contours)):
        # remove outer rectangle
        if hierarchy[0][i][2] != -1:
            continue

        subdivision = cv.Subdiv2D(srcRect)
        for j in range(len(contours[i])):
            subdivision.insert(tuple(contours[i][j][0]))

        triangleList = subdivision.getTriangleList()

        for t in triangleList:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])

            # skip triangle outside contour
            if cv.pointPolygonTest(contours[i], ((t[0] + t[2] + t[4]) / 3, (t[1] + t[3] + t[5]) / 3), False) < 1:
                continue

            cv.line(dst, pt1, pt2, (0, 255, 0), 1, cv.LINE_AA)
            cv.line(dst, pt2, pt3, (0, 255, 0), 1, cv.LINE_AA)
            cv.line(dst, pt3, pt1, (0, 255, 0), 1, cv.LINE_AA)

    # cv.imwrite("./images/dst.png", dst)
    cv.imshow("dst", dst)
    cv.waitKey(0)
    cv.destroyAllWindows()
