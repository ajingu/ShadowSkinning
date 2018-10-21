import cv2

from lib.contour import find_contours_and_hierarchy

if __name__ == "__main__":
    src = cv2.imread("./images/shadow.jpg")
    dst = src.copy()
    contours, hierarchy = find_contours_and_hierarchy(src)

    srcRect = (0, 0, src.shape[1], src.shape[0])

    for i in range(len(contours)):
        # remove outer rectangle
        if hierarchy[0][i][2] != -1:
            continue

        subdivision = cv2.Subdiv2D(srcRect)
        for j in range(len(contours[i])):
            subdivision.insert(tuple(contours[i][j][0]))

        triangleList = subdivision.getTriangleList()

        for t in triangleList:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])

            # skip triangle outside contour
            if cv2.pointPolygonTest(contours[i], ((t[0] + t[2] + t[4]) / 3, (t[1] + t[3] + t[5]) / 3), False) < 1:
                continue

            cv2.line(dst, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.line(dst, pt2, pt3, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.line(dst, pt3, pt1, (0, 255, 0), 1, cv2.LINE_AA)

    # cv2.imwrite("./images/mesh_division.png", dst)
    cv2.imshow("mesh_division", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
