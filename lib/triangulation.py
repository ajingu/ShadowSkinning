import cv2

from lib.draw import draw_triangle


class SimpleTriangulation:
    def __init__(self, img, human_contour):
        self.human_contour = human_contour

        subdivision = cv2.Subdiv2D((0, 0, img.shape[1], img.shape[0]))
        for j in range(len(human_contour)):
            subdivision.insert(tuple(human_contour[j][0]))

        self.triangle_list = subdivision.getTriangleList()

    def draw_triangles(self, img):
        for t in self.triangle_list:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])

            triangle_center = ((t[0] + t[2] + t[4]) / 3, (t[1] + t[3] + t[5]) / 3)

            # skip triangle outside contour
            if cv2.pointPolygonTest(self.human_contour, triangle_center, False) < 1:
                continue

            draw_triangle(img, pt1, pt2, pt3)
