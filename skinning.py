import sys

import cv2
import matplotlib.pyplot as plt

from tf_pose.common import read_imgfile, CocoPart

from lib.common import calculate_squared_distance, draw_triangle
from lib.contour import find_contours_and_hierarchy
from lib.skeleton import SkeletonImplement


if __name__ == '__main__':
    src = read_imgfile("./images/shadow.jpg", None, None)
    dst = src.copy()

    # body parts coordinates
    skeletonImplement = SkeletonImplement()
    humans = skeletonImplement.infer_skeletons(src)
    image_h, image_w = src.shape[:2]
    body_part_centers = {}

    for human in humans:
        for i in range(CocoPart.Background.value):
            if i not in human.body_parts.keys():
                continue

            body_part = human.body_parts[i]
            body_part_center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
            body_part_centers[i] = body_part_center

    # contour points coordinates & triangle vertices
    contours, hierarchy = find_contours_and_hierarchy(src)

    contour_points = {}  # index -> coord
    triangle_vertices = []  # array of triangle vertices([pt1_index, pt2_index, pt3_index])

    triangle_centers = []

    for i in range(len(contours)):
        if hierarchy[0][i][2] != -1:
            continue

        subdivision = cv2.Subdiv2D((0, 0, src.shape[1], src.shape[0]))
        for j in range(len(contours[i])):
            contour_point = tuple(contours[i][j][0])
            subdivision.insert(contour_point)
            contour_points[j] = contour_point

        contour_points_inv = dict((coord, index) for index, coord in contour_points.items())

        triangleList = subdivision.getTriangleList()
        for t in triangleList:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])

            triangle_center = (int((t[0] + t[2] + t[4]) / 3), int((t[1] + t[3] + t[5]) / 3))
            if cv2.pointPolygonTest(contours[i], triangle_center, False) < 1:
                continue

            triangle_vertices.append([contour_points_inv[pt1], contour_points_inv[pt2], contour_points_inv[pt3]])
            triangle_centers.append(triangle_center)

    # influence(nearest neighbour)
    nearest_body_parts = {}  # triangle_index -> body_part_index

    for i in range(len(triangle_centers)):
        triangle_center = triangle_centers[i]
        tmp = sys.maxsize
        nearest_body_part_index = None

        for j in range(len(body_part_centers)):
            squared_distance = calculate_squared_distance(triangle_center, body_part_centers[j])

            if tmp < squared_distance:
                continue

            tmp = squared_distance
            nearest_body_part_index = j

        nearest_body_parts[i] = nearest_body_part_index

    # visualization
    for i in [166, 173]:
        cv2.circle(dst, triangle_centers[i], 1, (255, 0, 0))
        pt1, pt2, pt3 = [contour_points[index] for index in triangle_vertices[i]]
        draw_triangle(dst, pt1, pt2. pt3)
        cv2.circle(dst, body_part_centers[nearest_body_parts[i]], 3, (0, 0, 255), thickness=3, lineType=8, shift=0)

    plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    plt.show()
