import sys

import cv2

from tf_pose.common import CocoPart

from lib.algorithm import calculate_nearest_neighbour, calculate_nearest_neighbour_within_contour, \
    calculate_k_nearest_neighbour_within_contour


class Skinning:
    def __init__(self, src, human, human_contour, algorithm="nearest_neighbour"):
        self.body_part_positions = []
        self.contour_vertex_positions = []
        self.triangle_vertex_indices = []
        self.nearest_body_part_indices = []
        self.influence = []

        image_height, image_width = src.shape[:2]

        # body_part_positions
        for i in range(CocoPart.Background.value):
            if i not in human.body_parts.keys():
                self.body_part_positions.append((0,0))
                continue

            body_part = human.body_parts[i]
            body_part_position = (int(body_part.x * image_width + 0.5), int(body_part.y * image_height + 0.5))
            self.body_part_positions.append(body_part_position)

        # contour_vertex_positions
        subdivision = cv2.Subdiv2D((0, 0, image_width, image_height))
        for j in range(len(human_contour)):
            contour_point = tuple(human_contour[j][0])
            subdivision.insert(contour_point)
            self.contour_vertex_positions.append(contour_point)

        # triangle_vertex_indices
        contour_vertex_indices = dict((coord, index) for index, coord in enumerate(self.contour_vertex_positions))
        triangle_list = subdivision.getTriangleList()
        for t in triangle_list:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])

            triangle_center = (int((t[0] + t[2] + t[4]) / 3), int((t[1] + t[3] + t[5]) / 3))
            if cv2.pointPolygonTest(human_contour, triangle_center, False) < 1:
                continue

            self.triangle_vertex_indices.append([
                contour_vertex_indices[pt1],
                contour_vertex_indices[pt2],
                contour_vertex_indices[pt3]
            ])

        # nearest_body_part_indices
        if algorithm == "nearest_neighbour":
            self.nearest_body_part_indices = calculate_nearest_neighbour(self.contour_vertex_positions,
                                                                         self.body_part_positions)
        elif algorithm == "nearest_neighbour_within_contour":
            self.nearest_body_part_indices = calculate_nearest_neighbour_within_contour(src,
                                                                                        self.contour_vertex_positions,
                                                                                        self.body_part_positions)
        elif algorithm == "k_nearest_neighbour_within_contour":
            self.nearest_body_part_indices = calculate_k_nearest_neighbour_within_contour(src, self.contour_vertex_positions, self.body_part_positions)
        else:
            print("The algorithm name is not found.")
            sys.exit(1)

        # influence
        for i in range(len(self.nearest_body_part_indices)):
            self.influence.append(1)
