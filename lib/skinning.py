import sys

import cv2

from tf_pose.common import CocoPart

from lib.common import calculate_squared_distance


class NearestNeighbourSkinning:
    def __init__(self, src, human, human_contour):
        self.body_part_positions = []
        self.contour_vertex_positions = []
        self.triangle_vertex_indices = []
        self.nearest_body_part_indices = []
        self.influence = []

        image_height, image_width = src.shape[:2]

        # body_part_positions
        for i in range(CocoPart.Background.value):
            if i not in human.body_parts.keys():
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
        for i in range(len(self.contour_vertex_positions)):
            contour_vertex_position = self.contour_vertex_positions[i]
            tmp = sys.maxsize
            nearest_body_part_index = None

            for j in range(len(self.body_part_positions)):
                squared_distance = calculate_squared_distance(contour_vertex_position, self.body_part_positions[j])

                if tmp < squared_distance:
                    continue

                tmp = squared_distance
                nearest_body_part_index = j

            self.nearest_body_part_indices.append(nearest_body_part_index)

        # influence
        for i in range(len(self.nearest_body_part_indices)):
            self.influence.append(1)
