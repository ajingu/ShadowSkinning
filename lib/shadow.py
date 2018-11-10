import time

import cv2
import numpy as np

from lib.skeleton import SkeletonPart


class Shadow:
    def __init__(self, src_shape, human, human_contour, arrangement_interval=10):
        self.body_part_positions = []
        self.triangle_vertex_indices = []

        image_height, image_width = src_shape[:2]

        firstmillis = int(round(time.time() * 1000))
        # body_part_positions
        for i in range(SkeletonPart.LAnkle.value + 1):
            if i not in human.body_parts.keys():
                self.body_part_positions.append((0, 0))
                continue

            body_part = human.body_parts[i]
            body_part_position = (int(body_part.x * image_width + 0.5), int(body_part.y * image_height + 0.5))
            self.body_part_positions.append(body_part_position)

        bodymillis = int(round(time.time() * 1000))
        print("body_part_positions: {}ms".format(bodymillis - firstmillis))

        # vertex_positions
        subdivision = cv2.Subdiv2D((0, 0, image_width, image_height))
        subdivision.insert(human_contour)
        self.vertex_positions = [tuple(contour_list[0]) for contour_list in human_contour]

        vertexmillis = int(round(time.time() * 1000))
        print("vertex_positions: {}ms".format(vertexmillis - bodymillis))

        # augment vertices
        start_x, start_y, width, height = cv2.boundingRect(human_contour)

        end_x = start_x + width
        end_y = start_y + height
        x = start_x
        y = start_y

        vertex_positions_set = set(self.vertex_positions)

        while True:
            y += arrangement_interval
            if y > end_y:
                break

            while True:
                x += arrangement_interval
                if x > end_x:
                    x = start_x
                    break

                point = (x, y)
                if point in vertex_positions_set:
                    continue
                if cv2.pointPolygonTest(human_contour, point, False) < 1:
                    continue

                subdivision.insert(point)
                self.vertex_positions.append(point)

        augmentationmillis = int(round(time.time() * 1000))
        print("augmentation: {}ms".format(augmentationmillis - vertexmillis))

        # triangle_vertex_indices
        vertex_indices = dict((coord, index) for index, coord in enumerate(self.vertex_positions))
        triangle_list = subdivision.getTriangleList()
        for t in triangle_list:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])

            triangle_center = (int((t[0] + t[2] + t[4]) / 3), int((t[1] + t[3] + t[5]) / 3))
            if cv2.pointPolygonTest(human_contour, triangle_center, False) < 1:
                continue

            self.triangle_vertex_indices.append([
                vertex_indices[pt1],
                vertex_indices[pt2],
                vertex_indices[pt3]
            ])

        trianglemillis = int(round(time.time() * 1000))
        print("triangle: {}ms".format(trianglemillis - augmentationmillis))
