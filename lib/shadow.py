import cv2

from lib.skeleton import SkeletonPart


class Shadow:
    def __init__(self, src_shape, human, human_contour, arrangement_interval=10):
        self.body_part_positions = []
        self.vertex_positions = []
        self.triangle_vertex_indices = []

        image_height, image_width = src_shape[:2]

        # body_part_positions
        for i in range(SkeletonPart.LAnkle.value + 1):
            if i not in human.body_parts.keys():
                self.body_part_positions.append((0, 0))
                continue

            body_part = human.body_parts[i]
            body_part_position = (int(body_part.x * image_width + 0.5), int(body_part.y * image_height + 0.5))
            self.body_part_positions.append(body_part_position)

        # vertex_positions
        subdivision = cv2.Subdiv2D((0, 0, image_width, image_height))
        for j in range(len(human_contour)):
            contour_point = tuple(human_contour[j][0])
            subdivision.insert(contour_point)
            self.vertex_positions.append(contour_point)

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
