import sys

from lib.common import calculate_squared_distance


def calculate_nearest_neighbour(contour_vertex_positions, body_part_positions):
    nearest_body_part_indices = []

    for i in range(len(contour_vertex_positions)):
        contour_vertex_position = contour_vertex_positions[i]
        tmp = sys.maxsize
        nearest_body_part_index = None

        for j in range(len(body_part_positions)):
            squared_distance = calculate_squared_distance(contour_vertex_position, body_part_positions[j])

            if tmp < squared_distance:
                continue

            tmp = squared_distance
            nearest_body_part_index = j

        nearest_body_part_indices.append(nearest_body_part_index)

    return nearest_body_part_indices


def calculate_nearest_neighbour_within_contour(img, contour_vertex_positions, body_part_positions):
    nearest_body_part_indices = []

    return nearest_body_part_indices
