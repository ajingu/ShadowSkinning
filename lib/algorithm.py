import sys

from lib.common import calculate_squared_distance


def calculate_nearest_neighbour(contour_vertex_positions, body_part_positions):
    nearest_body_part_indices = []

    for _, contour_vertex_position in enumerate(contour_vertex_positions):
        tmp = sys.maxsize
        nearest_body_part_index = None

        for j, body_part_position in enumerate(body_part_positions):
            squared_distance = calculate_squared_distance(contour_vertex_position, body_part_position)

            if tmp < squared_distance:
                continue

            tmp = squared_distance
            nearest_body_part_index = j

        nearest_body_part_indices.append(nearest_body_part_index)

    return nearest_body_part_indices


def calculate_nearest_neighbour_within_contour(binary_img, contour_vertex_positions, body_part_positions):
    black_pixels_body_part_dict = {}  # must find using some ways
    remaining_pixels_body_part_dict = black_pixels_body_part_dict.copy()
    number_of_remaining_pixels = len(remaining_pixels_body_part_dict)

    for i, body_part_position in enumerate(body_part_positions):
        black_pixels_body_part_dict[body_part_position] = i

    while number_of_remaining_pixels > 0:
        for pixel_position in remaining_pixels_body_part_dict.keys():
            # if up
            #   number_of_remaining_pixels -= 1
            #   continue
            # if left
            #   number_of_remaining_pixels -= 1
            #   continue
            # if down
            #   number_of_remaining_pixels -= 1
            #   continue
            # if right
            #   number_of_remaining_pixels -= 1
            #   continue
            pass

    nearest_body_part_indices = [black_pixels_body_part_dict[position] for position in contour_vertex_positions]

    return nearest_body_part_indices
