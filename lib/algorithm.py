import random
import sys

import cv2
import matplotlib.pyplot as plt

from lib.blob import extract_human_blob
from lib.skeleton import SkeletonColors


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


def calculate_nearest_neighbour_within_contour(img, contour_vertex_positions, body_part_positions):
    human_blob = extract_human_blob(img)
    if human_blob is None:
        print("Human blob is not found in this image.")
        sys.exit(0)

    black_pixel_positions = [(x, y) for x in range(img.shape[1]) for y in range(img.shape[0]) if human_blob[y][x] == 0]
    black_pixel_positions.extend(contour_vertex_positions)
    black_pixels_body_part_dict = {position: None for position in set(black_pixel_positions)}
    remaining_pixels_body_part_dict = black_pixels_body_part_dict.copy()

    for i, body_part_position in enumerate(body_part_positions):
        black_pixels_body_part_dict[body_part_position] = i
        try:
            del remaining_pixels_body_part_dict[body_part_position]
        except KeyError:
            pass

    while len(remaining_pixels_body_part_dict) > 0:
        for pixel_position in list(remaining_pixels_body_part_dict):
            tmp = []

            for delta_x, delta_y in [[0, -1], [0, 1], [1, 0], [-1, 0]]:
                neighbour_pixel_position = (pixel_position[0] + delta_x, pixel_position[1] + delta_y)
                if neighbour_pixel_position in black_pixels_body_part_dict:
                    body_part_index = black_pixels_body_part_dict[neighbour_pixel_position]
                    if body_part_index is not None:
                        tmp.append(body_part_index)

            if len(tmp) > 0:
                black_pixels_body_part_dict[pixel_position] = random.choice(tmp)
                del remaining_pixels_body_part_dict[pixel_position]

    # dst = img.copy()
    # for position, index in black_pixels_body_part_dict.items():
    #    if index == 13:
    #        print("it's 13")
    #    cv2.circle(dst, position, 1, SkeletonColors[index])
    # plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    # plt.show()

    nearest_body_part_indices = [black_pixels_body_part_dict[position] for position in contour_vertex_positions]

    return nearest_body_part_indices


def calculate_squared_distance(pt1, pt2):
    return (pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2
