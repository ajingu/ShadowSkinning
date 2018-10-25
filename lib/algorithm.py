import sys
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

from tf_pose.common import CocoColors

from lib.blob import extract_human_blob
from lib.binarization import to_binary_image
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


def calculate_nearest_neighbour_within_contour(img, contour_vertex_positions, body_part_positions):
    human_blob = extract_human_blob(img)
    black_pixel_positions = find_black_pixel_positions(human_blob)
    black_pixel_positions.extend(contour_vertex_positions)
    black_pixels_body_part_dict = {position: None for position in set(black_pixel_positions)}
    remaining_pixels_body_part_dict = black_pixels_body_part_dict.copy()

    for i, body_part_position in enumerate(body_part_positions):
        black_pixels_body_part_dict[body_part_position] = i
        del remaining_pixels_body_part_dict[body_part_position]

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
    #    cv2.circle(dst, position, 1, CocoColors[index])
    # plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    # plt.show()

    nearest_body_part_indices = [black_pixels_body_part_dict[position] for position in contour_vertex_positions]

    return nearest_body_part_indices


def calculate_k_nearest_neighbour_within_contour(img, contour_vertex_positions, body_part_positions, iteration=50):
    human_blob = extract_human_blob(img)
    black_pixel_positions = find_black_pixel_positions(human_blob)
    black_pixel_positions.extend(contour_vertex_positions)

    influence_dict = {black_pixel_position: np.full(len(body_part_positions), None)
                      for black_pixel_position in black_pixel_positions}

    for index, body_part_position in enumerate(body_part_positions):
        influence_dict[body_part_position][index] = 1

    count = 0
    for _ in range(iteration):  # 輪郭点それぞれに一個以上集まったら？
        count += 1

        for body_part_index in range(len(body_part_positions)):
            queue = []
            for pixel_position in black_pixel_positions:
                if influence_dict[pixel_position][body_part_index] is not None:
                    continue

                for delta_x, delta_y in [[0, -1], [0, 1], [1, 0], [-1, 0]]:
                    neighbour_pixel_position = (pixel_position[0] + delta_x, pixel_position[1] + delta_y)
                    if neighbour_pixel_position not in influence_dict:
                        continue

                    if influence_dict[neighbour_pixel_position][body_part_index] is not None:
                        queue.append([pixel_position, body_part_index, count])
                        break

            for queue_position, queue_index, queue_count in queue:
                influence_dict[queue_position][queue_index] = queue_count

    k_nearest_body_part_indices = []
    for i, contour_vertex_position in enumerate(contour_vertex_positions):
        influence_array = influence_dict[contour_vertex_position]
        # if i in [50, 250, 400]:
        #    print(influence_array)
        influence_array = np.where(influence_array == None, iteration + 1, influence_array)
        sorted_influence_array = np.argsort(influence_array)
        k_nearest_body_part_indices.append([[body_part_index, influence_array[body_part_index]]
                                            for i, body_part_index in enumerate(sorted_influence_array)
                                            if influence_array[body_part_index] < iteration + 1])

    # dst = img.copy()
    # for j in [50, 250, 400]:
    #    print(k_nearest_body_part_indices[j])
    #    cv2.circle(dst, contour_vertex_positions[j], 3, (0, 255, 0), cv2.FILLED)
    #plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    #plt.show()

    return k_nearest_body_part_indices


def find_black_pixel_positions(src):
    binary = to_binary_image(src)

    height, width = src.shape[:2]

    black_pixel_positions = [(x, y) for x in range(width) for y in range(height) if binary[y][x] == 0]

    return black_pixel_positions
