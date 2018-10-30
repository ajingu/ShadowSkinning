import cv2
import numpy as np

from lib.binarization import to_binary_inv_image


def extract_human_blob(src):
    dst = src.copy()
    binary_inv = to_binary_inv_image(src)
    number_of_labels, lab = cv2.connectedComponents(binary_inv)
    hist, _ = np.histogram(lab.flatten(), bins=number_of_labels - 1, range=(1, number_of_labels - 1))
    max_area_label = hist.argmax() + 1
    height, width = lab.shape[:2]

    for y in range(height):
        for x in range(width):
            if lab[y][x] == max_area_label:
                dst[y][x] = [0, 0, 0]
            else:
                dst[y][x] = [255, 255, 255]
    return dst
