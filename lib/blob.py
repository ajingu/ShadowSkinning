import cv2
import numpy as np

from lib.binarization import to_binary_image


# RGB -> GRAY
def extract_human_blob(src, binary_thresh):
    binary = to_binary_image(src, binary_thresh)
    binary[-1, :] = 255

    number_of_labels, lab, data, _ = cv2.connectedComponentsWithStats(binary)

    if number_of_labels < 2:
        print("The number of labels is less than 2.")
        return None

    max_area_label = np.argmax(data[1:, 4]) + 1
    lab = np.where(lab == max_area_label, 0, 255)
    return lab
