import cv2
import numpy as np

from lib.binarization import to_binary_inv_image


# RGB -> GRAY
def extract_human_blob(src):
    binary = to_binary_inv_image(src)
    binary[-1, :] = 0

    number_of_labels, lab, data, _ = cv2.connectedComponentsWithStats(binary, connectivity=4)

    if number_of_labels < 2:
        print("The number of labels is less than 2.")
        return None

    max_area_label = np.argmax(data[1:, 4]) + 1
    lab = np.where(lab == max_area_label, 0, 255)
    return lab
