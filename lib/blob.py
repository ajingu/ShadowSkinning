import cv2
import numpy as np

from lib.binarization import to_binary_inv_image


# RGB -> GRAY
def extract_human_blob(src):
    binary_inv = to_binary_inv_image(src)
    number_of_labels, lab = cv2.connectedComponents(binary_inv)

    if number_of_labels < 2:
        return None

    hist, _ = np.histogram(lab.flatten(), bins=number_of_labels - 1, range=(1, number_of_labels - 1))
    max_area_label = hist.argmax() + 1
    lab = np.where(lab == max_area_label, 0, 255)
    return lab
