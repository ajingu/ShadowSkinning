import cv2

from lib.binarization import to_binary_inv_image
import matplotlib.pyplot as plt


def find_human_contour(src, binary_thresh=240, maximum_inner_blob_area=1000):
    binary_inv = to_binary_inv_image(src, binary_thresh)
    binary_inv[-1, :] = 0
    # plt.gray()
    # plt.title("blob")
    # plt.imshow(binary_inv)
    # plt.show()
    _, contours, hierarchy = cv2.findContours(binary_inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # print(hierarchy)

    hierarchy = hierarchy[0]

    max_index = -1
    max_area = 0
    human_contour = None

    for i, contour in enumerate(contours):
        # skip inner blobs
        if hierarchy[i][3] > -1:
            continue

        area = cv2.contourArea(contour)
        if max_area < area:
            max_index = i
            max_area = area
            human_contour = contour

    # print(max_index)

    # skip blobs which include inner blobs
    if max_index == -1:
        return None

    # skip contours which include big inner blobs
    child_index = hierarchy[max_index][2]
    while child_index > -1:
        if cv2.contourArea(contours[child_index]) > maximum_inner_blob_area:
            print("Too big inner blobs.")
            return None

        child_index = hierarchy[child_index][0]

    return human_contour


def draw_contour(img, contour):
    cv2.drawContours(img, [contour], 0, (0, 0, 255), 1)
