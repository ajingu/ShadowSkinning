import cv2


def to_binary_image(src):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def to_binary_inv_image(src):
    binary = to_binary_image(src)
    binary_inv = cv2.bitwise_not(binary)
    return binary_inv
