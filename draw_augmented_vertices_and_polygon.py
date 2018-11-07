import sys

import cv2

from lib.contour import find_human_contour
from lib.draw import draw_triangle
from lib.shadow import Shadow
from lib.skeleton import SkeletonImplement

if __name__ == "__main__":
    src = cv2.imread("./images/shadow.jpg")

    human_contour = find_human_contour(src)
    if human_contour is None:
        print("Not a human contour has detected in the image.")
        sys.exit(0)

    skeletonImplement = SkeletonImplement()
    human = skeletonImplement.infer_skeleton(src)
    if human is None:
        print("Not a human has detected in the image.")
        sys.exit(0)

    shadow = Shadow(src.shape, human, human_contour, arrangement_interval=10)

    polygons_image = src.copy()
    for pt1_index, pt2_index, pt3_index in shadow.triangle_vertex_indices:
        draw_triangle(polygons_image,
                      shadow.vertex_positions[pt1_index],
                      shadow.vertex_positions[pt2_index],
                      shadow.vertex_positions[pt3_index])

    cv2.imshow("augmented_polygons", polygons_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
