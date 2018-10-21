import cv2
import matplotlib.pyplot as plt

from tf_pose.common import read_imgfile

from lib.common import draw_triangle
from lib.contour import find_contours_and_hierarchy, find_human_contour
from lib.skeleton import SkeletonImplement
from lib.skinning import NearestNeighbourSkinning


if __name__ == '__main__':
    src = read_imgfile("./images/shadow.jpg", None, None)
    dst = src.copy()
    contours, hierarchy = find_contours_and_hierarchy(src)
    human_contour = find_human_contour(contours, hierarchy)

    skeletonImplement = SkeletonImplement()
    humans = skeletonImplement.infer_skeletons(src)

    skinning = NearestNeighbourSkinning(src, humans[0], human_contour)

    # visualization
    for i in [166, 173]:
        cv2.circle(dst, skinning.triangle_centers[i], 1, (255, 0, 0))
        pt1, pt2, pt3 = [skinning.contour_vertex_positions[index] for index in skinning.triangle_vertex_indices[i]]
        draw_triangle(dst, pt1, pt2, pt3)
        cv2.circle(dst, skinning.body_part_positions[skinning.nearest_body_part_indices[i]], 3, (0, 0, 255),
                   thickness=3, lineType=8, shift=0)

    plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    plt.show()
