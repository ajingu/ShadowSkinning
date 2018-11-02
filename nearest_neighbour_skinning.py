import sys

import cv2
import matplotlib.pyplot as plt

from tf_pose.common import read_imgfile

from lib.draw import draw_circle
from lib.contour import find_human_contour
from lib.skeleton import SkeletonImplement, SkeletonTest
from lib.skinning import Skinning

if __name__ == '__main__':
    src = read_imgfile("./images/shadow.jpg", None, None)
    dst = src.copy()
    human_contour = find_human_contour(src)

    skeletonImplement = SkeletonImplement()
    human = skeletonImplement.infer_skeleton(src)
    human = skeletonImplement.remove_unused_joints(human)

    skeletonTest = SkeletonTest(human, human_contour, src.shape)
    skeletonTest.report()
    if not skeletonTest.is_reliable():
        print("This skeleton model is not reliable.")
        sys.exit(0)

    # skinning = Skinning(src, humans, human_contour)
    skinning = Skinning(src, human, human_contour, algorithm="nearest_neighbour_within_contour")

    # visualization
    for i in [100, 300, 500]:
        draw_circle(dst, skinning.contour_vertex_positions[i], (255, 0, 0))
        draw_circle(dst, skinning.body_part_positions[skinning.nearest_body_part_indices[i]])

    # cv2.imwrite("./images/nearest.png", dst)
    plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    plt.show()
