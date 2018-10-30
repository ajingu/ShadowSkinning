import cv2
import matplotlib.pyplot as plt

from tf_pose.common import read_imgfile

from lib.common import draw_circle
from lib.contour import find_human_contour
from lib.skeleton import SkeletonImplement
from lib.skinning import Skinning

if __name__ == '__main__':
    src = read_imgfile("./images/shadow.jpg", None, None)
    dst = src.copy()
    human_contour = find_human_contour(src)

    skeletonImplement = SkeletonImplement()
    humans = skeletonImplement.infer_skeletons(src)

    skinning = Skinning(src, humans[0], human_contour, algorithm="k_nearest_neighbour_within_contour")

    # visualization
    #for i in [100, 300, 500]:
    #    draw_circle(dst, skinning.contour_vertex_positions[i], (255, 0, 0))
    #    draw_circle(dst, skinning.body_part_positions[skinning.nearest_body_part_indices[i]])

    # cv2.imwrite("./images/nearest.png", dst)
    plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    plt.show()
