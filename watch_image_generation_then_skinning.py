import sys

import cv2
import matplotlib.pyplot as plt

from tf_pose.common import read_imgfile

from lib.draw import draw_circle
from lib.contour import find_human_contour
from lib.skeleton import SkeletonImplement, SkeletonTest
from lib.skinning import Skinning
from lib.watch import watch_image_generation

TARGET_DIRECTORY_PATH = "./images"


def run_skinning(image_path, frame_index):
    src = read_imgfile(image_path, None, None)

    try:
        dst = src.copy()
    except AttributeError:
        print(image_path + " is not found.")
        return

    print("Frame Index:", frame_index)

    human_contour = find_human_contour(src)

    skeleton_implement = SkeletonImplement()
    human = skeleton_implement.infer_skeleton(src)
    human = skeleton_implement.remove_unused_joints(human)

    skeleton_test = SkeletonTest(human, human_contour, src.shape)
    skeleton_test.report()
    if not skeleton_test.is_reliable():
        print("This skeleton model is not reliable.")
        sys.exit(0)

    # skinning = Skinning(src, humans, human_contour)
    skinning = Skinning(src, human, human_contour, algorithm="nearest_neighbour_within_contour")

    # visualization
    for i in [100, 300, 500]:
        draw_circle(dst, skinning.contour_vertex_positions[i], (255, 0, 0))
        draw_circle(dst, skinning.body_part_positions[skinning.nearest_body_part_indices[i]])

    plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == '__main__':
    watch_image_generation(run_skinning, TARGET_DIRECTORY_PATH)
